"""Query implementation for codebase search."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .schema import QueryResult
from .shared import EMBEDDER, QUERY_EMBED_PARAMS, SQLITE_DB

# Reciprocal Rank Fusion constant. 60 is the value used in the original RRF
# paper (Cormack et al., 2009) and the de-facto default in code/doc search;
# higher values flatten differences between top ranks, lower values sharpen
# them.
_RRF_K = 60


def _build_filter_clause(
    languages: list[str] | None,
    paths: list[str] | None,
) -> tuple[str, list[Any]]:
    """Build a SQL WHERE-clause fragment + params for language/path filters.

    Returns the fragment without the leading ``AND`` so callers can splice it
    after their own predicate (e.g. ``WHERE foo MATCH ? <fragment>``).
    """
    conditions: list[str] = []
    params: list[Any] = []
    if languages:
        placeholders = ",".join("?" for _ in languages)
        conditions.append(f"language IN ({placeholders})")
        params.extend(languages)
    if paths:
        path_clauses = " OR ".join("file_path GLOB ?" for _ in paths)
        conditions.append(f"({path_clauses})")
        params.extend(paths)
    fragment = (" AND " + " AND ".join(conditions)) if conditions else ""
    return fragment, params


def _knn_query(
    conn: sqlite3.Connection,
    embedding_bytes: bytes,
    k: int,
    languages: list[str] | None = None,
    paths: list[str] | None = None,
) -> list[tuple[Any, ...]]:
    """Run a vec0 KNN query, optionally constrained by language/path.

    When the only filter is a single language we use the vec0 partition key
    for an indexed search. Anything else falls through to a full scan with
    ``vec_distance_L2`` so we can apply arbitrary WHERE filters.
    """
    if paths or (languages and len(languages) > 1):
        return _full_scan_query(conn, embedding_bytes, k, 0, languages, paths)

    language = languages[0] if languages else None
    if language is not None:
        return conn.execute(
            """
            SELECT id, file_path, language, content, start_line, end_line, distance
            FROM code_chunks_vec
            WHERE embedding MATCH ? AND k = ? AND language = ?
            ORDER BY distance
            """,
            (embedding_bytes, k, language),
        ).fetchall()
    return conn.execute(
        """
        SELECT id, file_path, language, content, start_line, end_line, distance
        FROM code_chunks_vec
        WHERE embedding MATCH ? AND k = ?
        ORDER BY distance
        """,
        (embedding_bytes, k),
    ).fetchall()


def _full_scan_query(
    conn: sqlite3.Connection,
    embedding_bytes: bytes,
    limit: int,
    offset: int,
    languages: list[str] | None = None,
    paths: list[str] | None = None,
) -> list[tuple[Any, ...]]:
    """Full scan with SQL-level distance computation and filtering."""
    fragment, filter_params = _build_filter_clause(languages, paths)
    where = f"WHERE 1=1{fragment}" if fragment else ""
    params: list[Any] = [embedding_bytes, *filter_params, limit, offset]

    return conn.execute(
        f"""
        SELECT id, file_path, language, content, start_line, end_line,
               vec_distance_L2(embedding, ?) as distance
        FROM code_chunks_vec
        {where}
        ORDER BY distance
        LIMIT ? OFFSET ?
        """,
        params,
    ).fetchall()


def _escape_fts5_query(query: str) -> str:
    """Wrap a free-text query as an FTS5 phrase, escaping embedded quotes.

    Users type bare identifiers (``CvoFile``) or natural-language phrases. We
    treat the whole input as a single FTS5 phrase: this avoids leaking FTS5
    operator syntax (``AND``, ``NEAR``, ``-``, ``*``, parentheses) that would
    otherwise raise SQLITE_ERROR. A phrase miss simply contributes nothing to
    RRF — the vector side still ranks the result.
    """
    return '"' + query.replace('"', '""') + '"'


def _fts_query(
    conn: sqlite3.Connection,
    query: str,
    k: int,
    languages: list[str] | None = None,
    paths: list[str] | None = None,
) -> list[tuple[Any, ...]]:
    """Run a BM25 keyword query against the FTS5 sidecar, if present.

    Returns an empty list when the FTS5 table doesn't exist yet (project that
    was indexed before hybrid search shipped) — search degrades to vector-only
    rather than hard-failing.
    """
    fragment, filter_params = _build_filter_clause(languages, paths)
    fts_query = _escape_fts5_query(query)
    try:
        return conn.execute(
            f"""
            SELECT rowid AS id, file_path, language, content, start_line, end_line,
                   bm25(code_chunks_fts) AS rank
            FROM code_chunks_fts
            WHERE code_chunks_fts MATCH ?{fragment}
            ORDER BY rank
            LIMIT ?
            """,
            [fts_query, *filter_params, k],
        ).fetchall()
    except sqlite3.OperationalError:
        return []


def _rrf_merge(
    vec_rows: list[tuple[Any, ...]],
    fts_rows: list[tuple[Any, ...]],
    limit: int,
    offset: int,
) -> list[tuple[tuple[Any, ...], float]]:
    """Combine vector and FTS rankings via Reciprocal Rank Fusion.

    Both inputs are ordered best-first. Output is ``(row, rrf_score)`` pairs
    sorted by score descending, sliced to the requested window.

    FTS rows are inserted into the score map first so that when a literal-
    match chunk ties on score with a vec-only chunk (both at rank 1 in their
    respective retrievers, scoring exactly ``1/(K+1)``), the keyword match
    wins the stable-sort tie-break. For identifier-style queries this is
    almost always what the user wants.
    """
    scores: dict[Any, list[Any]] = {}
    for rank, row in enumerate(fts_rows, start=1):
        chunk_id = row[0]
        bucket = scores.setdefault(chunk_id, [row, 0.0])
        bucket[1] = bucket[1] + 1.0 / (_RRF_K + rank)
    for rank, row in enumerate(vec_rows, start=1):
        chunk_id = row[0]
        bucket = scores.setdefault(chunk_id, [row, 0.0])
        bucket[1] = bucket[1] + 1.0 / (_RRF_K + rank)

    ranked = sorted(scores.values(), key=lambda item: -item[1])
    window = ranked[offset : offset + limit]
    return [(item[0], item[1]) for item in window]


async def query_codebase(
    query: str,
    target_sqlite_db_path: Path,
    env: Any,
    limit: int = 10,
    offset: int = 0,
    languages: list[str] | None = None,
    paths: list[str] | None = None,
) -> list[QueryResult]:
    """Hybrid search: vec0 KNN + FTS5 BM25, fused with Reciprocal Rank Fusion.

    Vector search alone ranks short identifier queries poorly (``CvoFile``
    matches "looks like a Cvo widget" before chunks that literally contain
    the token). The FTS5 sidecar restores that signal; RRF combines both
    rankings without needing score normalization.
    """
    if not target_sqlite_db_path.exists():
        raise RuntimeError(
            f"Index database not found at {target_sqlite_db_path}. "
            "Please run a query with refresh_index=True first."
        )

    db = env.get_context(SQLITE_DB)
    embedder = env.get_context(EMBEDDER)
    query_params = env.get_context(QUERY_EMBED_PARAMS)

    query_embedding = await embedder.embed(query, **query_params)
    embedding_bytes = query_embedding.astype("float32").tobytes()

    # Pull a wider candidate pool from each retriever than the user asked for,
    # so RRF has room to surface chunks that neither side ranked in the top-N.
    fetch_k = max(50, (limit + offset) * 5)

    with db.readonly() as conn:
        vec_rows = _knn_query(conn, embedding_bytes, fetch_k, languages, paths)
        fts_rows = _fts_query(conn, query, fetch_k, languages, paths)

    merged = _rrf_merge(vec_rows, fts_rows, limit, offset)

    return [
        QueryResult(
            file_path=row[1],
            language=row[2],
            content=row[3],
            start_line=row[4],
            end_line=row[5],
            score=score,
        )
        for row, score in merged
    ]
