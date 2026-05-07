"""Dart source chunker — splits at top-level declaration boundaries."""

from __future__ import annotations

from pathlib import Path

from cocoindex.ops.text import RecursiveSplitter as _RecursiveSplitter

from cocoindex_code.chunking import Chunk, TextPosition

# Mirror indexer.py chunk constants so sub-chunk sizing is consistent.
_CHUNK_SIZE = 1000
_MIN_CHUNK_SIZE = 250
_CHUNK_OVERLAP = 150
# Classes larger than this fall back to RecursiveSplitter to avoid truncation
# at the embedding model's token limit (~512 tokens for the default model).
_MAX_CHUNK_CHARS = 1500

# Dart type-declaration keywords that can start a single-line top-level
# declaration (Seam 2).  Excludes `const`/`var`/`late`/etc. which introduce
# variables rather than type declarations.
_DECL_KEYWORDS = frozenset({
    "class", "abstract", "final", "sealed", "base", "interface",
    "mixin", "enum", "extension", "typedef",
})

_splitter = _RecursiveSplitter()


class _Parser:
    """Stateful Dart brace-depth tracker.

    Handles ``//`` line comments, ``/* */`` block comments, single-line string
    literals, and triple-quoted multiline strings so that braces inside those
    constructs are not counted toward the depth.
    """

    __slots__ = ("depth", "_in_block_comment", "_in_multiline_string", "_string_delim")

    def __init__(self) -> None:
        self.depth = 0
        self._in_block_comment = False
        self._in_multiline_string = False
        self._string_delim = ""

    def feed(self, line: str) -> tuple[int, bool]:
        """Process one line; return ``(net_brace_change, has_real_open_brace)``.

        ``has_real_open_brace`` is ``True`` when ``{`` appears in code context
        (outside comments and string literals).  Updates internal state for
        cross-line constructs (block comments and triple-quoted strings).
        """
        net = 0
        has_brace = False
        i = 0
        n = len(line)

        while i < n:
            # ── inside multiline string ──────────────────────────────────────
            if self._in_multiline_string:
                delim_len = len(self._string_delim)
                if line[i : i + delim_len] == self._string_delim:
                    self._in_multiline_string = False
                    i += delim_len
                else:
                    i += 1
                continue

            # ── inside block comment ─────────────────────────────────────────
            if self._in_block_comment:
                if line[i : i + 2] == "*/":
                    self._in_block_comment = False
                    i += 2
                else:
                    i += 1
                continue

            # ── line comment ends processing ─────────────────────────────────
            if line[i : i + 2] == "//":
                break

            # ── block comment start ──────────────────────────────────────────
            if line[i : i + 2] == "/*":
                self._in_block_comment = True
                i += 2
                continue

            c = line[i]

            # ── string literal ───────────────────────────────────────────────
            if c in ('"', "'"):
                triple = line[i : i + 3]
                if triple in ('"""', "'''"):
                    end = line.find(triple, i + 3)
                    if end >= 0:
                        i = end + 3  # entire triple-string on this line
                    else:
                        self._in_multiline_string = True
                        self._string_delim = triple
                        i += 3
                else:
                    # Single-line string: scan to matching close quote.
                    i += 1
                    while i < n:
                        if line[i] == "\\" and i + 1 < n:
                            i += 2
                            continue
                        if line[i] == c:
                            i += 1
                            break
                        i += 1
                continue

            # ── brace counting ───────────────────────────────────────────────
            if c == "{":
                net += 1
                has_brace = True
            elif c == "}":
                net -= 1

            i += 1

        return net, has_brace


def _pos(line: int) -> TextPosition:
    return TextPosition(byte_offset=0, char_offset=0, line=line, column=0)


def dart_chunker(_path: Path, content: str) -> tuple[str | None, list[Chunk]]:
    """Split Dart source at top-level class/function boundaries.

    Two seam types:
    - Seam 1: brace depth drops from >0 back to 0 (multi-line block closes).
    - Seam 2: single-line balanced declaration — depth stays at 0, a real ``{``
      appears in code context, and the line begins with a type-declaration
      keyword (``class``, ``enum``, ``mixin``, etc.).

    Chunks exceeding ``_MAX_CHUNK_CHARS`` are split further by a
    ``RecursiveSplitter`` fallback so they stay within embedding-model limits.
    """
    lines = content.splitlines()
    if not lines:
        return "dart", []

    parser = _Parser()
    boundaries: list[int] = [0]

    for i, line in enumerate(lines):
        prev_depth = parser.depth
        net, has_brace = parser.feed(line)
        parser.depth = max(0, parser.depth + net)

        seam = False
        if prev_depth > 0 and parser.depth == 0:
            seam = True  # Seam 1: multi-line block just closed
        elif prev_depth == 0 and parser.depth == 0 and net == 0 and has_brace:
            # Seam 2: single-line balanced type declaration at top level
            words = line.split()
            if words and words[0] in _DECL_KEYWORDS:
                seam = True

        if seam:
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and j not in boundaries:
                boundaries.append(j)

    chunks: list[Chunk] = []
    for k, start in enumerate(boundaries):
        end = boundaries[k + 1] if k + 1 < len(boundaries) else len(lines)
        text = "\n".join(lines[start:end]).strip()
        if not text:
            continue

        if len(text) > _MAX_CHUNK_CHARS:
            for sub in _splitter.split(
                "\n".join(lines[start:end]),
                chunk_size=_CHUNK_SIZE,
                min_chunk_size=_MIN_CHUNK_SIZE,
                chunk_overlap=_CHUNK_OVERLAP,
                language="dart",
            ):
                if sub.text.strip():
                    chunks.append(
                        Chunk(
                            text=sub.text,
                            start=_pos(start + sub.start.line),
                            end=_pos(start + sub.end.line),
                        )
                    )
        else:
            chunks.append(Chunk(text=text, start=_pos(start + 1), end=_pos(end)))

    return "dart", chunks or [Chunk(text=content.strip(), start=_pos(1), end=_pos(len(lines)))]


__all__ = ["dart_chunker"]
