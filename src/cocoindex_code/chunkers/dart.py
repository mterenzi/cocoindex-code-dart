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


def _find_method_boundaries(lines: list[str]) -> list[int]:
    """Find line indices where each method/inner block begins inside a class.

    Anchored at the body level (depth 1): when depth drops from >1 back to 1,
    the next non-empty line is treated as the start of a new method.  The
    returned list always starts with 0 — the class header (and anything
    preceding the first method body) is grouped with the first method.

    Returns ``[0]`` if no interior method boundaries are found (e.g. the chunk
    is a single function with no nested blocks).
    """
    parser = _Parser()
    boundaries = [0]
    body_started = False

    for i, line in enumerate(lines):
        prev_depth = parser.depth
        net, _ = parser.feed(line)
        parser.depth = max(0, parser.depth + net)

        if not body_started and parser.depth >= 1:
            body_started = True
            continue

        if body_started and prev_depth > 1 and parser.depth == 1:
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and j not in boundaries:
                tail = "\n".join(lines[j:]).strip()
                # Avoid emitting a tail chunk that is just the class's closing brace.
                if tail and tail != "}":
                    boundaries.append(j)

    return boundaries


def _splitter_split(lines: list[str], offset: int) -> list[Chunk]:
    """Run RecursiveSplitter on the joined lines, returning Chunks anchored
    at ``offset`` (0-based line index of ``lines[0]`` in the original file).
    """
    raw = "\n".join(lines)
    return [
        Chunk(
            text=sub.text,
            start=_pos(offset + sub.start.line),
            end=_pos(offset + sub.end.line),
        )
        for sub in _splitter.split(
            raw,
            chunk_size=_CHUNK_SIZE,
            min_chunk_size=_MIN_CHUNK_SIZE,
            chunk_overlap=_CHUNK_OVERLAP,
            language="dart",
        )
        if sub.text.strip()
    ]


def _split_oversized(lines: list[str], offset: int) -> list[Chunk]:
    """Split an oversized class/block by method boundaries, packing consecutive
    methods up to ``_CHUNK_SIZE``.  Falls back to ``RecursiveSplitter`` when
    no method seams are found, and applies it post-hoc to any single packed
    chunk that's still too large (e.g. a 5000-char method).
    """
    boundaries = _find_method_boundaries(lines)
    if len(boundaries) <= 1:
        return _splitter_split(lines, offset)

    # Per-method byte sizes (including newlines) for packing decisions.
    sizes: list[int] = []
    for k, start in enumerate(boundaries):
        end = boundaries[k + 1] if k + 1 < len(boundaries) else len(lines)
        sizes.append(sum(len(lines[m]) + 1 for m in range(start, end)))

    # Greedily pack adjacent methods up to _CHUNK_SIZE.
    packed: list[Chunk] = []
    pack_start = 0
    pack_size = 0
    for idx in range(len(boundaries)):
        if pack_size > 0 and pack_size + sizes[idx] > _CHUNK_SIZE:
            start = boundaries[pack_start]
            end = boundaries[idx]
            text = "\n".join(lines[start:end]).strip()
            if text:
                packed.append(Chunk(text=text, start=_pos(offset + start + 1), end=_pos(offset + end)))
            pack_start = idx
            pack_size = sizes[idx]
        else:
            pack_size += sizes[idx]

    start = boundaries[pack_start]
    text = "\n".join(lines[start:]).strip()
    if text:
        packed.append(Chunk(text=text, start=_pos(offset + start + 1), end=_pos(offset + len(lines))))

    # Post-pass: any chunk still too large (a single huge method) gets the splitter.
    final: list[Chunk] = []
    for c in packed:
        if len(c.text) > _MAX_CHUNK_CHARS:
            sub_offset = c.start.line - 1
            final.extend(_splitter_split(c.text.splitlines(), sub_offset))
        else:
            final.append(c)
    return final


def dart_chunker(_path: Path, content: str) -> tuple[str | None, list[Chunk]]:
    """Split Dart source at top-level class/function boundaries.

    Two seam types:
    - Seam 1: brace depth drops from >0 back to 0 (multi-line block closes).
    - Seam 2: single-line balanced declaration — depth stays at 0, a real ``{``
      appears in code context, and the line begins with a type-declaration
      keyword (``class``, ``enum``, ``mixin``, etc.).

    Chunks exceeding ``_MAX_CHUNK_CHARS`` are recursively split at method
    boundaries (depth-2→1 seams inside the class body) and packed up to
    ``_CHUNK_SIZE``, with a ``RecursiveSplitter`` fallback for blocks with
    no internal method boundaries.
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
            chunks.extend(_split_oversized(lines[start:end], start))
        else:
            chunks.append(Chunk(text=text, start=_pos(start + 1), end=_pos(end)))

    return "dart", chunks or [Chunk(text=content.strip(), start=_pos(1), end=_pos(len(lines)))]


__all__ = ["dart_chunker"]
