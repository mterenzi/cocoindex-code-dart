"""Unit tests for the Dart chunker."""

from __future__ import annotations

from pathlib import Path

from cocoindex_code.chunkers.dart import _MAX_CHUNK_CHARS, _Parser, dart_chunker

_PATH = Path("test.dart")


# ---------------------------------------------------------------------------
# _Parser.feed — brace counting
# ---------------------------------------------------------------------------


def test_parser_open_brace() -> None:
    p = _Parser()
    net, has_brace = p.feed("class Foo {")
    assert net == 1
    assert has_brace is True


def test_parser_close_brace() -> None:
    p = _Parser()
    net, has_brace = p.feed("}")
    assert net == -1
    assert has_brace is False


def test_parser_balanced() -> None:
    p = _Parser()
    net, _ = p.feed("void f() {}")
    assert net == 0


def test_parser_ignores_line_comment() -> None:
    p = _Parser()
    net, has_brace = p.feed("  // { this is a comment")
    assert net == 0
    assert has_brace is False


def test_parser_stops_at_line_comment() -> None:
    p = _Parser()
    net, has_brace = p.feed("int x = 1; // {")
    assert net == 0
    assert has_brace is False


def test_parser_ignores_block_comment_braces() -> None:
    p = _Parser()
    net, has_brace = p.feed("/* { block comment } */")
    assert net == 0
    assert has_brace is False


def test_parser_block_comment_spans_lines() -> None:
    p = _Parser()
    p.feed("/* open comment {")
    net, has_brace = p.feed("  still in comment }")
    assert net == 0
    assert has_brace is False
    p.feed("*/")
    # After closing, braces should count again.
    net2, has_brace2 = p.feed("class Foo {")
    assert net2 == 1
    assert has_brace2 is True


def test_parser_ignores_single_quoted_string_braces() -> None:
    p = _Parser()
    net, has_brace = p.feed("const x = '{ not a brace }';")
    assert net == 0
    assert has_brace is False


def test_parser_ignores_double_quoted_string_braces() -> None:
    p = _Parser()
    net, has_brace = p.feed('const x = "{ not a brace }";')
    assert net == 0
    assert has_brace is False


def test_parser_ignores_triple_quoted_string_braces() -> None:
    p = _Parser()
    net, has_brace = p.feed('const x = """{ not a brace }""";')
    assert net == 0
    assert has_brace is False


def test_parser_multiline_triple_string_spans_lines() -> None:
    p = _Parser()
    p.feed('const msg = """')
    net, has_brace = p.feed("  line with { brace }")
    assert net == 0
    assert has_brace is False
    p.feed('""";')
    net2, has_brace2 = p.feed("class Foo {")
    assert net2 == 1
    assert has_brace2 is True


def test_parser_escaped_quote_in_string() -> None:
    p = _Parser()
    net, has_brace = p.feed(r"const x = 'it\'s { fine }';")
    assert net == 0
    assert has_brace is False


# ---------------------------------------------------------------------------
# dart_chunker — basic splitting
# ---------------------------------------------------------------------------


def test_empty_content() -> None:
    lang, chunks = dart_chunker(_PATH, "")
    assert lang == "dart"
    assert chunks == []


def test_single_class() -> None:
    src = "class Foo {\n  int x = 1;\n}\n"
    lang, chunks = dart_chunker(_PATH, src)
    assert lang == "dart"
    assert len(chunks) == 1
    assert "class Foo" in chunks[0].text


def test_two_classes_split() -> None:
    src = (
        "class Foo {\n  int x = 1;\n}\n"
        "\n"
        "class Bar {\n  String y = 'hi';\n}\n"
    )
    lang, chunks = dart_chunker(_PATH, src)
    assert lang == "dart"
    assert len(chunks) == 2
    assert "class Foo" in chunks[0].text
    assert "class Bar" in chunks[1].text


def test_imports_stay_with_first_chunk() -> None:
    src = (
        "import 'dart:core';\n"
        "import 'package:flutter/material.dart';\n"
        "\n"
        "class MyApp extends StatelessWidget {\n"
        "  @override\n"
        "  Widget build(BuildContext context) => Container();\n"
        "}\n"
    )
    lang, chunks = dart_chunker(_PATH, src)
    assert lang == "dart"
    assert len(chunks) == 1
    assert "import" in chunks[0].text
    assert "class MyApp" in chunks[0].text


def test_imports_then_two_classes() -> None:
    src = (
        "import 'dart:core';\n"
        "\n"
        "class A {\n  void run() {}\n}\n"
        "\n"
        "class B {\n  void stop() {}\n}\n"
    )
    lang, chunks = dart_chunker(_PATH, src)
    assert lang == "dart"
    assert len(chunks) == 2
    assert "import" in chunks[0].text
    assert "class A" in chunks[0].text
    assert "class B" in chunks[1].text


def test_annotation_stays_with_class() -> None:
    src = (
        "class A {\n  void go() {}\n}\n"
        "\n"
        "@override\n"
        "class B {\n  void stop() {}\n}\n"
    )
    lang, chunks = dart_chunker(_PATH, src)
    assert lang == "dart"
    assert len(chunks) == 2
    assert "@override" in chunks[1].text
    assert "class B" in chunks[1].text


def test_arrow_function_file_is_single_chunk() -> None:
    src = (
        "String greet(String name) => 'Hello, $name!';\n"
        "int add(int a, int b) => a + b;\n"
    )
    lang, chunks = dart_chunker(_PATH, src)
    assert lang == "dart"
    assert len(chunks) == 1


def test_line_numbers_are_1_based() -> None:
    src = "class Foo {\n  int x = 0;\n}\n"
    _, chunks = dart_chunker(_PATH, src)
    assert chunks[0].start.line == 1


def test_second_chunk_line_number() -> None:
    src = (
        "class A {\n  void go() {}\n}\n"   # lines 1-3
        "\n"                                # line 4
        "class B {\n  void stop() {}\n}\n"  # lines 5-7
    )
    _, chunks = dart_chunker(_PATH, src)
    assert len(chunks) == 2
    assert chunks[1].start.line == 5


def test_enum_splits_correctly() -> None:
    src = (
        "enum Color { red, green, blue }\n"
        "\n"
        "class Painter {\n  Color c = Color.red;\n}\n"
    )
    lang, chunks = dart_chunker(_PATH, src)
    assert lang == "dart"
    assert len(chunks) == 2
    assert "enum Color" in chunks[0].text
    assert "class Painter" in chunks[1].text


def test_language_always_dart() -> None:
    src = "class X {}\n"
    lang, _ = dart_chunker(_PATH, src)
    assert lang == "dart"


# ---------------------------------------------------------------------------
# Smarter Seam 2 — keyword gating
# ---------------------------------------------------------------------------


def test_map_literal_const_does_not_split() -> None:
    # `const config = {'k': 'v'};` has balanced {} but starts with `const`,
    # not a type-declaration keyword — Seam 2 must not fire.
    src = (
        "const config = {'k': 'v'};\n"
        "\n"
        "class Foo {\n  void run() {}\n}\n"
    )
    _, chunks = dart_chunker(_PATH, src)
    # imports + const + class all in one chunk (no seam at const line)
    assert len(chunks) == 1
    assert "config" in chunks[0].text
    assert "class Foo" in chunks[0].text


def test_block_comment_braces_do_not_split() -> None:
    src = (
        "/* A class that manages { state } */\n"
        "class Foo {\n  int x = 0;\n}\n"
        "\n"
        "class Bar {\n  int y = 0;\n}\n"
    )
    _, chunks = dart_chunker(_PATH, src)
    assert len(chunks) == 2
    assert "class Foo" in chunks[0].text
    assert "class Bar" in chunks[1].text


def test_string_braces_do_not_affect_seam1() -> None:
    # String with { inside a class body should not corrupt depth tracking.
    src = (
        "class Formatter {\n"
        "  String fmt(int n) => 'value: {$n}';\n"
        "}\n"
        "\n"
        "class Renderer {\n"
        "  void render() {}\n"
        "}\n"
    )
    _, chunks = dart_chunker(_PATH, src)
    assert len(chunks) == 2
    assert "Formatter" in chunks[0].text
    assert "Renderer" in chunks[1].text


# ---------------------------------------------------------------------------
# Large chunk capping
# ---------------------------------------------------------------------------


def test_large_class_is_split() -> None:
    # Build a class whose text length exceeds _MAX_CHUNK_CHARS.
    methods = "\n".join(f"  void method{i}() {{ return; }}" for i in range(80))
    src = f"class BigClass {{\n{methods}\n}}\n"
    assert len(src) > _MAX_CHUNK_CHARS, "test fixture is too small"

    _, chunks = dart_chunker(_PATH, src)
    assert len(chunks) > 1, "oversized class should produce multiple chunks"
    # Every chunk should contain Dart content
    assert all(c.text.strip() for c in chunks)


def test_small_class_is_not_split() -> None:
    src = "class Small {\n  int x = 0;\n}\n"
    assert len(src) < _MAX_CHUNK_CHARS
    _, chunks = dart_chunker(_PATH, src)
    assert len(chunks) == 1
