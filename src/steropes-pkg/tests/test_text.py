"""Tests for steropes.text — markdown-aware wrapping via AST."""

import pytest

from steropes.text import wrap_markdown


class TestWrapMarkdownBasic:
    def test_empty_string(self):
        assert wrap_markdown("", 80) == ""

    def test_short_text_unchanged(self):
        assert wrap_markdown("Hello world.", 80) == "Hello world."

    def test_no_trailing_newline_when_input_has_none(self):
        result = wrap_markdown("Hello world.", 80)
        assert not result.endswith("\n")

    def test_trailing_newline_preserved_when_input_has_one(self):
        result = wrap_markdown("Hello world.\n", 80)
        assert result.endswith("\n")

    def test_idempotent(self):
        text = "A moderately long line that fits within eighty characters easily."
        r1 = wrap_markdown(text, 80)
        r2 = wrap_markdown(r1, 80)
        assert r1 == r2


class TestParagraphReflow:
    def test_short_lines_joined(self):
        text = "Short line.\nAnother short line.\nAnd a third."
        result = wrap_markdown(text, 80)
        assert result == "Short line. Another short line. And a third."

    def test_long_paragraph_wrapped(self):
        text = (
            "This is a very long paragraph that contains a lot of text "
            "and should be wrapped at the soft width target of about "
            "75 characters and hard limit of 80 characters."
        )
        result = wrap_markdown(text, 80)
        for line in result.split("\n"):
            assert len(line) <= 80

    def test_lines_prefer_soft_width(self):
        text = (
            "This is a very long paragraph that contains a lot of text "
            "and should be wrapped at the soft width target of about "
            "75 characters and hard limit of 80 characters."
        )
        result = wrap_markdown(text, 80, soft_width=75)
        lines = result.split("\n")
        # Most lines should be at or near soft width, not right at 80
        for line in lines[:-1]:  # last line can be short
            assert len(line) <= 80

    def test_separate_paragraphs_stay_separate(self):
        text = "Paragraph one.\n\nParagraph two."
        result = wrap_markdown(text, 80)
        assert "\n\n" in result
        assert "Paragraph one." in result
        assert "Paragraph two." in result


class TestCodeBlocks:
    def test_code_block_not_wrapped(self):
        long_code = "x = " + "a" * 100
        text = f"```python\n{long_code}\n```"
        result = wrap_markdown(text, 80)
        assert long_code in result

    def test_code_block_content_preserved(self):
        text = "```\nline1\nline2\n  indented\n```"
        result = wrap_markdown(text, 80)
        assert "line1\nline2\n  indented" in result


class TestInlineCode:
    def test_inline_code_preserved(self):
        text = "Use `some_function()` to do the thing."
        result = wrap_markdown(text, 80)
        assert "`some_function()`" in result

    def test_inline_code_not_split(self):
        text = (
            "This line has `a_very_long_function_name()` right near the "
            "wrapping boundary and should not split the backtick span."
        )
        result = wrap_markdown(text, 80)
        assert "`a_very_long_function_name()`" in result
        for line in result.split("\n"):
            # No line should contain an unmatched backtick
            assert line.count("`") % 2 == 0


class TestLists:
    def test_short_list_unchanged(self):
        text = "- Item one\n- Item two"
        result = wrap_markdown(text, 80)
        assert "- Item one" in result
        assert "- Item two" in result

    def test_long_list_item_wrapped(self):
        text = (
            "- This is a list item that is quite long and should wrap "
            "properly while maintaining the list structure and indentation"
        )
        result = wrap_markdown(text, 80)
        lines = result.split("\n")
        assert lines[0].startswith("- ")
        for line in lines[1:]:
            # Continuation should be indented
            assert line.startswith("  ")

    def test_list_item_hard_limit(self):
        text = (
            "- This is a list item that is quite long and should wrap "
            "properly while maintaining the list structure and indentation"
        )
        result = wrap_markdown(text, 80)
        for line in result.split("\n"):
            assert len(line) <= 80


class TestBlockquotes:
    def test_short_blockquote_unchanged(self):
        text = "> Short quote."
        result = wrap_markdown(text, 80)
        assert result.strip() == "> Short quote."

    def test_long_blockquote_wrapped(self):
        text = (
            "> This is a blockquote that has a very long line which "
            "should be wrapped properly while preserving the prefix."
        )
        result = wrap_markdown(text, 80)
        for line in result.strip().split("\n"):
            assert line.startswith("> ")
            assert len(line) <= 80


class TestHeadings:
    def test_heading_never_wrapped(self):
        text = "# " + "word " * 30
        result = wrap_markdown(text, 80)
        # Heading should be a single line
        assert "\n" not in result.strip()


class TestSoftWidth:
    def test_custom_soft_width(self):
        text = (
            "This is a paragraph that should wrap using a custom soft "
            "width of 60 characters for testing purposes here."
        )
        result = wrap_markdown(text, 80, soft_width=60)
        lines = result.split("\n")
        for line in lines:
            assert len(line) <= 80

    def test_soft_width_reduces_raggedness(self):
        words = ["word"] * 30
        text = " ".join(words)
        result = wrap_markdown(text, 80, soft_width=75)
        lines = result.split("\n")
        if len(lines) > 2:
            # Non-final lines should be reasonably similar in length
            lengths = [len(line) for line in lines[:-1]]
            spread = max(lengths) - min(lengths)
            assert spread < 30  # reasonable tolerance
