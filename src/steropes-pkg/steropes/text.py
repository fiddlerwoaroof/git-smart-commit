"""Text wrapping utilities that preserve markdown structure via AST."""

from __future__ import annotations

import re
from typing import Optional

from mistletoe import Document
from mistletoe.block_token import Paragraph, Quote, ListItem
from mistletoe.markdown_renderer import MarkdownRenderer
from mistletoe.span_token import InlineCode, LineBreak, RawText


def wrap_markdown(
    text: str,
    width: int = 80,
    *,
    soft_width: Optional[int] = None,
) -> str:
    """Wrap prose in *text* at *width*, preserving markdown structure.

    Parses the text into a markdown AST (via mistletoe), rewraps only
    ``Paragraph`` nodes, and serializes back.  Structural elements — code
    blocks, lists, headings, blockquotes — are preserved exactly.

    *soft_width* (default ``width - 5``) is the preferred break point;
    lines break at or before *width* (hard limit) but prefer *soft_width*
    to reduce ragged-right edges.
    """
    if not text:
        return text

    if soft_width is None:
        soft_width = max(width - 5, width * 3 // 4)

    with MarkdownRenderer() as renderer:
        doc = Document(text)
        _rewrap_tree(doc, renderer, width, soft_width, prefix_len=0)
        result = renderer.render(doc)

    # mistletoe always adds a trailing newline; match original
    if not text.endswith("\n") and result.endswith("\n"):
        result = result[:-1]

    return result


def _rewrap_tree(
    node, renderer: MarkdownRenderer, width: int, soft_width: int, prefix_len: int
) -> None:
    """Recursively walk the AST and rewrap Paragraph nodes in place."""
    if not hasattr(node, "children") or node.children is None:
        return

    for child in node.children:
        if isinstance(child, Paragraph):
            _rewrap_paragraph(child, renderer, width, soft_width, prefix_len)
        elif isinstance(child, Quote):
            # Blockquote adds "> " (2 chars) per nesting level
            _rewrap_tree(child, renderer, width, soft_width, prefix_len + 2)
        elif isinstance(child, ListItem):
            # List items have a marker + space; estimate ~2-4 chars
            _rewrap_tree(child, renderer, width, soft_width, prefix_len + 2)
        else:
            _rewrap_tree(child, renderer, width, soft_width, prefix_len)


def _rewrap_paragraph(
    para: Paragraph,
    renderer: MarkdownRenderer,
    width: int,
    soft_width: int,
    prefix_len: int,
) -> None:
    """Flatten a Paragraph's inline children, rewrap, and replace children."""
    # Render each inline token to its markdown representation, collecting
    # them as atomic "tokens" for the wrapping algorithm.
    tokens: list[tuple[str, bool]] = []  # (text, is_atomic)

    for child in para.children:
        if isinstance(child, LineBreak) and child.soft:
            # Soft break = space (we're reflowing)
            tokens.append((" ", False))
        elif isinstance(child, LineBreak):
            # Hard break — preserve
            tokens.append(("\\\n", True))
        elif isinstance(child, RawText):
            tokens.append((child.content, False))
        else:
            # Inline code, emphasis, links, etc. — render to markdown and
            # treat as an atomic token that must not be split.
            rendered = renderer.render(child).rstrip("\n")
            tokens.append((rendered, True))

    # Join into a flat string, then split into word-level tokens
    word_tokens = _tokenize_for_wrap(tokens)

    if not word_tokens:
        return

    # Wrap
    avail_hard = max(width - prefix_len, 20)
    avail_soft = max(soft_width - prefix_len, 15)
    lines = _wrap_tokens(word_tokens, avail_hard, avail_soft)

    # Rebuild paragraph children from wrapped lines
    new_children: list[RawText | LineBreak] = []
    for i, line in enumerate(lines):
        if i > 0:
            new_children.append(_make_soft_linebreak())
        new_children.append(RawText(line))

    para.children = new_children


def _tokenize_for_wrap(
    tokens: list[tuple[str, bool]],
) -> list[str]:
    """Convert inline token pairs into a flat list of word-level strings.

    Atomic tokens (inline code, emphasis, etc.) are kept whole.
    Non-atomic text is split on whitespace into individual words.
    """
    result: list[str] = []
    for text, is_atomic in tokens:
        if is_atomic:
            if text == "\\\n":
                # Hard line break — emit as-is
                result.append(text)
            else:
                result.append(text)
        else:
            for word in text.split():
                result.append(word)
    return result


def _wrap_tokens(
    tokens: list[str], hard_width: int, soft_width: int
) -> list[str]:
    """Distribute *tokens* across lines respecting soft/hard width limits.

    Returns a list of lines (strings).  Each line is at most *hard_width*
    characters.  Lines are preferentially broken at *soft_width* to
    reduce ragged-right edges.
    """
    lines: list[str] = []
    current = ""

    for token in tokens:
        # Hard line break
        if token == "\\\n":
            lines.append(current)
            current = ""
            continue

        if not current:
            current = token
            continue

        candidate = current + " " + token
        cand_len = len(candidate)

        if cand_len <= soft_width:
            current = candidate
        elif cand_len <= hard_width:
            # Between soft and hard — break if current line is already
            # reasonably full (reduces ragged right)
            if len(current) >= soft_width * 2 // 3:
                lines.append(current)
                current = token
            else:
                current = candidate
        else:
            # Exceeds hard width — must break
            lines.append(current)
            current = token

    if current:
        lines.append(current)

    return lines


def _make_soft_linebreak() -> LineBreak:
    """Create a soft ``LineBreak`` token."""
    m = re.match(r"( *|\\)\n", "\n")
    assert m is not None
    return LineBreak(m)
