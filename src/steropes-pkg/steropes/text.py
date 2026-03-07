"""Text wrapping utilities that preserve markdown structure."""

import re


def wrap_markdown(text: str, width: int = 80) -> str:
    """Wrap text at width, preserving markdown structure.

    - Code blocks (```) are not wrapped
    - Inline code (`...`) within a line is preserved
    - List items wrap with a hanging indent; continuation lines (indented to
      at least the marker end) are wrapped with the same indent
    - Blockquotes are preserved with the correct continuation prefix
    - Regular paragraphs preserve their leading indent on continuation lines
    """
    if not text:
        return text

    lines = text.split('\n')
    result = []
    in_code_block = False
    # Number of leading spaces that mark a list continuation line; None when
    # we are not inside a list item.
    list_continuation_indent: int | None = None

    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        # Code block fence toggles verbatim mode
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            list_continuation_indent = None
            result.append(line)
            continue

        if in_code_block:
            result.append(line)
            continue

        # Blank line resets list state
        if not stripped:
            list_continuation_indent = None
            result.append(line)
            continue

        # New list item
        list_match = re.match(r'^(\s*)([-*+]|\d+\.)\s+', line)
        if list_match:
            marker_end = list_match.end()
            list_continuation_indent = marker_end
            content = line[marker_end:]
            wrapped_content = _wrap_line_preserving_inline(content, width - marker_end)
            wrapped_lines = wrapped_content.split('\n')
            result.append(line[:marker_end] + wrapped_lines[0])
            for wrapped_line in wrapped_lines[1:]:
                result.append(' ' * marker_end + wrapped_line)
            continue

        # List continuation line: indented to at least the marker position
        if list_continuation_indent is not None and indent >= list_continuation_indent:
            cont_prefix = ' ' * indent
            wrapped_content = _wrap_line_preserving_inline(stripped, width - indent)
            wrapped_lines = wrapped_content.split('\n')
            result.append(cont_prefix + wrapped_lines[0])
            for wrapped_line in wrapped_lines[1:]:
                result.append(cont_prefix + wrapped_line)
            continue

        # Not a list continuation — reset list state
        list_continuation_indent = None

        # Blockquote
        if stripped.startswith('>'):
            quote_match = re.match(r'^(\s*)(>+\s?)', line)
            if quote_match:
                full_prefix = quote_match.group(1) + quote_match.group(2)
                content = line[len(full_prefix):]
                wrapped_content = _wrap_line_preserving_inline(content, width - len(full_prefix))
                wrapped_lines = wrapped_content.split('\n')
                result.append(full_prefix + wrapped_lines[0])
                for wrapped_line in wrapped_lines[1:]:
                    result.append(full_prefix + wrapped_line)
                continue

        # Regular paragraph — preserve leading indent on all continuation lines
        cont_prefix = ' ' * indent
        wrapped_content = _wrap_line_preserving_inline(stripped, width - indent)
        wrapped_lines = wrapped_content.split('\n')
        result.append(cont_prefix + wrapped_lines[0])
        for wrapped_line in wrapped_lines[1:]:
            result.append(cont_prefix + wrapped_line)

    return '\n'.join(result)


def _wrap_line_preserving_inline(text: str, width: int) -> str:
    """Wrap a line while preserving inline code spans."""
    if len(text) <= width:
        return text

    # Split on inline code to preserve it
    parts = re.split(r'(`[^`]+`)', text)
    lines = []
    current_line = ''

    for part in parts:
        if not part:
            continue
        if part.startswith('`') and part.endswith('`'):
            # Inline code - add to current line if it fits, else start new line
            if current_line and len(current_line) + 1 + len(part) > width:
                lines.append(current_line.rstrip())
                current_line = part
            else:
                if current_line:
                    current_line += ' '
                current_line += part
        else:
            # Regular text - wrap it
            words = part.split(' ')
            for word in words:
                if not word:
                    continue
                if current_line and len(current_line) + 1 + len(word) > width:
                    lines.append(current_line.rstrip())
                    current_line = word
                else:
                    if current_line:
                        current_line += ' '
                    current_line += word

    if current_line:
        lines.append(current_line.rstrip())

    return '\n'.join(lines)
