"""Tests for steropes.ui — ANSI terminal output utilities."""

import io
import sys
from unittest.mock import patch

from steropes.ui import ANSI_BOLD, ANSI_DIM, ANSI_RESET, ansi, log, log_reasoning


class TestAnsi:
    def test_non_tty_no_codes_returns_plain(self):
        mock_stdout = io.StringIO()
        mock_stdout.isatty = lambda: False  # type: ignore[attr-defined]
        with patch("sys.stdout", mock_stdout):
            result = ansi("hello")
            assert result == "hello"

    def test_non_tty_returns_plain(self):
        mock_stdout = io.StringIO()
        mock_stdout.isatty = lambda: False  # type: ignore[attr-defined]
        with patch("sys.stdout", mock_stdout):
            result = ansi("hello", ANSI_BOLD)
            assert result == "hello"
            assert "\033" not in result

    def test_tty_applies_codes(self):
        mock_stdout = io.StringIO()
        mock_stdout.isatty = lambda: True  # type: ignore[attr-defined]
        with patch("sys.stdout", mock_stdout):
            result = ansi("hello", ANSI_BOLD)
            assert result == ANSI_BOLD + "hello" + ANSI_RESET

    def test_tty_multiple_codes(self):
        mock_stdout = io.StringIO()
        mock_stdout.isatty = lambda: True  # type: ignore[attr-defined]
        with patch("sys.stdout", mock_stdout):
            result = ansi("text", ANSI_BOLD, ANSI_DIM)
            assert result == ANSI_BOLD + ANSI_DIM + "text" + ANSI_RESET


class TestLog:
    def test_log_prints(self, capsys):
        log("test message")
        assert capsys.readouterr().out == "test message\n"


class TestLogReasoning:
    def test_multiline(self, capsys):
        # log_reasoning uses ansi() which checks isatty — in test, stdout
        # is not a TTY so ansi() returns plain text
        log_reasoning("line1\nline2")
        output = capsys.readouterr().out
        lines = output.strip().split("\n")
        assert len(lines) == 2
        # Each line should have the "  │ " prefix (plain, since not TTY)
        assert "│" in lines[0]
        assert "line1" in lines[0]
        assert "line2" in lines[1]
