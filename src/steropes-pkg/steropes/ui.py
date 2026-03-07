"""ANSI terminal output utilities for the steropes agent framework."""

import sys

ANSI_RESET  = "\033[0m"
ANSI_BOLD   = "\033[1m"
ANSI_DIM    = "\033[2m"
ANSI_RED    = "\033[31m"
ANSI_GREEN  = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_CYAN   = "\033[36m"


def ansi(text: str, *codes: str) -> str:
    """Wrap text in ANSI escape codes when stdout is a TTY (no-op otherwise)."""
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + text + ANSI_RESET


def log(msg: str) -> None:
    """Print a progress line to stdout."""
    print(msg)


def log_reasoning(text: str) -> None:
    """Print reasoning/thinking tokens dimmed and indented."""
    prefix = ansi("  │ ", ANSI_DIM)
    for line in text.splitlines():
        print(prefix + ansi(line, ANSI_DIM))
