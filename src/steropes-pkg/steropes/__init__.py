"""steropes — A lightweight LLM agent framework.

Public API re-exports for convenient single-import usage.
"""

__version__ = "0.1.0"

from .client import LLMClient, QueryResultArgs
from .config import AgentConfig, ApiConfig
from .parsers import OllamaParser, OpenAIParser, ResponseParser
from .text import wrap_markdown
from .tools import tool
from .types import TokenUsage, ToolCall
from .ui import (
    ANSI_BOLD,
    ANSI_CYAN,
    ANSI_DIM,
    ANSI_GREEN,
    ANSI_RED,
    ANSI_RESET,
    ANSI_YELLOW,
    ansi,
    log,
    log_reasoning,
)

__all__ = [
    # client
    "LLMClient",
    "QueryResultArgs",
    # config
    "AgentConfig",
    "ApiConfig",
    # parsers
    "ResponseParser",
    "OllamaParser",
    "OpenAIParser",
    # text
    "wrap_markdown",
    # tools
    "tool",
    # types
    "TokenUsage",
    "ToolCall",
    # ui
    "ANSI_BOLD",
    "ANSI_CYAN",
    "ANSI_DIM",
    "ANSI_GREEN",
    "ANSI_RED",
    "ANSI_RESET",
    "ANSI_YELLOW",
    "ansi",
    "log",
    "log_reasoning",
]
