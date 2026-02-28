"""Tool decorator and utilities for the steropes agent framework."""

import json
from typing import Callable

from pydantic import BaseModel


def tool(args_model: type[BaseModel]):
    """Decorator factory. @tool(MyArgsModel) attaches LLM tool metadata to a function."""
    def decorator(fn: Callable) -> Callable:
        spec = {
            "type": "function",
            "function": {
                "name": fn.__name__,
                "description": (fn.__doc__ or "").strip(),
                "parameters": args_model.model_json_schema(),
            },
        }
        fn._tool_spec = spec
        fn._tool_model = args_model
        return fn
    return decorator


def _summarize_args(arguments: dict, max_len: int = 80) -> str:
    """Return a compact one-line summary of tool call arguments for logging."""
    raw = json.dumps(arguments, separators=(",", ":"))
    if len(raw) <= max_len:
        return raw
    return raw[:max_len - 3] + "..."


def _total_message_chars(messages: list[dict]) -> int:
    """Count total characters across all message content fields."""
    return sum(len(str(m.get("content", "") or "")) for m in messages)
