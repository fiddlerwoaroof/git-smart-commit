"""Tool decorator and utilities for the steropes agent framework."""

import json
from collections.abc import Callable

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
    return raw[: max_len - 3] + "..."


def _total_message_chars(messages: list[dict]) -> int:
    """Count total characters across all message content fields."""
    return sum(len(str(m.get("content", "") or "")) for m in messages)


def _total_message_tokens(
    messages: list[dict],
    count_fn: Callable[[str], int],
    cache: dict[int, int] | None = None,
) -> int:
    """Count total tokens across all message content fields.

    *count_fn* maps a string to its token count (e.g. LLMClient.count_tokens).
    *cache* is keyed on (id(message_dict)) so unchanged messages aren't
    re-tokenized.  Callers should invalidate entries when a message is replaced
    (e.g. during compaction).
    """
    if cache is None:
        cache = {}
    total = 0
    for m in messages:
        key = id(m)
        if key not in cache:
            cache[key] = count_fn(str(m.get("content", "") or ""))
        total += cache[key]
    return total
