"""LLM response parsers for Ollama and OpenAI-compatible APIs."""

import json

from .types import ToolCall, TokenUsage


class ResponseParser:
    """Backend-specific API response parsing.

    Subclass and override get_message() for new backends.  Override
    REASONING_FIELDS to control which response fields are checked for
    thinking/reasoning tokens (checked in order, first non-empty wins).
    """

    REASONING_FIELDS: tuple[str, ...] = ("thinking", "reasoning", "reasoning_content")

    def get_message(self, data: dict) -> dict:
        """Extract the assistant message dict from a raw API response."""
        raise NotImplementedError

    def extract_reasoning(self, data: dict) -> str | None:
        msg = self.get_message(data)
        for field in self.REASONING_FIELDS:
            value = msg.get(field)
            if value and isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def extract_tool_call(self, data: dict) -> ToolCall | None:
        msg = self.get_message(data)
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            tc = tool_calls[0]
            raw = tc["function"]["arguments"]
            args = json.loads(raw) if isinstance(raw, str) else raw
            return ToolCall(name=tc["function"]["name"], arguments=args,
                            call_id=tc.get("id"))
        return None

    def extract_tool_args(self, data: dict) -> dict | None:
        """Extract tool-call arguments, falling back to content-as-JSON."""
        tc = self.extract_tool_call(data)
        if tc is not None:
            return tc.arguments
        content = self.get_message(data).get("content", "").strip()
        if content:
            content = content.removeprefix("```json").removeprefix("```").strip().removesuffix("```").strip()
            return json.loads(content)
        return None

    def extract_text(self, data: dict) -> str | None:
        content = self.get_message(data).get("content", "").strip()
        return content or None

    def format_assistant(self, data: dict) -> dict:
        return self.get_message(data) or {"role": "assistant", "content": ""}

    def format_tool_result(self, call_id: str | None, content: str) -> dict:
        return {"role": "tool", "content": content}

    def extract_usage(self, data: dict) -> TokenUsage:
        raise NotImplementedError


class OllamaParser(ResponseParser):
    """Response parser for the native Ollama /api/chat endpoint."""

    def get_message(self, data: dict) -> dict:
        return data.get("message", {})

    def extract_usage(self, data: dict) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
        )


class OpenAIParser(ResponseParser):
    """Response parser for OpenAI-compatible /chat/completions endpoints."""

    def get_message(self, data: dict) -> dict:
        choices = data.get("choices", [])
        return choices[0].get("message", {}) if choices else {}

    def format_tool_result(self, call_id: str | None, content: str) -> dict:
        if call_id is not None:
            return {"role": "tool", "tool_call_id": call_id, "content": content}
        return {"role": "tool", "content": content}

    def extract_usage(self, data: dict) -> TokenUsage:
        usage = data.get("usage", {})
        return TokenUsage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )
