"""Core data types for the steropes agent framework."""

from dataclasses import dataclass


@dataclass
class ToolCall:
    """Represents a tool call extracted from an LLM response."""
    name: str
    arguments: dict
    call_id: str | None = None


@dataclass
class TokenUsage:
    """Accumulated token usage across all API calls in a session."""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __str__(self) -> str:
        return (f"{self.total_tokens:,} tokens "
                f"({self.prompt_tokens:,} in / {self.completion_tokens:,} out)")
