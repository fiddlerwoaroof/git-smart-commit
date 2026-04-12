"""Configuration dataclasses for the steropes agent framework."""

from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Tuning constants for the agentic loop and context management.

    All values have sensible defaults matching the original git-smart-commit
    behavior. Applications can override individual fields as needed.
    """

    max_agentic_turns: int = 20
    max_agentic_turns_cap: int = 50
    read_file_limit: int = 50_000

    context_trim_threshold: int = 300_000  # legacy: used if soft/hard not set
    context_soft_threshold: int | None = None  # τ_soft: async compaction trigger
    context_hard_threshold: int | None = None  # τ_hard: blocking compaction trigger
    tool_result_summarize_skip: int = 500
    tool_result_summarize_input: int = 20_000
    turns_warn_at: int = 6
    recent_tool_result_chars: int = 10_000
    compaction_preserve_marker: str | None = None  # if set, content before this marker survives compaction


@dataclass
class ApiConfig:
    """API connection and model configuration.

    No hardcoded defaults — the application must provide base_url and model.
    """

    base_url: str
    model: str
    api_key: str | None = None
    num_ctx: int = 128000
    temperature: float | None = None  # None = use server/model default

    @property
    def is_ollama(self) -> bool:
        """True when targeting a native Ollama endpoint (not OpenAI-compatible)."""
        return "/v1" not in self.base_url

    @property
    def is_anthropic(self) -> bool:
        """True when targeting Anthropic's API (enables cache_control on system prompt)."""
        return "anthropic.com" in self.base_url

    @property
    def chat_url(self) -> str:
        if self.is_ollama:
            return f"{self.base_url}/api/chat"
        return f"{self.base_url}/chat/completions"

    @property
    def auth_headers(self) -> dict[str, str]:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}
