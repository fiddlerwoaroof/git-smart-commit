"""Tests for steropes.config — configuration dataclasses."""

from steropes.config import AgentConfig, ApiConfig


class TestApiConfig:
    def test_is_ollama_true(self):
        cfg = ApiConfig(base_url="http://localhost:11434", model="llama3")
        assert cfg.is_ollama is True

    def test_is_ollama_false_openai(self):
        cfg = ApiConfig(base_url="https://api.openai.com/v1", model="gpt-4")
        assert cfg.is_ollama is False

    def test_is_ollama_false_custom(self):
        cfg = ApiConfig(base_url="http://my-server/v1", model="model")
        assert cfg.is_ollama is False

    def test_is_anthropic_true(self):
        cfg = ApiConfig(base_url="https://api.anthropic.com/v1", model="claude")
        assert cfg.is_anthropic is True

    def test_is_anthropic_false(self):
        cfg = ApiConfig(base_url="https://api.openai.com/v1", model="gpt-4")
        assert cfg.is_anthropic is False

    def test_chat_url_ollama(self):
        cfg = ApiConfig(base_url="http://localhost:11434", model="llama3")
        assert cfg.chat_url == "http://localhost:11434/api/chat"

    def test_chat_url_openai(self):
        cfg = ApiConfig(base_url="https://api.openai.com/v1", model="gpt-4")
        assert cfg.chat_url == "https://api.openai.com/v1/chat/completions"

    def test_auth_headers_with_key(self):
        cfg = ApiConfig(base_url="http://x", model="m", api_key="sk-test")
        assert cfg.auth_headers == {"Authorization": "Bearer sk-test"}

    def test_auth_headers_no_key(self):
        cfg = ApiConfig(base_url="http://x", model="m")
        assert cfg.auth_headers == {}

    def test_auth_headers_empty_string_key(self):
        # Empty string is falsy — should return empty headers
        cfg = ApiConfig(base_url="http://x", model="m", api_key="")
        assert cfg.auth_headers == {}


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.max_agentic_turns == 20
        # Thresholds are now measured in tokens (see config.py docstring)
        assert cfg.context_trim_threshold == 75_000
        assert cfg.context_soft_threshold is None
        assert cfg.context_hard_threshold is None
        assert cfg.compaction_preserve_marker is None

    def test_custom_thresholds(self):
        cfg = AgentConfig(
            context_soft_threshold=200_000,
            context_hard_threshold=300_000,
        )
        assert cfg.context_soft_threshold == 200_000
        assert cfg.context_hard_threshold == 300_000

    def test_preserve_marker(self):
        cfg = AgentConfig(compaction_preserve_marker="MARKER\n")
        assert cfg.compaction_preserve_marker == "MARKER\n"
