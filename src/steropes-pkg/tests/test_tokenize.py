"""Tests for LLMClient.count_tokens — llama-server /tokenize integration.

The real endpoint is unavailable in CI, so we stub the HTTP client with
unittest.mock. Covers the happy path, the fallback when /tokenize errors,
and the server-root derivation for OpenAI-compatible base_urls that
include a /v1 suffix. See gsc-psn.
"""

from unittest.mock import MagicMock

import httpx

from steropes.client import LLMClient
from steropes.config import AgentConfig, ApiConfig


def _client(base_url: str) -> LLMClient:
    cfg = ApiConfig(base_url=base_url, model="test-model")
    return LLMClient(cfg, agent_config=AgentConfig(), log_fn=lambda _: None)


def _mock_response(status: int, payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    return resp


class TestCountTokens:
    def test_returns_token_count_from_endpoint(self):
        client = _client("http://localhost:8001/v1")
        client._http = MagicMock()
        client._http.post.return_value = _mock_response(
            200, {"tokens": [1, 2, 3, 4, 5]}
        )

        assert client.count_tokens("hello world") == 5

    def test_strips_v1_suffix_for_server_root(self):
        """llama-server's /tokenize lives at the server root, not under /v1."""
        client = _client("http://localhost:8001/v1")
        client._http = MagicMock()
        client._http.post.return_value = _mock_response(200, {"tokens": []})

        client.count_tokens("hi")

        called_url = client._http.post.call_args[0][0]
        assert called_url == "http://localhost:8001/tokenize"

    def test_handles_trailing_slash(self):
        client = _client("http://localhost:8001/v1/")
        client._http = MagicMock()
        client._http.post.return_value = _mock_response(200, {"tokens": [1]})

        client.count_tokens("x")

        called_url = client._http.post.call_args[0][0]
        assert called_url == "http://localhost:8001/tokenize"

    def test_sends_content_in_body(self):
        client = _client("http://localhost:8001/v1")
        client._http = MagicMock()
        client._http.post.return_value = _mock_response(200, {"tokens": [1]})

        client.count_tokens("the quick brown fox")

        kwargs = client._http.post.call_args.kwargs
        assert kwargs["json"] == {"content": "the quick brown fox"}

    def test_fallback_when_endpoint_missing(self):
        """404 response falls back to chars/4 approximation."""
        client = _client("http://localhost:8001/v1")
        client._http = MagicMock()
        client._http.post.return_value = _mock_response(404, {})

        # 20 chars → 5 tokens under the fallback
        assert client.count_tokens("a" * 20) == 5

    def test_fallback_when_endpoint_errors(self):
        """Network error falls back to chars/4."""
        client = _client("http://localhost:8001/v1")
        client._http = MagicMock()
        client._http.post.side_effect = httpx.ConnectError("boom")

        assert client.count_tokens("a" * 16) == 4

    def test_empty_string_returns_zero(self):
        """No network call needed for empty input."""
        client = _client("http://localhost:8001/v1")
        client._http = MagicMock()

        assert client.count_tokens("") == 0
        client._http.post.assert_not_called()

    def test_ollama_base_url_unchanged(self):
        """Ollama base URLs don't have /v1 — tokenize URL should still work."""
        client = _client("http://localhost:11434")
        client._http = MagicMock()
        client._http.post.return_value = _mock_response(200, {"tokens": [1, 2]})

        client.count_tokens("hi")
        called_url = client._http.post.call_args[0][0]
        assert called_url == "http://localhost:11434/tokenize"
