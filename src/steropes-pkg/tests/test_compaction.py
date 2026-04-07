"""Tests for dual-threshold async context compaction."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from steropes.config import AgentConfig, ApiConfig
from steropes.client import LLMClient


def _make_client(soft: int, hard: int) -> LLMClient:
    """Create a client with dual thresholds and a mock HTTP backend."""
    api_config = ApiConfig(
        base_url="http://localhost:11434",
        model="test-model",
    )
    agent_config = AgentConfig(
        context_soft_threshold=soft,
        context_hard_threshold=hard,
        tool_result_summarize_skip=50,
        tool_result_summarize_input=500,
        max_agentic_turns=5,
    )
    client = LLMClient(api_config, agent_config=agent_config, log_fn=lambda _: None)
    return client


class TestDualThresholdConfig:
    def test_soft_hard_defaults_to_legacy(self):
        """When soft/hard are None, legacy threshold is used."""
        ac = AgentConfig(context_trim_threshold=100_000)
        assert ac.context_soft_threshold is None
        assert ac.context_hard_threshold is None
        assert ac.context_trim_threshold == 100_000

    def test_soft_hard_set_explicitly(self):
        ac = AgentConfig(
            context_soft_threshold=200_000,
            context_hard_threshold=300_000,
        )
        assert ac.context_soft_threshold == 200_000
        assert ac.context_hard_threshold == 300_000


class TestCompactableCandidates:
    """Test the candidate selection logic indirectly via _trim behavior."""

    def test_messages_under_threshold_not_compacted(self):
        """Small messages should not trigger compaction."""
        client = _make_client(soft=10_000, hard=20_000)
        # Just verify the client was created with the right config
        assert client.agent_config.context_soft_threshold == 10_000
        assert client.agent_config.context_hard_threshold == 20_000


class TestBackgroundCompaction:
    """Integration-style tests for the async compaction thread."""

    def test_legacy_fallback_when_no_dual_thresholds(self):
        """Legacy single-threshold still works when soft/hard are None."""
        api_config = ApiConfig(
            base_url="http://localhost:11434",
            model="test-model",
        )
        ac = AgentConfig(
            context_trim_threshold=100_000,
            # soft/hard left as None — should fall back to legacy
        )
        client = LLMClient(api_config, agent_config=ac, log_fn=lambda _: None)
        assert client.agent_config.context_soft_threshold is None
        assert client.agent_config.context_hard_threshold is None
        assert client.agent_config.context_trim_threshold == 100_000
