"""Tests for steropes.tools — tool decorator and utility functions."""

from pydantic import BaseModel, Field

from steropes.tools import (
    _summarize_args,
    _total_message_chars,
    _total_message_tokens,
    tool,
)


class TestSummarizeArgs:
    def test_short_args(self):
        result = _summarize_args({"file": "main.py", "line": 42})
        assert "main.py" in result
        assert "42" in result

    def test_empty_dict(self):
        assert _summarize_args({}) == "{}"

    def test_long_args_truncated(self):
        args = {"data": "x" * 200}
        result = _summarize_args(args, max_len=80)
        assert len(result) <= 80
        assert result.endswith("...")

    def test_custom_max_len(self):
        args = {"key": "a" * 50}
        result = _summarize_args(args, max_len=30)
        assert len(result) <= 30

    def test_nested_structure(self):
        args = {"items": [1, 2, 3], "nested": {"a": "b"}}
        result = _summarize_args(args)
        # Uses compact separators (no spaces)
        assert "[1,2,3]" in result
        assert '"a":"b"' in result


class TestTotalMessageChars:
    def test_empty_list(self):
        assert _total_message_chars([]) == 0

    def test_single_message(self):
        messages = [{"role": "user", "content": "hello"}]
        assert _total_message_chars(messages) == 5

    def test_multiple_messages(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "tool", "content": "result data"},
        ]
        assert _total_message_chars(messages) == 3 + 5 + 11

    def test_missing_content(self):
        messages = [{"role": "assistant"}]
        assert _total_message_chars(messages) == 0

    def test_non_string_content(self):
        # Anthropic structured content (list of dicts)
        messages = [{"role": "system", "content": [{"type": "text", "text": "hi"}]}]
        result = _total_message_chars(messages)
        # str() of a list — just verify it doesn't crash
        assert result > 0


class TestTotalMessageTokens:
    """Token-based counterpart of _total_message_chars.

    Uses an injected count_fn so tests are deterministic and don't need
    a real tokenizer. See gsc-psn.
    """

    def test_empty_list(self):
        assert _total_message_tokens([], lambda s: len(s) // 4) == 0

    def test_sums_per_message_tokens(self):
        # A count_fn that returns the length of the string directly lets us
        # assert totals without needing a real tokenizer.
        messages = [
            {"role": "user", "content": "aa"},    # 2
            {"role": "tool", "content": "bbbbb"}, # 5
        ]
        assert _total_message_tokens(messages, len) == 7

    def test_cache_prevents_recount(self):
        calls: list[str] = []

        def counter(text: str) -> int:
            calls.append(text)
            return len(text)

        msg_a = {"role": "user", "content": "hello"}
        msg_b = {"role": "tool", "content": "world"}
        cache: dict[int, int] = {}

        total1 = _total_message_tokens([msg_a, msg_b], counter, cache)
        total2 = _total_message_tokens([msg_a, msg_b], counter, cache)
        assert total1 == total2 == 10
        # Second call should hit the cache for both messages — no extra calls.
        assert len(calls) == 2

    def test_cache_invalidation_on_replacement(self):
        """When a message dict is replaced (new identity), it must be recounted."""
        calls: list[str] = []

        def counter(text: str) -> int:
            calls.append(text)
            return len(text)

        cache: dict[int, int] = {}
        messages = [{"role": "user", "content": "hello"}]
        _total_message_tokens(messages, counter, cache)
        assert calls == ["hello"]

        # Simulate compaction: swap in a new dict with shorter content.
        # The cache must not return the stale 5 — the id() is different.
        messages[0] = {"role": "user", "content": "hi"}
        total = _total_message_tokens(messages, counter, cache)
        assert total == 2
        assert calls == ["hello", "hi"]

    def test_missing_content_counts_zero(self):
        messages = [{"role": "assistant"}]
        assert _total_message_tokens(messages, len) == 0


class _SampleArgs(BaseModel):
    name: str = Field(description="The name")
    count: int = Field(default=1, description="How many")

    def post_process(self):
        return self


class TestToolDecorator:
    def test_preserves_function_behavior(self):
        @tool(_SampleArgs)
        def my_tool(args, **kw):
            return f"hello {args.name}"

        result = my_tool(_SampleArgs(name="world"))
        assert result == "hello world"

    def test_attaches_tool_spec(self):
        @tool(_SampleArgs)
        def my_tool(args, **kw):
            """This is the docstring."""
            return "ok"

        spec = my_tool._tool_spec
        assert "function" in spec
        fn = spec["function"]
        assert fn["name"] == "my_tool"
        assert "docstring" in fn["description"].lower() or "This is" in fn["description"]
        assert "parameters" in fn

    def test_attaches_tool_model(self):
        @tool(_SampleArgs)
        def my_tool(args, **kw):
            return "ok"

        assert my_tool._tool_model is _SampleArgs

    def test_parameters_from_schema(self):
        @tool(_SampleArgs)
        def my_tool(args, **kw):
            return "ok"

        params = my_tool._tool_spec["function"]["parameters"]
        assert "properties" in params
        assert "name" in params["properties"]
