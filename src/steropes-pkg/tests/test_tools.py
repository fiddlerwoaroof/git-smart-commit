"""Tests for steropes.tools — tool decorator and utility functions."""

from pydantic import BaseModel, Field

from steropes.tools import _summarize_args, _total_message_chars, tool


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
