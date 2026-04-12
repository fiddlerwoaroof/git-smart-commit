"""Tests for steropes.parsers — response parsing for Ollama and OpenAI."""

from steropes.parsers import OllamaParser, OpenAIParser
from steropes.types import TokenUsage


class TestOllamaParser:
    def setup_method(self):
        self.parser = OllamaParser()

    def test_get_message(self):
        data = {"message": {"role": "assistant", "content": "hello"}}
        assert self.parser.get_message(data) == {"role": "assistant", "content": "hello"}

    def test_get_message_missing(self):
        assert self.parser.get_message({}) == {}

    def test_extract_usage(self):
        data = {"prompt_eval_count": 100, "eval_count": 50}
        usage = self.parser.extract_usage(data)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50

    def test_extract_usage_missing(self):
        usage = self.parser.extract_usage({})
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0

    def test_extract_text(self):
        data = {"message": {"role": "assistant", "content": "  hello  "}}
        assert self.parser.extract_text(data) == "hello"

    def test_extract_text_empty(self):
        data = {"message": {"role": "assistant", "content": ""}}
        assert self.parser.extract_text(data) is None

    def test_extract_text_no_message(self):
        assert self.parser.extract_text({}) is None

    def test_extract_tool_call_present(self):
        data = {
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": {"path": "main.py"},
                        }
                    }
                ],
            }
        }
        tc = self.parser.extract_tool_call(data)
        assert tc is not None
        assert tc.name == "read_file"
        assert tc.arguments == {"path": "main.py"}

    def test_extract_tool_call_absent(self):
        data = {"message": {"role": "assistant", "content": "no tools"}}
        assert self.parser.extract_tool_call(data) is None

    def test_extract_tool_call_string_args(self):
        data = {
            "message": {
                "tool_calls": [
                    {
                        "function": {
                            "name": "my_tool",
                            "arguments": '{"key": "value"}',
                        }
                    }
                ]
            }
        }
        tc = self.parser.extract_tool_call(data)
        assert tc is not None
        assert tc.arguments == {"key": "value"}

    def test_format_tool_result(self):
        result = self.parser.format_tool_result("call_1", "output data")
        assert result["role"] == "tool"
        assert result["content"] == "output data"

    def test_extract_reasoning(self):
        data = {"message": {"thinking": "let me think..."}}
        assert self.parser.extract_reasoning(data) == "let me think..."

    def test_extract_reasoning_absent(self):
        data = {"message": {"content": "hello"}}
        assert self.parser.extract_reasoning(data) is None


class TestOpenAIParser:
    def setup_method(self):
        self.parser = OpenAIParser()

    def test_get_message(self):
        data = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
        assert self.parser.get_message(data) == {"role": "assistant", "content": "hi"}

    def test_get_message_empty_choices(self):
        assert self.parser.get_message({"choices": []}) == {}

    def test_get_message_no_choices(self):
        assert self.parser.get_message({}) == {}

    def test_extract_usage(self):
        data = {"usage": {"prompt_tokens": 200, "completion_tokens": 80}}
        usage = self.parser.extract_usage(data)
        assert usage.prompt_tokens == 200
        assert usage.completion_tokens == 80

    def test_extract_usage_missing(self):
        usage = self.parser.extract_usage({})
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0

    def test_format_tool_result_with_call_id(self):
        result = self.parser.format_tool_result("call_abc", "output")
        assert result["role"] == "tool"
        assert result["content"] == "output"
        assert result["tool_call_id"] == "call_abc"

    def test_format_tool_result_no_call_id(self):
        result = self.parser.format_tool_result(None, "output")
        assert result["role"] == "tool"
        assert result["content"] == "output"
        assert "tool_call_id" not in result

    def test_extract_tool_call_with_id(self):
        data = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_xyz",
                                "function": {
                                    "name": "emit_commit",
                                    "arguments": '{"data": 1}',
                                },
                            }
                        ]
                    }
                }
            ]
        }
        tc = self.parser.extract_tool_call(data)
        assert tc is not None
        assert tc.name == "emit_commit"
        assert tc.call_id == "call_xyz"
        assert tc.arguments == {"data": 1}
