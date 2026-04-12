"""Tests for steropes.types — core data types."""

from steropes.types import TokenUsage, ToolCall


class TestTokenUsage:
    def test_addition(self):
        a = TokenUsage(100, 50)
        b = TokenUsage(200, 75)
        c = a + b
        assert c.prompt_tokens == 300
        assert c.completion_tokens == 125

    def test_addition_with_zero(self):
        a = TokenUsage(0, 0)
        b = TokenUsage(100, 50)
        assert (a + b).prompt_tokens == 100
        assert (a + b).completion_tokens == 50

    def test_addition_returns_new_instance(self):
        a = TokenUsage(10, 20)
        b = TokenUsage(30, 40)
        c = a + b
        assert c is not a
        assert c is not b

    def test_total_tokens(self):
        u = TokenUsage(100, 50)
        assert u.total_tokens == 150

    def test_total_tokens_zero(self):
        assert TokenUsage().total_tokens == 0

    def test_str_formatting(self):
        u = TokenUsage(1000, 500)
        s = str(u)
        assert "1,500 tokens" in s
        assert "1,000 in" in s
        assert "500 out" in s

    def test_str_zero(self):
        s = str(TokenUsage())
        assert "0 tokens" in s

    def test_defaults(self):
        u = TokenUsage()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0


class TestToolCall:
    def test_fields(self):
        tc = ToolCall(name="read_file", arguments={"path": "a.py"}, call_id="c1")
        assert tc.name == "read_file"
        assert tc.arguments == {"path": "a.py"}
        assert tc.call_id == "c1"

    def test_call_id_default_none(self):
        tc = ToolCall(name="test", arguments={})
        assert tc.call_id is None
