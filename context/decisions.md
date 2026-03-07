# Architectural Decisions

## OO Refactor (gsc-qov)

**Decision:** Replaced module-level globals with `LLMClient` and `GitAnalyzer` classes.

- `_api_config` (mutable module-level global mutated via `global` in `main()`) → `ApiConfig` dataclass instantiated in `main()`, owned by `LLMClient`
- `_TOOL_REGISTRY` (populated but never read) → removed as dead code
- `_build_payload`, `_extract_tool_args`, `call_ollama`, `call_ollama_with_tool` → methods on `LLMClient`
- `get_changed_files`, `build_diff_summary`, `classify_changes`, `merge_overlapping_commits`, `execute_commits`, `update_gitignore` → methods on `GitAnalyzer`
- `sys.exit` is only called in `main()` (the entry point); all other error paths raise exceptions

## Multi-turn Agentic Loop (gsc-bi8, gsc-67t)

**Decision:** Implement a multi-turn tool-use loop (`run_agentic_loop`) instead of the single-shot `call_with_tool` for the classification step.

**Rationale:** The LLM needs to be able to call investigation tools (`read_file`, and later `get_diff`, `get_git_log`, `search_diff`) before proposing commits. A single forced tool call cannot support this.

**Design:**
- `run_agentic_loop(messages, intermediate_tools, terminal_tool, **kwargs)` drives the conversation
- Intermediate tools (e.g. `read_file`) are executed immediately; results are appended to the message history
- The loop terminates when the terminal tool (`propose_commits`) is called
- `call_with_tool` is kept for single-shot cases (e.g. `merge_commits` during overlap resolution)

## ToolCall Dataclass (gsc-67t)

**Decision:** Introduce a `ToolCall` dataclass (name, arguments, call_id) to normalize the Ollama/OpenAI tool call response formats.

**Rationale:** Ollama and OpenAI return tool calls in slightly different formats (Ollama has no `id`; OpenAI requires `tool_call_id` in results). Centralizing extraction into `_extract_tool_call()` → `ToolCall` prevents format-handling logic from leaking into the loop body.

**Named helper methods on LLMClient:**
- `_build_payload_messages(messages, tools, stream)` — builds payload from a message list
- `_extract_tool_call(data)` → `ToolCall | None` — extracts the first tool call
- `_extract_text_content(data)` → `str` — extracts plain text when no tool was called
- `_format_assistant_tool_call(data)` → `dict` — formats assistant message for history
- `_format_tool_result(call_id, content)` → `dict` — formats tool result for history
- `_summarize_args(args)` → `str` — compact one-line arg summary for logging

## Tool System

**Decision:** `@tool(ArgsModel)` decorator attaches `_tool_spec` and `_tool_model` to functions; no global registry.

**Rationale:** The original `_TOOL_REGISTRY` dict was populated but never read — all dispatch went through `fn._tool_spec` and `fn._tool_model` attributes. Removing it simplifies the code with no behavioral change.

All tool functions accept `**kwargs` forwarded from the loop, with `analyzer: GitAnalyzer` being the primary context object passed through.

## steropes Extraction (gsc-2fj, gsc-7eo, gsc-03f, gsc-bx8)

**Decision:** Extract the general-purpose LLM agent framework from `git-smart-commit` into a separate package called `steropes`.

**Rationale:** The script contained ~700 lines of reusable LLM infrastructure (client, tool system, parsers, context management) intertwined with git-commit-specific logic. Extracting it enables reuse for other agent-based tools.

**Package structure:** `src/steropes-pkg/steropes/` with modules:
- `text.py` — `wrap_markdown`, `_wrap_line_preserving_inline`
- `config.py` — `AgentConfig` (tuning constants), `ApiConfig` (connection config, no hardcoded defaults)
- `types.py` — `ToolCall`, `TokenUsage` dataclasses
- `tools.py` — `@tool` decorator, `_summarize_args`, `_total_message_chars`
- `parsers.py` — `ResponseParser`, `OllamaParser`, `OpenAIParser`
- `client.py` — `LLMClient` (call, call_with_tool, agentic_loop), `QueryResultArgs`
- `ui.py` — `ansi()`, `log()`, `log_reasoning()`, ANSI constants

**Key design decisions:**
- `AgentConfig` dataclass replaces module-level tuning constants; `LLMClient` reads from it
- `ApiConfig` is parameterized (no hardcoded OLLAMA_BASE_URL/DEFAULT_MODEL); app provides values
- `LLMClient.__init__` accepts `log_fn` and `log_reasoning_fn` callbacks so the app controls output (e.g. respecting `--quiet`)
- `QueryResultArgs` moved to framework since it's integral to `agentic_loop`'s context management
- steropes depends only on `httpx` and `pydantic`; no git, subprocess, or textual
- Symlink support: script resolves its real path via `Path(__file__).resolve()` to find the local steropes package
