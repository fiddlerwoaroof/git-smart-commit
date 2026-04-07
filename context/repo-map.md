# Repository Map: git-smart-commit

## Top-Level Structure

- `git-smart-commit` — Main executable (uv script, ~3100+ lines Python). Analyzes git changes, groups hunks into logical commits via LLM, optionally executes them.
- `pyproject.toml` — Workspace config (requires-python >=3.12); steropes-pkg as member
- `uv.lock` — Dependency lockfile
- `Dockerfile` — Containerized deployment

## Steropes Package (`src/steropes-pkg/steropes/`)

Lightweight LLM agent framework with tool calling and context management.

- `__init__.py` — Public API re-exports (v0.1.0)
- `client.py` (~700 lines) — `LLMClient`: streaming LLM calls, `call_with_tool`, `agentic_loop` with context trimming/summarization, `result_store`
- `config.py` — `AgentConfig` (tuning constants), `ApiConfig` (API connection/model)
- `parsers.py` — `ResponseParser` (abstract), `OllamaParser`, `OpenAIParser`
- `tools.py` — `@tool(args_model)` decorator, `_summarize_args`, `_total_message_chars`
- `types.py` — `ToolCall`, `TokenUsage`
- `ui.py` — ANSI color constants (`ANSI_RESET`, `ANSI_BOLD`, etc.), `ansi()`, `log()`, `log_reasoning()`
- `text.py` — `wrap_markdown(text, width=80)`: markdown-aware wrapping preserving code blocks, lists, blockquotes, inline code

## Main Script Key Components (`git-smart-commit`)

### Constants & Prompts
- `HUNK_GROUPING_RULES`, `BREAKING_CHANGE_RULES`, `CODE_ISSUE_DETECTION`, `COMMIT_FORMAT_RULES`
- `SYSTEM_PROMPT` (one-shot mode), `AGENTIC_SYSTEM_PROMPT` (agentic loop)

### Data Models (Pydantic)
- `DiffHunk` — unified diff hunk
- `Issue` — code issue detection
- `Commit` — commit with message, type, scope, body, hunks, issues, breaking_change
- Tool arg models: `EmitCommitArgs`, `FinalizeCommitsArgs`, `MergeCommitsArgs`, `ReadFileArgs`, `GetDiffArgs`, `GetGitLogArgs`, `SearchDiffArgs`, `ListSymbolsArgs`

### Core Classes
- `GitAnalyzer` — collects diffs, generates hunk IDs (`filepath::number`), tracks remaining hunks, file reading, symbol listing
- `_Reclassify` — reclassification workflow

### Key Functions
- `split_hunks(diff_text)` — split unified diff into individual hunks
- `merge_commits()` — merge overlapping commits
- `emit_commit()` / `finalize_commits()` — incremental commit emission
- `print_proposed_commits()` — formatted commit display with rich markup
- `_run_text_confirmation()` — interactive confirmation flow
- `main()` — CLI entry point

## Training Pipeline
- `harvest-training-data.py` — walk local repos, collect diffs
- `label-training-data.py` — teacher model labeling
- `prepare-sft-data.py` — ChatML format + train/val splits
- `train.py` — QLoRA finetuning (Unsloth)
- `eval.py` — F1/accuracy evaluation
- `merge-and-quantize.sh` — LoRA merge + GGUF quantization
