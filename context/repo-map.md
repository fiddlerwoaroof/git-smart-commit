# Repository Map: git-smart-commit

## Top-Level Structure

- `git-smart-commit` — Main executable (uv script, ~3200+ lines Python). Analyzes git changes, groups hunks into logical commits via LLM, optionally executes them. Resolves its own real path to find steropes via `sys.path` insert, so it works both from the repo and via symlink.
- `pyproject.toml` — Workspace config (requires-python >=3.12); steropes-pkg as member
- `uv.lock` — Dependency lockfile
- `Dockerfile` — Containerized deployment

## Steropes Package (`src/steropes-pkg/steropes/`)

Lightweight LLM agent framework with tool calling and context management.

- `__init__.py` — Public API re-exports (v0.1.0)
- `client.py` (~800 lines) — `LLMClient`: streaming LLM calls, `call_with_tool`, `agentic_loop` with LCM-inspired dual-threshold async context compaction (τ_soft/τ_hard), three-level summarization escalation, `result_store` for lossless retrieval
- `config.py` — `AgentConfig` (tuning: `context_soft_threshold`, `context_hard_threshold`, etc.), `ApiConfig` (API connection/model, optional `temperature`)
- `parsers.py` — `ResponseParser` (abstract), `OllamaParser`, `OpenAIParser`
- `tools.py` — `@tool(args_model)` decorator, `_summarize_args`, `_total_message_chars`
- `types.py` — `ToolCall`, `TokenUsage`
- `ui.py` — ANSI color constants (`ANSI_RESET`, `ANSI_BOLD`, etc.), `ansi()`, `log()`, `log_reasoning()`
- `text.py` — `wrap_markdown(text, width, soft_width)`: AST-based markdown wrapping via mistletoe with paragraph reflow, soft/hard width targets

### Tests (`src/steropes-pkg/tests/`)

- `test_text.py` — 21 tests for markdown wrapping (reflow, code blocks, lists, blockquotes, headings, idempotency)
- `test_compaction.py` — Tests for dual-threshold config and compaction behavior

## Main Script Key Components (`git-smart-commit`)

### Bootstrap
- Lines 11-16: `sys.path` insert resolves symlink to find `src/steropes-pkg/` relative to script's real location

### Constants & Prompts
- `HUNK_GROUPING_RULES`, `BREAKING_CHANGE_RULES`, `CODE_ISSUE_DETECTION`, `COMMIT_FORMAT_RULES`
- `SYSTEM_PROMPT` (one-shot mode), `AGENTIC_SYSTEM_PROMPT` (agentic loop)
- `CONTEXT_SOFT_THRESHOLD` / `CONTEXT_HARD_THRESHOLD` — dual compaction thresholds (200k/300k chars; Ollama: 14k/20k)

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
