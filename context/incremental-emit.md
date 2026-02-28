# Incremental Commit Emission

## Context

Currently, the LLM investigates changes in a multi-turn agentic loop, then produces ALL commits in a single `propose_commits` call at the end. This is problematic because:
- The final tool call is the largest structured output, most likely to hit Ollama timeouts
- The LLM must hold the entire commit plan in memory for one big output
- No feedback about remaining work between investigation and final proposal

The fix: replace the single `propose_commits` terminal tool with an `emit_commit` investigation tool (called per-commit) + a lightweight `finalize_commits` terminal tool.

## Issues

- **gsc-tf0** — Add `EmitCommitArgs` and `FinalizeCommitsArgs` pydantic models
- **gsc-np1** — Add `emit_commit` and `finalize_commits` closures in `classify_changes`
- **gsc-i55** — Add `countdown_message_fn` callback to `agentic_loop`
- **gsc-59q** — Update `AGENTIC_SYSTEM_PROMPT` for incremental emit workflow
- **gsc-mvz** — Bump `TURNS_WARN_AT` and mark `propose_commits` as deprecated

## Design

All changes in `git-smart-commit` (single file).

### 1. New pydantic models (gsc-tf0)

Insert after `ProposeCommitsArgs` (~line 1370):

**`EmitCommitArgs`**: Wraps a single `Commit`. Has `post_process()` that delegates to `Commit.post_process()`.

**`FinalizeCommitsArgs`**: Just `gitignore: list[str]`. Has trivial `post_process()`.

### 2. Closures in `classify_changes` (gsc-np1)

Create local state and two closure-based tools in `classify_changes` (line 1821):

```python
emitted_commits: list[dict] = []
emitted_hunks: set[str] = set()
```

**`emit_commit(args, **_)`** — regular investigation tool:
- Validates hunk IDs against `hunk_map` (rejects unknown)
- Rejects hunks already in `emitted_hunks` (prevents overlap)
- Appends commit dict to `emitted_commits`, updates `emitted_hunks`
- Returns feedback: "Commit emitted: '{subject}'. Remaining unassigned hunks: [...]"
- Added to `tool_registry`

**`finalize_commits(args, **_)`** — terminal tool:
- Raises `ValueError` if `emitted_commits` is empty
- Calls `analyzer.merge_overlapping_commits()` as safety check
- Returns `(emitted_commits, gitignore)`
- Passed as `terminal_tool` to `agentic_loop`

Pattern reference: `query_tool_result` closure at line 921-946 (accepts `**_: Any`).

### 3. Countdown callback in `agentic_loop` (gsc-i55)

Add optional `countdown_message_fn: Callable | None = None` parameter to `agentic_loop` (line 841).

Update countdown warning (lines 1068-1078):
- If callback provided, use it; otherwise use default message
- In `classify_changes`, pass a closure that checks `emitted_commits` state:
  - No commits emitted → "emit commits then finalize NOW"
  - Hunks remain → "N hunks unassigned, emit or finalize NOW"
  - All assigned → "call finalize NOW"

### 4. System prompt update (gsc-59q)

Update `AGENTIC_SYSTEM_PROMPT` (lines 290-393):
- Tool docs: add `emit_commit(commit)` description
- Investigation strategy: "As you identify logical groupings, call emit_commit immediately. Check remaining hunks in the response. When done, call finalize_commits."
- Replace final line "call the propose_commits tool" → "call finalize_commits"
- Keep all other rules (conventional commits, issue detection, etc.) unchanged

### 5. Tuning and cleanup (gsc-mvz)

- Increase `TURNS_WARN_AT` from 4 to 6 (incremental emit needs more tool calls)
- Add deprecation comment to `propose_commits` (keep code — `Commit` model, `SAMPLE_OUTPUT`, `MergeCommitsArgs` still reference it)

## What stays unchanged

- `agentic_loop` core logic (emit_commit is a regular tool, finalize is terminal)
- TUI/text confirmation flow (still receives `list[dict]`)
- `execute_commits` (same commit dict structure)
- `--json`/`--plan` serialization format
- `_Reclassify` re-classification loop (closures reset on each call)
- `merge_overlapping_commits`

## Verification

1. Run `./git-smart-commit` on the repo with a few unstaged changes
2. Confirm the LLM calls `emit_commit` incrementally (visible in turn log)
3. Confirm the LLM calls `finalize_commits` at the end
4. Confirm the TUI shows the correct commits
5. Confirm staging and committing works correctly
6. Test error path: if LLM uses wrong hunk ID in emit_commit, it should get error feedback and retry
