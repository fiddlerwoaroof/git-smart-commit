# git-smart-commit

Analyzes your working tree changes, groups them into logical commits using
Qwen3-Coder-30B-A3B (via Ollama), and optionally executes them.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.ai) running locally
- `httpx` (`pip install httpx`)
- The model: `ollama pull qwen2.5-coder:32b`

> **Note on model name**: The tool defaults to `qwen2.5-coder:32b` since
> Qwen3-Coder-30B-A3B may not yet be in Ollama's registry under a stable name.
> Override with `--model` when it is available (e.g. `--model qwen3-coder:30b`).

## Installation

```bash
# Make executable and drop somewhere on your PATH
chmod +x git-smart-commit
cp git-smart-commit ~/.local/bin/
```

## Usage

```bash
# Interactive: review proposed commits before executing
git-smart-commit

# Non-interactive: commit immediately without prompting
git-smart-commit --yes

# Dry run: show proposed commits only
git-smart-commit --dry-run

# JSON output (for use in scripts or skill integrations)
git-smart-commit --json

# Different repo
git-smart-commit --repo ~/projects/myapp

# Different model
git-smart-commit --model qwen3-coder:30b
```

## Integration as a skill

For use as a sub-agent tool (e.g. from Gemini CLI or Claude Code skills),
the `--json` flag outputs a clean JSON array of proposed commits:

```json
[
  {
    "message": "feat(auth): add JWT refresh token support",
    "files": ["src/auth/refresh.py", "tests/test_refresh.py"]
  },
  {
    "message": "chore(deps): bump httpx to 0.27",
    "files": ["requirements.txt", "poetry.lock"]
  }
]
```

The orchestrator can inspect this, modify it, confirm with the user,
then call `git-smart-commit --yes` to execute — or pipe the JSON
into its own git tooling.

## Exit codes

| Code | Meaning               |
|------|-----------------------|
| 0    | Success               |
| 1    | No changes found      |
| 2    | Ollama/model error    |
| 3    | Git error             |
| 4    | User cancelled        |

## How it works

1. Runs `git diff` (staged + unstaged) and `git ls-files --others` to collect all changes
2. Builds per-file diffs (truncated at ~30k tokens for large files)
3. Sends the full diff summary to Qwen3-Coder via Ollama with a classification prompt
4. Model returns a JSON array grouping files into logical commits with conventional commit messages
5. Optionally stages and commits each group in order
