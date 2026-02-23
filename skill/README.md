# smart-commit skill

A Claude Code slash command that runs `git-smart-commit` interactively,
presents proposed commits for review, then executes them on confirmation.

## Installation

Copy `SKILL.md` to your Claude Code skills directory:

```bash
# Project-level (this repo only)
mkdir -p .claude/skills/smart-commit
cp skill/SKILL.md .claude/skills/smart-commit/

# Personal (all your projects)
mkdir -p ~/.claude/skills/smart-commit
cp skill/SKILL.md ~/.claude/skills/smart-commit/
```

`git-smart-commit` must be on your `PATH`.

## Usage

```
/smart-commit
/smart-commit --dry-run
/smart-commit --model qwen3-coder:8b
/smart-commit --api-base https://openrouter.ai/api/v1 --api-key $KEY --model deepseek/deepseek-r1
/smart-commit --yes
```
