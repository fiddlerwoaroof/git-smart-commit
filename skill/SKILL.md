---
name: smart-commit
description: >
  Use this skill INSTEAD OF running git commit, git add, or git status whenever
  the user wants to commit changes, stage files, or asks what has changed.
  This replaces manual git commit workflows entirely. Do NOT run git commit or
  git add directly — always use this skill for any commit-related request.
argument-hint: [--dry-run] [--model MODEL] [--repo PATH] [--api-base URL] [--api-key KEY] [--critique TEXT]
allowed-tools: Bash(git-smart-commit *)
---

## Working tree status

!`git status --short 2>/dev/null || echo "(not a git repo)"`

Use `git-smart-commit` to analyze the changes above and group them into logical commits.

**Before starting:** Split `$ARGUMENTS` into two groups:
- **Passthrough flags** — forwarded to `git-smart-commit` for classification:
  `--model`, `--repo`, `--api-base`, `--api-key`, `--critique`
- **Control-flow flags** — handled by this workflow, NOT passed to `--json` commands:
  `--dry-run`, `--yes`

**Workflow:**

1. If the status is empty, tell the user there is nothing to commit and stop.

2. Generate and save the plan using only passthrough flags:
   ```
   git-smart-commit --json --save-plan /tmp/gsc-plan.json [passthrough flags]
   ```
   This is read-only — it does not modify anything. The plan is saved so the
   execute step replays it exactly, rather than re-running classification.

3. Display each proposed commit clearly, e.g.:

   ```
   [1] feat(auth): add JWT refresh token support
       Adds /refresh endpoint and rotation logic.
       + src/auth/refresh.py
       + tests/test_refresh.py

   [2] chore(deps): bump httpx to 0.27
       + requirements.txt
   ```

   Also list any flagged code issues and suggested .gitignore patterns.

4. **Rate the proposed commits 1–10.** Consider:
   - `git-smart-commit` can only stage **whole files** — a file cannot be split
     across commits, so penalise groupings only for things that are actually
     fixable given that constraint (e.g. unrelated files lumped together when
     they could be separated, wrong commit type, vague subject line).
   - Do NOT penalise for mixed concerns within a single file — that is
     unavoidable and not a flaw in the proposal.

   If the rating is **below 7**, re-run with feedback and overwrite the saved plan:
   ```
   git-smart-commit --json --save-plan /tmp/gsc-plan.json [passthrough flags] --critique "<specific feedback>"
   ```
   Display the revised proposals instead.

5. If `--dry-run` was in `$ARGUMENTS`, stop here.

6. Otherwise ask: **"Proceed with these commits? [y/N]"**
   - **Yes** (or if `--yes` was in `$ARGUMENTS`) → execute the saved plan exactly
     (pass `--repo` if it was in the passthrough flags):
     ```
     git-smart-commit --plan /tmp/gsc-plan.json --yes [--repo PATH if specified]
     ```
   - **No** → ask what the user wants to change; they can re-invoke with adjusted flags
