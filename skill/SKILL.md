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

**CRITICAL RULE: Always generate a JSON plan before committing.**
Never invoke `git-smart-commit` without `--json` first. The commit step MUST
replay a saved plan via `--plan`; re-running classification at commit time is
forbidden (it could produce a different grouping than what the user approved).

**Workflow:**

1. If the status is empty, tell the user there is nothing to commit and stop.

2. Generate and save the plan using only passthrough flags:
   ```
   git-smart-commit --json --save-plan /tmp/gsc-plan.json [passthrough flags]
   ```
   This is read-only — it does not modify anything. The plan is saved so the
   execute step replays it exactly, rather than re-running classification.

3. Display each proposed commit clearly, including any flagged issues, e.g.:

   ```
   [1] feat(auth): add JWT refresh token support
       Adds /refresh endpoint and rotation logic.
       + src/auth/refresh.py
       + tests/test_refresh.py
       ⚠ src/auth/refresh.py: Exception() called with file= kwarg (line 42) — not a valid argument

   [2] chore(deps): bump httpx to 0.27
       + requirements.txt
   ```

   Also list any suggested .gitignore patterns at the end.

4. **Evaluate every flagged issue.** For each entry in the `issues` list, assess:
   - **Is it a real problem?** Distinguish genuine bugs (wrong API usage, incorrect
     syntax, unreachable code, obvious logic errors) from noise (style nits,
     opinionated patterns that aren't actually wrong).
   - **Severity:** classify each real issue as one of:
     - 🔴 **Blocking** — the code is likely broken or will crash; the user should
       fix this before committing.
     - 🟡 **Advisory** — worth noting but doesn't prevent the commit from being
       correct (e.g. a code smell, a missing edge-case check).
   - **Suggest a concrete fix** for blocking issues where possible.

   If any **blocking** issues exist, tell the user explicitly and ask whether they
   want to fix them before proceeding. Do not silently skip over them.

   If all issues are advisory or the list is empty, note it briefly and continue.

5. **Rate the proposed commits 1–10.** Consider:
   - `git-smart-commit` stages changes at individual **hunk** level — a file
     CAN be split across commits when its hunks are unrelated. Penalise groupings
     where unrelated hunks are lumped together (even from the same file), wrong
     commit type, or vague subject lines.
   - Do NOT factor lint issues into the commit grouping score — those are
     evaluated separately in step 4.

   If the rating is **below 7**, re-run with feedback and overwrite the saved plan:
   ```
   git-smart-commit --json --save-plan /tmp/gsc-plan.json [passthrough flags] --critique "<specific feedback>"
   ```
   Display the revised proposals instead (repeat steps 3–5).

6. If `--dry-run` was in `$ARGUMENTS`, stop here.

7. Otherwise ask: **"Proceed with these commits? [y/N/t]"**
   - **Yes** (or if `--yes` was in `$ARGUMENTS`) → execute the saved plan, bypassing
     the TUI (pass `--repo` if it was in the passthrough flags):
     ```
     git-smart-commit --plan /tmp/gsc-plan.json --yes [--repo PATH if specified]
     ```
   - **t (TUI)** → tell the user to run this command directly in their terminal to
     get the interactive two-panel review UI (navigate with ↑↓/jk, edit subjects
     with `e`, confirm with `y`, cancel with `n`/`q`):
     ```
     git-smart-commit --plan /tmp/gsc-plan.json [--repo PATH if specified]
     ```
   - **No** → ask what the user wants to change; they can re-invoke with adjusted flags
