---
name: smart-commit
description: >
  Use this skill INSTEAD OF running git commit, git add, or git status whenever
  the user wants to commit changes, stage files, or asks what has changed.
  This replaces manual git commit workflows entirely. Do NOT run git commit or
  git add directly — always use this skill for any commit-related request.
argument-hint: [--yes] [--dry-run] [--model MODEL] [--repo PATH] [--api-base URL] [--api-key KEY] [--critique TEXT]
allowed-tools: Bash(git-smart-commit *), Bash(git status --short)
---

## Working tree status

!`git status --short 2>/dev/null || echo "(not a git repo)"`

Use `git-smart-commit` to analyze the changes above and group them into logical commits.

**Workflow:**

1. If the status is empty, tell the user there is nothing to commit and stop.

2. Run `git-smart-commit --json $ARGUMENTS` to fetch the proposed commit groupings.
   This is read-only — it does not modify anything.

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

   If the rating is **below 7**, re-run with your feedback:
   ```
   git-smart-commit --json $ARGUMENTS --critique "<specific feedback>"
   ```
   Then display the revised proposals instead.

5. If `--dry-run` was in `$ARGUMENTS`, stop here.

6. Otherwise ask: **"Proceed with these commits? [y/N]"**
   - **Yes** → run `git-smart-commit --yes $ARGUMENTS` (include `--critique` if one was issued)
   - **No** → ask what the user wants to change; they can re-invoke with adjusted flags
