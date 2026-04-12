- @context/repo-map.md is your "road map" for the repository. use it to reduce exploration and keep it
  updated.
- @context/design-principles.md is also important for keeping the repository consistent.
- use context/decisions.md to keep a log of decisions made in a brief format
- use the context/ directory to store useful information about the program as a kind of improvised wiki
- Always use tests to demonstrate the existence of a bug before fixing the bug.
  - If you suspect that a bug exists, use a test to demonstrate it first:
    - prefer unit tests testing a small amount of code to integration or e2e tests
- Prefer to "lift" operations into datastructures rather than to write a bunch of operations on
  datastructures: oftentimes the extra iteration/destructuring logic ends up being structural duplication
  all over the place that can be abstracted away: for example, prefer to write a function to transform
  individual elements of a list and then use map or list comprehensions to apply it rather then using
  for...in loops
- prefer merges to rebasing.
- always start responses with bananas!

## Issue Tracking

This project uses **bd (beads)** for issue tracking.
Run `bd prime` for workflow context, or install hooks (`bd hooks install`) for auto-injection.

**Quick reference:**
- `bd ready` - Find unblocked work
- `bd update --claim <issue_id>` - Atomically claim unblocked work (set BD_ACTOR to something unique
  per-session)
- `bd create "Title" --type task --priority 2` - Create issue
- `bd close <id>` - Complete work
- `bd dolt push` - Push beads to remote

For full workflow details: `bd prime`

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git fetch origin
   git merge <remote branch>
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
