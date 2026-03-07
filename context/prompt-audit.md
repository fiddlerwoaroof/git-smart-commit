# Prompt Duplication Audit (gsc-hei)

Three prompts audited:
- `SYSTEM_PROMPT` — non-agentic mode (L98–193)
- `AGENTIC_SYSTEM_PROMPT` — agentic mode (L196–338)
- `MERGE_PROMPT` — commit-merge step (L956–975)

---

## 1. Verbatim-Identical Blocks

### 1a. Opening sentence
Both SYSTEM_PROMPT and AGENTIC_SYSTEM_PROMPT open with:

> "You are a senior software engineer helping organize messy working-tree changes into clean, logical git commits."

| Prompt | Lines |
|--------|-------|
| SYSTEM_PROMPT | 99–100 |
| AGENTIC_SYSTEM_PROMPT | 197–198 |

### 1b. Junk-file paragraph
Exact duplicate:

> "You do not need to commit every file. Skip junk files (editor backups, build artifacts, OS metadata). Collect .gitignore patterns in the gitignore argument, and list skipped hunk IDs in the skipped_hunks argument of finalize_commits."

| Prompt | Lines |
|--------|-------|
| SYSTEM_PROMPT | 107–110 |
| AGENTIC_SYSTEM_PROMPT | 252–255 |

Note: SYSTEM_PROMPT refers to `finalize_commits` here even though its own mode uses `propose_commits` — minor inconsistency.

### 1c. Code-issue detection examples (~40 lines)
Four example blocks (Java missing braces, C printf args, Python wrong keyword, Python exceptions-for-control-flow) plus the "Any detected issues should be added..." sentence:

| Prompt | Lines |
|--------|-------|
| SYSTEM_PROMPT | 112–144 |
| AGENTIC_SYSTEM_PROMPT | 257–289 |

### 1d. The entire Rules section
Every rule bullet is verbatim-identical between the two prompts:

| Rule | SYSTEM_PROMPT | AGENTIC_SYSTEM_PROMPT |
|------|---------------|----------------------|
| CRITICAL JSON | 147 | 292 |
| Hunk IDs ("src/main.py::1") | 148–149 | 293–294 |
| Split hunks across commits | 150–151 | 295–296 |
| Keep related hunks together | 152 | 297 |
| Prefer 2-5 cohesive commits | 153–158 | 298–303 |
| Dependency/lockfile | 159 | 304 |
| Test files with code | 160 | 305 |
| Conventional commit format | 161–163 | 306–308 |
| Type definitions (feat/fix/refactor/chore) | 164–169 | 309–314 |
| feat highest / chore lowest priority | 170–171 | 315–316 |
| No repeat type in subject | 172 | 317 |
| Body from diff only | 173–174 | 318–319 |
| Issues field guidance | 175–179 | 320–324 |
| Watch for wrong constructor args | 180–181 | 325–326 |
| BREAKING CHANGE detection | 182–187 | 327–332 |

---

## 2. Semantically-Equivalent but Differently-Worded

### 2a. Closing instruction
- **SYSTEM_PROMPT** (189–192): "You have access to a read_file tool. Use it when the diff alone is not enough context... Call read_file as many times as needed, then call `propose_commits` when ready."
- **AGENTIC_SYSTEM_PROMPT** (334–337): "Investigate the changes thoroughly before emitting commits. Once you have a clear picture of how the changes relate, emit commits for each logical grouping. Continue until all hunks are assigned or remaining hunks should be ignored, then call finalize_commits."

Same intent (investigate then finalize), different API names and different framing.

### 2b. Commit type priority (MERGE_PROMPT)
- **MERGE_PROMPT** (965–966): "If any commit is a feat type, it must take priority over the others... chore should only be used if no other category applies."
- **SYSTEM_PROMPT / AGENTIC** (170–171 / 315–316): "feat is the highest priority commit type / chore is the lowest priority commit type"

Same semantics, different wording.

---

## 3. Content Unique to Each Prompt

### SYSTEM_PROMPT only
- Task framing paragraph (L102–106): "Given a list of changed files and their diffs, your job is to group them into one or more commits..."
- read_file availability note at end (L189–192)
- Uses `propose_commits` API (not `emit_commit`/`finalize_commits`)

### AGENTIC_SYSTEM_PROMPT only
- Tool list with descriptions: read_file, get_diff, get_git_log, search_diff, query_tool_result, emit_commit, finalize_commits (L200–217)
- Context management section (L219–228): summarization, query_tool_result rules
- Investigation strategy section (L230–250): bias-to-action, skip-investigation heuristics, tool-use guidance

### MERGE_PROMPT only
- Merge-specific instructions: deduplicate hunks/issues, write new coherent subject/body
- `merge_commits` tool reference
- Breaking-change merging rule (combine descriptions)
- "Do NOT just concatenate the subjects" instruction

---

## 4. Refactoring Opportunities (drives downstream issues)

| Constant | Content | Downstream Issue |
|----------|---------|-----------------|
| `SHARED_RULES` | Entire Rules section (hunk IDs through BREAKING CHANGE) | gsc-pbe, gsc-zd7, gsc-b3k |
| `CODE_ISSUE_EXAMPLES` | ~40-line code example block + "Any detected issues..." sentence | gsc-952 |
| `SHARED_BREAKING_CHANGE` | BREAKING CHANGE detection rule | gsc-b3k |
| `SHARED_ISSUES_FIELD` | Issues field guidance + wrong-args watch | gsc-b3k |
| Harmonize MERGE_PROMPT | Align type-priority wording, add BREAKING CHANGE merging | gsc-dns |

Recommended extraction order:
1. `CODE_ISSUE_EXAMPLES` (gsc-952) — pure constant, no logic
2. `SHARED_RULES` combining conventional-commit format + type defs + hunk-grouping rules (gsc-pbe, gsc-zd7)
3. `SHARED_BREAKING_CHANGE` + `SHARED_ISSUES_FIELD` extracted from SHARED_RULES (gsc-b3k)
4. MERGE_PROMPT harmonization (gsc-dns) — depends on constants existing
