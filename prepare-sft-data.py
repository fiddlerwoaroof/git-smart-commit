#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
prepare-sft-data: Convert labeled training examples into ChatML SFT format.

Reads labeled-training-data.jsonl (output of label-training-data), builds
system/user/assistant message triples, optionally generates augmented noisy
variants, and writes stratified train/val splits.

Usage:
    prepare-sft-data [--input FILE] [--train FILE] [--val FILE]
                     [--val-fraction FLOAT] [--n-augment N]
                     [--max-diff-chars N] [--seed N] [--dry-run]

Output schema (one JSON object per line):
{
  "messages": [
    {"role": "system",    "content": "..."},
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "{\"commits\": [...], \"gitignore\": [...]}"}
  ]
}
"""

import argparse
import json
import random
import sys
import textwrap
from pathlib import Path


# ── System prompt (verbatim from git-smart-commit) ────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a senior software engineer helping organize messy working-tree changes
    into clean, logical git commits.

    Given a list of changed files and their diffs, your job is to group them into
    one or more commits. Each commit should represent a single logical change
    (e.g. "add feature X", "fix bug in Y", "update dependencies",
    "refactor Z").

    You do not need to commit every file. Skip junk files (editor backups,
    build artifacts, OS metadata). Instead, collect suggested .gitignore patterns
    for them in the gitignore argument.

    You also actively look for common coding issues that a linter
    would catch and code smells such as using conditionals where
    polymorphism is more appropriate or violations of the Law of
    Demeter: misused APIs, suspicious code patterns, etc. For example:

    ```java
    // missing if braces
    if (a == 1) # wrong, add an issue "missing braces for if statement"
       b;
       c;
    d;
    ```

    ```c
    // incorrect arguments for well-known functions and Constructors
    printf(1); # wrong, add an issue "printf called with invalid arguments"
    ```

    ```python
    // Wrong keyword for the programming language
    if True:
        throw new Exception("foo") # wrong, add an issue "invalid keywords for python"

    ```

    ```python
    # Using exceptions for control flow instead of sys.exit
    if answer == "no":
        raise Exception("Cancelled.")  # wrong: should be sys.exit(0) or return
    ```

    Any detected issues should be added to the list of issues in a
    particular commit.

    Rules:
    - CRITICAL: You can only commit WHOLE FILES. Partial staging is NOT supported. If a single file contains multiple unrelated changes, you MUST group them together into one single commit. Do not split a file across multiple commits.
    - Keep related changes together (same feature, same module, same concern).
    - Separate unrelated concerns into different commits, provided they do not violate the whole-file rule above.
    - Dependency/lockfile changes belong with the commit that caused them.
    - Test files belong with the code they test.
    - Use strict conventional commit format:
        subject: type(scope): short description  (under 72 chars)
        body: 4-10 lines, plain text, wrapped at 80 chars, no markdown
    - Type Definitions:
        * feat: Adds a net-new capability, flag, or behavior (e.g., adding .venv support, new CLI flags). Use this even for personal tools/dotfiles!
        * fix: Resolves a bug, crash, or incorrect behavior.
        * refactor: Structural changes that DO NOT change external behavior. If it adds a feature, it is a 'feat'.
        * chore: Routine maintenance, dependency bumps, or minor environment tweaks with no new logic.
        * (Other types: docs, test, style, build, ci)
    - feat is the highest priority commit type
    - chore is the lowest priority commit type
    - Do not repeat the commit type in the subject description (e.g., avoid "refactor(tools): refactor the parser").
    - Write body content based only on what you observe in the diff.
      Do not reference issue numbers or details not visible in the changes.
    - In the issues field, call out any bugs, incorrect API usage, or suspicious
      patterns you observe in the diff. Examples: wrong number of arguments,
      misused stdlib functions, unreachable code, obvious logic errors. Be specific:
      include the file, the offending line or pattern, and why it's wrong.
      Leave issues empty only if you find nothing suspicious.
    - Watch specifically for arguments passed to constructors or functions that
      don't accept them (e.g. Exception() does not accept a file= keyword argument).
""")

JSON_INSTRUCTION = textwrap.dedent("""\

    OUTPUT FORMAT: Output ONLY a JSON object — no tool calls, no markdown fences,
    no explanation. The object must match this schema exactly:
    {
      "commits": [
        {
          "subject": "type(scope): short description (under 72 chars)",
          "files": ["repo-relative/path/to/file.py"],
          "body": "4-10 line plain text commit body, no markdown",
          "issues": [
            {"message": "brief description with line info", "path": "file.py"}
          ]
        }
      ],
      "gitignore": ["*.pyc", ".DS_Store"]
    }
""")

SAMPLE_OUTPUT = {
    "commits": [
        {
            "subject": "refactor(main): cleanup imports",
            "files": ["main.py"],
            "body": "Removed unnecessary imports to improve load time and code cleanliness.",
            "issues": [{"message": "Unused import 'os' on line 1", "path": "main.py"}],
        }
    ],
    "gitignore": ["*.pyc", "__pycache__/"],
}


# ── Junk file templates for noise augmentation ────────────────────────────────

# (filename, kind, content_or_None, gitignore_pattern)
JUNK_TEMPLATES: list[tuple[str, str, str | None, str]] = [
    (".DS_Store", "binary", None, ".DS_Store"),
    ("src/.DS_Store", "binary", None, ".DS_Store"),
    ("__pycache__/main.cpython-311.pyc", "binary", None, "__pycache__/"),
    ("src/__pycache__/utils.cpython-311.pyc", "binary", None, "__pycache__/"),
    (".pytest_cache/v/cache/lastfailed", "text", "{}", ".pytest_cache/"),
    (".pytest_cache/README.md", "text", "# pytest cache\n", ".pytest_cache/"),
    ("main.py.swp", "binary", None, "*.swp"),
    (".idea/workspace.xml", "text", '<project version="4"/>\n', ".idea/"),
    (".vscode/settings.json", "text", '{"editor.formatOnSave": true}\n', ".vscode/"),
    ("dist/bundle.js.map", "binary", None, "*.map"),
    (".env.local", "text", "NEXT_PUBLIC_API_URL=http://localhost:3000\nDEBUG=true\n", ".env.local"),
    ("secrets.env", "text", "API_KEY=dev_only_do_not_commit\n", "*.env"),
    ("npm-debug.log", "text", "0 verbose cli /usr/local/bin/npm\n", "npm-debug.log"),
    ("yarn-error.log", "text", "error An unexpected error occurred.\n", "yarn-error.log"),
    ("Thumbs.db", "binary", None, "Thumbs.db"),
    ("desktop.ini", "text", "[.ShellClassInfo]\nIconResource=folder.ico\n", "desktop.ini"),
]


def _make_junk_diff(filename: str, kind: str, content: str | None) -> str:
    if kind == "binary":
        return (
            f"diff --git a/{filename} b/{filename}\n"
            f"new file mode 100644\n"
            f"index 0000000..deadbeef\n"
            f"Binary files /dev/null and b/{filename} differ\n"
        )
    lines = (content or "").splitlines() or [""]
    added = "\n".join(f"+{line}" for line in lines)
    return (
        f"diff --git a/{filename} b/{filename}\n"
        f"new file mode 100644\n"
        f"index 0000000..deadbeef\n"
        f"--- /dev/null\n"
        f"+++ b/{filename}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n"
        f"{added}\n"
    )


def make_augmented(example: dict, rng: random.Random) -> dict | None:
    """Return a new example with noise injected, or None if nothing new was added.

    The label commits are preserved unchanged. Gitignore is extended with
    patterns for the injected files. No teacher API call needed.
    """
    n = rng.randint(1, 3)
    chosen = rng.choices(JUNK_TEMPLATES, k=n)

    existing_files = set(example["changed_files"])
    injected_files: list[str] = []
    extra_diffs: list[str] = []
    extra_patterns: list[str] = []

    for filename, kind, content, pattern in chosen:
        if filename in existing_files:
            continue
        injected_files.append(filename)
        extra_diffs.append(_make_junk_diff(filename, kind, content))
        extra_patterns.append(pattern)
        existing_files.add(filename)

    if not injected_files:
        return None

    existing_gitignore = set(example["label"]["gitignore"])
    new_gitignore = sorted(existing_gitignore | set(extra_patterns))

    return {
        **example,
        "changed_files": example["changed_files"] + injected_files,
        "diff": example["diff"] + "\n" + "\n".join(extra_diffs),
        "noise_injected": True,
        "injected_files": injected_files,
        "is_augmented": True,
        "label": {
            **example["label"],
            "gitignore": new_gitignore,
        },
    }


# ── SFT message construction ──────────────────────────────────────────────────

def build_user_message(example: dict, max_diff_chars: int) -> str:
    file_list = "\n".join(f"  - {f}" for f in example["changed_files"])
    sample_json = json.dumps(SAMPLE_OUTPUT, indent=2)
    diff = example["diff"]
    if len(diff) > max_diff_chars:
        diff = diff[:max_diff_chars] + f"\n\n... (diff truncated at {max_diff_chars} chars)"
    return (
        f"Sample Output:\n{sample_json}\n\n"
        f"Changed files:\n{file_list}\n\n"
        f"Diffs:\n{diff}"
    )


def example_to_sft(example: dict, max_diff_chars: int) -> dict | None:
    """Convert a labeled example to a ChatML message triple.

    Returns None if the label is missing or malformed.
    """
    label = example.get("label")
    if not label:
        return None
    commits = label.get("commits")
    if not commits or not isinstance(commits, list):
        return None

    system = SYSTEM_PROMPT + JSON_INSTRUCTION
    user = build_user_message(example, max_diff_chars)
    assistant = json.dumps(
        {"commits": commits, "gitignore": label.get("gitignore", [])},
        separators=(",", ":"),
    )

    return {"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


# ── Stratified train/val split ────────────────────────────────────────────────

def stratified_split(
    examples: list[dict],
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split into train/val, stratified by (first language, repo_url).

    Augmented examples always go to train to avoid leakage.
    """
    # Augmented variants always go to train
    augmented = [ex for ex in examples if ex.get("is_augmented")]
    base = [ex for ex in examples if not ex.get("is_augmented")]

    # Group base examples by stratum
    strata: dict[str, list[dict]] = {}
    for ex in base:
        langs = ex.get("languages") or ["unknown"]
        lang = langs[0]
        repo = ex.get("repo_url", "unknown")
        key = f"{lang}:{repo}"
        strata.setdefault(key, []).append(ex)

    rng = random.Random(seed)
    train: list[dict] = []
    val: list[dict] = []

    for stratum_examples in strata.values():
        rng.shuffle(stratum_examples)
        n_val = max(1, round(len(stratum_examples) * val_fraction))
        # Single-example strata go to train
        if len(stratum_examples) == 1:
            train.extend(stratum_examples)
        else:
            val.extend(stratum_examples[:n_val])
            train.extend(stratum_examples[n_val:])

    train.extend(augmented)
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert labeled JSONL to ChatML SFT format with train/val split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", default="labeled-training-data.jsonl",
        help="Labeled JSONL from label-training-data (default: labeled-training-data.jsonl)",
    )
    parser.add_argument(
        "--train", default="train.jsonl",
        help="Output training JSONL (default: train.jsonl)",
    )
    parser.add_argument(
        "--val", default="val.jsonl",
        help="Output validation JSONL (default: val.jsonl)",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.1,
        help="Fraction of base examples to use for validation (default: 0.1)",
    )
    parser.add_argument(
        "--n-augment", type=int, default=1,
        help="Augmented noisy variants per clean example (default: 1, 0 to disable)",
    )
    parser.add_argument(
        "--max-diff-chars", type=int, default=30000,
        help="Truncate diffs longer than this (default: 30000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print stats without writing output files",
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Load labeled examples
    raw: list[dict] = []
    skipped = 0
    with open(args.input) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Warning: skipping malformed line {lineno}: {e}", file=sys.stderr)

    print(f"Loaded {len(raw)} labeled examples.", file=sys.stderr)

    # Convert to SFT format and generate augmented variants
    rng = random.Random(args.seed)
    sft_examples: list[dict] = []
    n_skipped_bad_label = 0
    n_augmented = 0

    for idx, ex in enumerate(raw):
        sft = example_to_sft(ex, args.max_diff_chars)
        if sft is None:
            n_skipped_bad_label += 1
            continue

        # Attach metadata for stratification
        sft["languages"] = ex.get("languages", [])
        sft["repo_url"] = ex.get("repo_url", "")
        sft["is_augmented"] = ex.get("is_augmented", False)

        sft_examples.append(sft)

        # Generate augmented variants for clean (non-noisy) examples
        if args.n_augment > 0 and not ex.get("noise_injected"):
            for aug_idx in range(args.n_augment):
                aug_ex = make_augmented(ex, random.Random(args.seed ^ (idx * 31 + aug_idx)))
                if aug_ex is None:
                    continue
                aug_sft = example_to_sft(aug_ex, args.max_diff_chars)
                if aug_sft is None:
                    continue
                aug_sft["languages"] = aug_ex.get("languages", [])
                aug_sft["repo_url"] = aug_ex.get("repo_url", "")
                aug_sft["is_augmented"] = True
                sft_examples.append(aug_sft)
                n_augmented += 1

    if n_skipped_bad_label:
        print(f"Skipped {n_skipped_bad_label} examples with missing/malformed labels.", file=sys.stderr)
    print(f"Generated {n_augmented} augmented variants.", file=sys.stderr)
    print(f"Total SFT examples: {len(sft_examples)}", file=sys.stderr)

    # Stratified split
    train, val = stratified_split(sft_examples, val_fraction=args.val_fraction, seed=args.seed)

    # Strip metadata fields before writing
    meta_keys = {"languages", "repo_url", "is_augmented"}

    def strip_meta(ex: dict) -> dict:
        return {k: v for k, v in ex.items() if k not in meta_keys}

    print(f"Train: {len(train)}  Val: {len(val)}", file=sys.stderr)

    if args.dry_run:
        # Show a sample assistant message
        if train:
            sample = train[0]
            assistant_content = sample["messages"][2]["content"]
            print(f"\nSample assistant turn ({len(assistant_content)} chars):")
            print(assistant_content[:500])
        return

    with open(args.train, "w") as f:
        for ex in train:
            f.write(json.dumps(strip_meta(ex)) + "\n")
    print(f"Wrote {len(train)} training examples → {args.train}", file=sys.stderr)

    with open(args.val, "w") as f:
        for ex in val:
            f.write(json.dumps(strip_meta(ex)) + "\n")
    print(f"Wrote {len(val)} validation examples → {args.val}", file=sys.stderr)


if __name__ == "__main__":
    main()
