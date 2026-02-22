#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
# ]
# ///
"""
label-training-data: Call a teacher model to label harvested training examples.

Reads a harvested JSONL file (from harvest-training-data), constructs the same
prompt that git-smart-commit uses (system prompt + file list + diff), calls a
teacher model via OpenAI-compatible API, and saves labeled output.

Idempotent: skips examples already labeled in the output file.
Progress is saved incrementally so crashes don't lose work.

Usage:
    label-training-data --model MODEL [--input FILE] [--output FILE]
                        [--api-url URL] [--api-key KEY]
                        [--concurrency N] [--limit N]
                        [--noise-prob FLOAT] [--seed N] [--dry-run]

Output schema per line (original fields + label):
{
  "repo_url": "...", "diff": "...", "changed_files": [...], ...
  "noise_injected": false,
  "injected_files": [],
  "label": {
    "commits": [{"subject": "type(scope): desc", "files": [...], "body": "...", "issues": [...]}],
    "gitignore": ["*.pyc"],
    "model": "model-name",
    "usage": {"prompt_tokens": N, "completion_tokens": N, "total_tokens": N}
  }
}
"""

import argparse
import asyncio
import json
import random
import re
import sys
import textwrap
from pathlib import Path

import httpx


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


# ── Noise injection ────────────────────────────────────────────────────────────

# Each entry: (filename_template, kind, content_or_None)
# kind: "binary" | "text"
JUNK_TEMPLATES = [
    # macOS metadata
    (".DS_Store", "binary", None),
    ("src/.DS_Store", "binary", None),
    # Python artifacts
    ("__pycache__/main.cpython-311.pyc", "binary", None),
    ("src/__pycache__/utils.cpython-311.pyc", "binary", None),
    (".pytest_cache/v/cache/lastfailed", "text", "{}"),
    (".pytest_cache/README.md", "text", "# pytest cache\n"),
    # Editor swap / metadata
    ("main.py.swp", "binary", None),
    (".idea/workspace.xml", "text", '<project version="4"/>\n'),
    (".vscode/settings.json", "text", '{"editor.formatOnSave": true}\n'),
    # Build / dist artifacts
    ("dist/bundle.js.map", "binary", None),
    ("target/debug/incremental/.fingerprint", "binary", None),
    # Environment / secrets
    (".env.local", "text", "NEXT_PUBLIC_API_URL=http://localhost:3000\nDEBUG=true\n"),
    ("secrets.env", "text", "API_KEY=development_only_do_not_commit\n"),
    # Logs
    ("npm-debug.log", "text", "0 verbose cli /usr/local/bin/npm\n"),
    ("yarn-error.log", "text", "error An unexpected error occurred.\n"),
    # Windows / other OS
    ("Thumbs.db", "binary", None),
    ("desktop.ini", "text", "[.ShellClassInfo]\nIconResource=folder.ico\n"),
]


def _make_junk_diff(filename: str, kind: str, content: str | None) -> str:
    short_hash = "deadbeef"
    if kind == "binary":
        return (
            f"diff --git a/{filename} b/{filename}\n"
            f"new file mode 100644\n"
            f"index 0000000..{short_hash}\n"
            f"Binary files /dev/null and b/{filename} differ\n"
        )
    lines = (content or "").splitlines() or [""]
    added = "\n".join(f"+{line}" for line in lines)
    return (
        f"diff --git a/{filename} b/{filename}\n"
        f"new file mode 100644\n"
        f"index 0000000..{short_hash}\n"
        f"--- /dev/null\n"
        f"+++ b/{filename}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n"
        f"{added}\n"
    )


def inject_noise(example: dict, rng: random.Random) -> dict:
    """Return a copy of example with 1-3 random junk files added."""
    n = rng.randint(1, 3)
    chosen = rng.choices(JUNK_TEMPLATES, k=n)

    existing = set(example["changed_files"])
    injected: list[str] = []
    extra_diffs: list[str] = []

    for filename, kind, content in chosen:
        if filename in existing:
            continue
        injected.append(filename)
        extra_diffs.append(_make_junk_diff(filename, kind, content))
        existing.add(filename)

    if not injected:
        return {**example, "noise_injected": False, "injected_files": []}

    return {
        **example,
        "changed_files": example["changed_files"] + injected,
        "diff": example["diff"] + "\n" + "\n".join(extra_diffs),
        "noise_injected": True,
        "injected_files": injected,
    }


# ── Prompt construction ────────────────────────────────────────────────────────

def build_messages(example: dict) -> list[dict]:
    """Build the OpenAI messages list for one example."""
    system = SYSTEM_PROMPT + JSON_INSTRUCTION

    file_list = "\n".join(f"  - {f}" for f in example["changed_files"])
    sample_json = json.dumps(SAMPLE_OUTPUT, indent=2)

    user = (
        f"Sample Output:\n{sample_json}\n\n"
        f"Changed files:\n{file_list}\n\n"
        f"Diffs:\n{example['diff']}"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ── JSON extraction ────────────────────────────────────────────────────────────

def extract_json_object(text: str) -> str:
    """Extract the first complete JSON object from text.

    Strips markdown fences if present, then finds the outermost { ... }.
    """
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")

    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ValueError("Incomplete JSON object in response")


# ── API call ──────────────────────────────────────────────────────────────────

async def call_teacher(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    messages: list[dict],
    api_url: str,
    api_key: str,
    model: str,
    max_retries: int = 3,
) -> tuple[dict, dict]:
    """Call teacher model. Returns (parsed_result, usage_dict).

    Retries up to max_retries times on parse failure, injecting the error
    into the prompt so the model can self-correct.
    """
    current_messages = list(messages)
    last_error: str = ""

    async with semaphore:
        for attempt in range(1, max_retries + 1):
            if attempt > 1 and last_error:
                # Append error context so the model can self-correct
                current_messages = list(messages) + [
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response could not be parsed as JSON: {last_error}\n"
                            "Output ONLY a valid JSON object matching the schema. "
                            "No explanation, no markdown."
                        ),
                    }
                ]

            response = await client.post(
                f"{api_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": current_messages,
                    "temperature": 0.2,
                    "max_tokens": 4096,
                },
            )
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage", {})

            try:
                raw = extract_json_object(content)
                result = json.loads(raw)
                if not isinstance(result.get("commits"), list):
                    raise ValueError("'commits' must be a list")
                if not result["commits"]:
                    raise ValueError("'commits' list is empty")
                return result, usage
            except (ValueError, json.JSONDecodeError) as e:
                last_error = f"{e} | content[:300]={content[:300]!r}"
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Failed to parse after {max_retries} attempts: {last_error}"
                    )


# ── Idempotency ───────────────────────────────────────────────────────────────

def example_key(example: dict) -> str:
    return f"{example['repo_url']}@{example['base_commit']}..{example['head_commit']}"


def load_labeled_keys(output_path: str) -> set[str]:
    """Return the set of example keys already present in the output file."""
    keys: set[str] = set()
    path = Path(output_path)
    if not path.exists():
        return keys
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                if "label" in ex:
                    keys.add(example_key(ex))
            except json.JSONDecodeError:
                pass
    return keys


# ── Progress ──────────────────────────────────────────────────────────────────

class Progress:
    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self.failed = 0
        self.total_tokens = 0
        self._lock = asyncio.Lock()

    async def record(self, *, success: bool, tokens: int = 0) -> None:
        async with self._lock:
            if success:
                self.done += 1
                self.total_tokens += tokens
            else:
                self.failed += 1
            pct = (self.done + self.failed) / self.total * 100 if self.total else 0
            print(
                f"\r  {self.done + self.failed}/{self.total} ({pct:.0f}%)  "
                f"ok={self.done}  fail={self.failed}  "
                f"tokens={self.total_tokens:,}    ",
                end="",
                file=sys.stderr,
                flush=True,
            )


# ── Per-example worker ────────────────────────────────────────────────────────

async def process_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    out_lock: asyncio.Lock,
    out_f,
    example: dict,
    idx: int,
    args: argparse.Namespace,
    progress: Progress,
) -> None:
    # Deterministic per-example RNG so noise is reproducible regardless of
    # completion order.
    task_rng = random.Random(args.seed ^ (idx * 2654435761))

    if task_rng.random() < args.noise_prob:
        labeled_example = inject_noise(example, task_rng)
    else:
        labeled_example = {**example, "noise_injected": False, "injected_files": []}

    messages = build_messages(labeled_example)

    try:
        result, usage = await call_teacher(
            client,
            semaphore,
            messages,
            api_url=args.api_url,
            api_key=args.api_key,
            model=args.model,
        )
    except Exception as e:
        key = example_key(example)
        print(f"\n  [FAIL] {key}: {e}", file=sys.stderr)
        await progress.record(success=False)
        return

    labeled_example["label"] = {
        "commits": result.get("commits", []),
        "gitignore": result.get("gitignore", []),
        "model": args.model,
        "usage": usage,
    }

    async with out_lock:
        out_f.write(json.dumps(labeled_example) + "\n")
        out_f.flush()

    await progress.record(success=True, tokens=usage.get("total_tokens", 0))


# ── Orchestration ─────────────────────────────────────────────────────────────

async def run(args: argparse.Namespace) -> None:
    labeled_keys = load_labeled_keys(args.output)
    print(f"Already labeled: {len(labeled_keys)} examples", file=sys.stderr)

    examples: list[dict] = []
    with open(args.input) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Warning: skipping malformed line {lineno}: {e}", file=sys.stderr)

    print(f"Total examples in input: {len(examples)}", file=sys.stderr)

    pending = [ex for ex in examples if example_key(ex) not in labeled_keys]
    if args.limit:
        pending = pending[: args.limit]

    print(f"To label: {len(pending)} examples", file=sys.stderr)

    if not pending:
        print("Nothing to do.", file=sys.stderr)
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    out_lock = asyncio.Lock()
    progress = Progress(len(pending))

    timeout = httpx.Timeout(connect=30.0, read=180.0, write=30.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        with open(args.output, "a") as out_f:
            tasks = [
                process_one(
                    client, semaphore, out_lock, out_f,
                    ex, idx, args, progress,
                )
                for idx, ex in enumerate(pending)
            ]
            await asyncio.gather(*tasks)

    print(
        f"\nDone. Labeled {progress.done} examples, {progress.failed} failed.",
        file=sys.stderr,
    )
    if progress.total_tokens:
        print(f"Total tokens used: {progress.total_tokens:,}", file=sys.stderr)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label harvested training examples with a teacher model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", default="repo-dataset.jsonl",
        help="Input JSONL from harvest-training-data (default: repo-dataset.jsonl)",
    )
    parser.add_argument(
        "--output", default="labeled-training-data.jsonl",
        help="Output labeled JSONL (default: labeled-training-data.jsonl)",
    )
    parser.add_argument(
        "--model", required=True,
        help="Teacher model name (e.g. step-3.5-flash, deepseek-v3, qwen-plus)",
    )
    parser.add_argument(
        "--api-url", default="http://localhost:11434/v1",
        help="OpenAI-compatible API base URL (default: http://localhost:11434/v1)",
    )
    parser.add_argument(
        "--api-key", default="ollama",
        help="API key / bearer token (default: ollama)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=8,
        help="Max concurrent API requests (default: 8)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N pending examples (default: all)",
    )
    parser.add_argument(
        "--noise-prob", type=float, default=0.3,
        help="Probability of injecting junk files into an example (default: 0.3)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed for noise injection (default: 42)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be labeled without calling the API",
    )

    args = parser.parse_args()

    if args.dry_run:
        labeled_keys = load_labeled_keys(args.output)
        examples: list[dict] = []
        try:
            with open(args.input) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            examples.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except FileNotFoundError:
            print(f"Input file not found: {args.input}", file=sys.stderr)
            sys.exit(1)

        pending = [ex for ex in examples if example_key(ex) not in labeled_keys]
        if args.limit:
            pending = pending[: args.limit]

        print(f"Input:          {args.input} ({len(examples)} examples)")
        print(f"Output:         {args.output}")
        print(f"Already labeled:{len(labeled_keys)}")
        print(f"To label:       {len(pending)}")
        print(f"Model:          {args.model}")
        print(f"API URL:        {args.api_url}")
        print(f"Concurrency:    {args.concurrency}")
        print(f"Noise prob:     {args.noise_prob}")
        return

    if not Path(args.input).exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
