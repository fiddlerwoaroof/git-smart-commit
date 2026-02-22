#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
# ]
# ///
"""
eval.py: Evaluate a model on the git-smart-commit validation set.

Calls the model (via Ollama or any OpenAI-compatible API) on each validation
example and compares the output to teacher labels.

Metrics computed:
  - file_grouping_f1:  average Jaccard similarity between predicted and
                       labeled file sets across matched commit pairs
  - type_accuracy:     fraction of commits where predicted type matches label
  - junk_routing:      fraction of injected junk files routed to .gitignore
                       rather than appearing in a commit
  - parse_rate:        fraction of examples where the model returned valid JSON

Usage:
    eval.py --model MODEL [--val FILE] [--api-url URL] [--api-key KEY]
            [--concurrency N] [--limit N] [--output FILE] [--verbose]
"""

import argparse
import asyncio
import json
import re
import sys
import textwrap
from pathlib import Path

import httpx


# ── Prompt construction (mirrors label-training-data and prepare-sft-data) ────

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

    Rules:
    - CRITICAL: You can only commit WHOLE FILES. Partial staging is NOT supported.
    - Keep related changes together (same feature, same module, same concern).
    - Separate unrelated concerns into different commits.
    - Dependency/lockfile changes belong with the commit that caused them.
    - Test files belong with the code they test.
    - Use strict conventional commit format:
        subject: type(scope): short description  (under 72 chars)
        body: 4-10 lines, plain text, wrapped at 80 chars, no markdown
    - Types: feat > fix > refactor > chore (and docs, test, style, build, ci)
    - Do not repeat the commit type in the subject description.
    - Write body content based only on what you observe in the diff.
""")

JSON_INSTRUCTION = textwrap.dedent("""\

    OUTPUT FORMAT: Output ONLY a JSON object — no tool calls, no markdown fences,
    no explanation. Schema:
    {
      "commits": [
        {
          "subject": "type(scope): short description (under 72 chars)",
          "files": ["repo-relative/path/to/file.py"],
          "body": "4-10 line plain text commit body, no markdown",
          "issues": [{"message": "description", "path": "file.py"}]
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
            "body": "Removed unnecessary imports.",
            "issues": [],
        }
    ],
    "gitignore": ["*.pyc", "__pycache__/"],
}


def build_messages(example: dict, max_diff_chars: int = 30000) -> list[dict]:
    """Build the messages list for one eval example."""
    system = SYSTEM_PROMPT + JSON_INSTRUCTION
    file_list = "\n".join(f"  - {f}" for f in example["changed_files"])
    sample_json = json.dumps(SAMPLE_OUTPUT, indent=2)
    diff = example["diff"]
    if len(diff) > max_diff_chars:
        diff = diff[:max_diff_chars] + f"\n... (truncated)"
    user = (
        f"Sample Output:\n{sample_json}\n\n"
        f"Changed files:\n{file_list}\n\n"
        f"Diffs:\n{diff}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ── JSON extraction ───────────────────────────────────────────────────────────

def extract_json_object(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found")
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
    raise ValueError("Incomplete JSON object")


# ── API call ──────────────────────────────────────────────────────────────────

async def call_model(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    messages: list[dict],
    api_url: str,
    api_key: str,
    model: str,
) -> dict | None:
    """Call the model and return parsed JSON result, or None on failure."""
    async with semaphore:
        try:
            response = await client.post(
                f"{api_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": 4096,
                },
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"] or ""
            raw = extract_json_object(content)
            result = json.loads(raw)
            if not isinstance(result.get("commits"), list):
                return None
            return result
        except Exception:
            return None


# ── Metrics ───────────────────────────────────────────────────────────────────

def commit_type(subject: str) -> str:
    """Extract the type prefix from a conventional commit subject."""
    m = re.match(r"^([a-z]+)[\(:]", subject.strip())
    return m.group(1) if m else ""


def file_grouping_f1(pred_commits: list[dict], label_commits: list[dict]) -> float:
    """Average best-match Jaccard similarity between predicted and labeled commits.

    For each label commit, find the predicted commit with maximum Jaccard
    similarity and average those scores.
    """
    if not label_commits:
        return 1.0 if not pred_commits else 0.0

    scores = []
    for lc in label_commits:
        label_files = set(lc.get("files", []))
        if not label_files:
            continue
        best = 0.0
        for pc in pred_commits:
            pred_files = set(pc.get("files", []))
            if not pred_files:
                continue
            inter = len(label_files & pred_files)
            union = len(label_files | pred_files)
            jacc = inter / union if union else 0.0
            best = max(best, jacc)
        scores.append(best)

    return sum(scores) / len(scores) if scores else 0.0


def type_accuracy(pred_commits: list[dict], label_commits: list[dict]) -> float:
    """Fraction of label commits whose type is matched by the best predicted commit."""
    if not label_commits:
        return 1.0

    matches = 0
    for lc in label_commits:
        ltype = commit_type(lc.get("subject", ""))
        if not ltype:
            continue
        for pc in pred_commits:
            if commit_type(pc.get("subject", "")) == ltype:
                matches += 1
                break

    return matches / len(label_commits)


def junk_routing_score(
    pred: dict,
    injected_files: list[str],
) -> float | None:
    """Fraction of injected junk files that appear in gitignore but not in commits.

    Returns None if there are no injected files (metric not applicable).
    """
    if not injected_files:
        return None

    committed_files: set[str] = set()
    for c in pred.get("commits", []):
        committed_files.update(c.get("files", []))

    gitignore_patterns = pred.get("gitignore", [])

    def is_gitignored(filename: str) -> bool:
        name = Path(filename).name
        for pattern in gitignore_patterns:
            if pattern.startswith("*"):
                suffix = pattern[1:]
                if filename.endswith(suffix) or name.endswith(suffix):
                    return True
            elif pattern.endswith("/"):
                if ("/" + pattern[:-1] + "/") in ("/" + filename) or filename.startswith(pattern[:-1]):
                    return True
            else:
                if name == pattern or filename == pattern or filename.endswith("/" + pattern):
                    return True
        return False

    correctly_routed = sum(
        1 for f in injected_files
        if f not in committed_files and is_gitignored(f)
    )
    return correctly_routed / len(injected_files)


# ── Per-example evaluation ────────────────────────────────────────────────────

async def eval_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    example: dict,
    args: argparse.Namespace,
) -> dict:
    """Evaluate one example. Returns a result dict."""
    messages = build_messages(example, max_diff_chars=30000)
    label = example.get("label", {})
    label_commits = label.get("commits", [])
    injected_files = example.get("injected_files", [])

    pred = await call_model(
        client, semaphore, messages,
        api_url=args.api_url,
        api_key=args.api_key,
        model=args.model,
    )

    result: dict = {
        "repo_url": example.get("repo_url", ""),
        "languages": example.get("languages", []),
        "noise_injected": example.get("noise_injected", False),
        "parsed": pred is not None,
    }

    if pred is not None:
        result["file_grouping_f1"] = file_grouping_f1(pred.get("commits", []), label_commits)
        result["type_accuracy"] = type_accuracy(pred.get("commits", []), label_commits)
        junk = junk_routing_score(pred, injected_files)
        if junk is not None:
            result["junk_routing"] = junk
        if args.verbose:
            result["predicted"] = pred
            result["label"] = label

    return result


# ── Aggregation and reporting ─────────────────────────────────────────────────

def print_report(results: list[dict]) -> None:
    n = len(results)
    parsed = [r for r in results if r["parsed"]]
    parse_rate = len(parsed) / n if n else 0

    fg_scores = [r["file_grouping_f1"] for r in parsed if "file_grouping_f1" in r]
    type_scores = [r["type_accuracy"] for r in parsed if "type_accuracy" in r]
    junk_scores = [r["junk_routing"] for r in parsed if "junk_routing" in r]

    def avg(xs: list[float]) -> str:
        return f"{sum(xs)/len(xs):.3f}" if xs else "n/a"

    print(f"\n{'─'*50}")
    print(f"Evaluation Results  ({n} examples)")
    print(f"{'─'*50}")
    print(f"  Parse rate:          {parse_rate:.1%}  ({len(parsed)}/{n})")
    print(f"  File grouping F1:    {avg(fg_scores)}  (n={len(fg_scores)})")
    print(f"  Type accuracy:       {avg(type_scores)}  (n={len(type_scores)})")
    print(f"  Junk routing:        {avg(junk_scores)}  (n={len(junk_scores)})")
    print(f"{'─'*50}\n")

    # Breakdown by language
    lang_groups: dict[str, list[dict]] = {}
    for r in parsed:
        lang = (r.get("languages") or ["unknown"])[0]
        lang_groups.setdefault(lang, []).append(r)

    if len(lang_groups) > 1:
        print("  By language:")
        for lang, group in sorted(lang_groups.items(), key=lambda x: -len(x[1])):
            fg = [r["file_grouping_f1"] for r in group if "file_grouping_f1" in r]
            print(f"    {lang:<20} n={len(group):<5} file_f1={avg(fg)}")
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

async def run(args: argparse.Namespace) -> None:
    examples: list[dict] = []
    with open(args.val) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed line {lineno}: {e}", file=sys.stderr)

    if args.limit:
        examples = examples[: args.limit]

    print(f"Evaluating {len(examples)} examples with model: {args.model}", file=sys.stderr)

    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(connect=30.0, read=180.0, write=30.0, pool=5.0)

    results: list[dict] = []
    lock = asyncio.Lock()
    done_count = 0

    async def eval_and_collect(ex: dict) -> None:
        nonlocal done_count
        result = await eval_one(client, semaphore, ex, args)
        async with lock:
            results.append(result)
            done_count += 1
            pct = done_count / len(examples) * 100
            print(
                f"\r  {done_count}/{len(examples)} ({pct:.0f}%)  "
                f"parse_rate={sum(1 for r in results if r['parsed'])/len(results):.1%}    ",
                end="", file=sys.stderr, flush=True,
            )

    async with httpx.AsyncClient(timeout=timeout) as client:
        await asyncio.gather(*[eval_and_collect(ex) for ex in examples])

    print(file=sys.stderr)
    print_report(results)

    if args.output:
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Detailed results written to: {args.output}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a model on the git-smart-commit validation set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--val", default="val.jsonl",
        help="Validation JSONL from prepare-sft-data (default: val.jsonl)",
    )
    parser.add_argument(
        "--model", required=True,
        help="Model name to evaluate (e.g. qwen3-coder:30b-a3b-q8_0)",
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
        "--concurrency", type=int, default=4,
        help="Max concurrent requests (default: 4)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate at most N examples (default: all)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Write detailed per-example results to JSONL file",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Include predicted and label in output JSONL",
    )
    args = parser.parse_args()

    if not Path(args.val).exists():
        print(f"Error: validation file not found: {args.val}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
