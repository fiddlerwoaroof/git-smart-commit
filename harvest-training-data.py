#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
harvest-training-data: Generate git-smart-commit training inputs from local repos.

Walks existing local git repositories, picks random commits, rewinds N commits
along first-parent, captures the diff, and writes ready-to-use JSONL training
inputs.

SAFE: Never modifies your working tree. All operations are read-only git
commands (log, diff, diff-tree, show, rev-list). No checkout, no stash, no
worktree.

The expensive frontier model step is NOT done here — this just produces inputs.
Noise file injection happens at reconstruction time (not here).

Usage:
    harvest-training-data [--repo-dirs DIR ...] [--output FILE] [--count N]
                          [--max-files MAX] [--max-diff-chars MAX]
                          [--min-commits MIN] [--max-commits MAX]
                          [--seed SEED] [--max-per-repo N] [--all-branches]
                          [--fetch] [--dry-run]

Each output line is a JSON object:
{
    "repo_url": "github.com/owner/repo",
    "branch": "main",
    "base_commit": "abc123...",
    "head_commit": "def456...",
    "n_commits": 3,
    "diff": "...",
    "changed_files": ["a.py", "b.py"],
    "original_commits": [{"sha": "...", "subject": "...", "files": [...]}],
    "existing_gitignore": ["*.pyc", "__pycache__/", ...],
    "languages": ["python", "rust"]
}

Reconstruction: clone from repo_url, checkout base_commit, then the diff
covers everything up to head_commit.
"""

import argparse
import glob
import json
import os
import random
import subprocess
import sys
from pathlib import Path


# ── Language detection ────────────────────────────────────────────────────────

EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".pyx": "python",
    ".pyi": "python",
    ".java": "java",
    ".kt": "kotlin",
    ".gradle": "java",
    ".lisp": "common-lisp",
    ".lsp": "common-lisp",
    ".cl": "common-lisp",
    ".asd": "common-lisp",
    ".clj": "clojure",
    ".cljs": "clojure",
    ".cljc": "clojure",
    ".edn": "clojure",
    ".rb": "ruby",
    ".rake": "ruby",
    ".gemspec": "ruby",
    ".rs": "rust",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".go": "go",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".ex": "elixir",
    ".exs": "elixir",
    ".hs": "haskell",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".el": "emacs-lisp",
    ".vim": "vim",
    ".lua": "lua",
    ".nix": "nix",
    ".scala": "scala",
    ".sbt": "scala",
    ".swift": "swift",
    ".ml": "ocaml",
    ".mli": "ocaml",
}

FILENAME_TO_LANGUAGE = {
    "cargo.toml": "rust",
    "cargo.lock": "rust",
    "pom.xml": "java",
    "build.gradle": "java",
    "build.gradle.kts": "kotlin",
    "gemfile": "ruby",
    "rakefile": "ruby",
    "setup.py": "python",
    "pyproject.toml": "python",
    "requirements.txt": "python",
    "project.clj": "clojure",
    "deps.edn": "clojure",
    "package.json": "javascript",
    "yarn.lock": "javascript",
    "go.mod": "go",
    "go.sum": "go",
    "flake.nix": "nix",
    "mix.exs": "elixir",
    "stack.yaml": "haskell",
    "cabal.project": "haskell",
}


def detect_languages(changed_files: list[str]) -> list[str]:
    """Detect programming languages from file extensions and known filenames."""
    langs = set()
    for f in changed_files:
        ext = Path(f).suffix.lower()
        if ext in EXTENSION_TO_LANGUAGE:
            langs.add(EXTENSION_TO_LANGUAGE[ext])
        name = Path(f).name.lower()
        if name in FILENAME_TO_LANGUAGE:
            langs.add(FILENAME_TO_LANGUAGE[name])
    return sorted(langs)


# ── Gitignore mining ──────────────────────────────────────────────────────────


def mine_gitignore(repo: Path, commit: str) -> list[str]:
    """Extract .gitignore patterns from the repo at a given commit.

    Checks the root .gitignore and any nested .gitignore files.
    All read-only: uses git show and git ls-tree.
    """
    patterns = []

    out = git_ok(["ls-tree", "-r", "--name-only", commit], repo)
    if not out:
        return []

    gitignore_files = [
        f for f in out.strip().splitlines() if Path(f).name == ".gitignore"
    ]

    for gi_path in gitignore_files:
        content = git_ok(["show", f"{commit}:{gi_path}"], repo)
        if not content:
            continue
        parent = str(Path(gi_path).parent)
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if parent != ".":
                patterns.append(f"{parent}/{line}")
            else:
                patterns.append(line)

    return patterns


# ── Git helpers ───────────────────────────────────────────────────────────────


def git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run a git command, return CompletedProcess.

    Uses surrogateescape for non-UTF-8 output (e.g. binary diffs that
    slip past git's binary detection).
    """
    return subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=False,  # raw bytes
        timeout=120,
    )


def git_ok(args: list[str], cwd: Path) -> str | None:
    """Run git command, return stdout as string on success, None on failure.

    Returns None if the output is not valid UTF-8 (binary content).
    """
    r = git(args, cwd)
    if r.returncode != 0:
        return None
    try:
        return r.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return None


def get_remote_url(repo: Path) -> str | None:
    """Extract a normalized repo identifier from remotes.

    Tries origin first, then upstream, then any remote.
    """
    for remote in ["origin", "upstream"]:
        out = git_ok(["remote", "get-url", remote], repo)
        if out:
            return normalize_url(out.strip())

    # Fall back to first available remote
    out = git_ok(["remote"], repo)
    if out:
        first_remote = out.strip().splitlines()[0].strip()
        if first_remote:
            out = git_ok(["remote", "get-url", first_remote], repo)
            if out:
                return normalize_url(out.strip())

    return None


def normalize_url(url: str) -> str:
    """Normalize a git URL to github.com/owner/repo form."""
    for prefix in ["https://", "http://", "git@", "ssh://git@", "ssh://"]:
        if url.startswith(prefix):
            url = url[len(prefix) :]
    # git@github.com:owner/repo.git -> github.com/owner/repo
    if ":" in url:
        host_part = url.split(":")[0]
        if "/" not in host_part:
            url = url.replace(":", "/", 1)
    url = url.removesuffix(".git")
    url = url.rstrip("/")
    return url


def fetch_all(repo: Path) -> bool:
    """Fetch all remotes. Returns True on success."""
    r = git(["fetch", "--all", "--quiet"], repo)
    return r.returncode == 0


def get_default_branch(repo: Path) -> str | None:
    """Get the default branch for the repo.

    Tries HEAD first, then common branch names.
    """
    out = git_ok(["symbolic-ref", "--short", "HEAD"], repo)
    if out:
        return out.strip()

    for branch in ["main", "master", "trunk", "develop", "stable", "production"]:
        r = git(["rev-parse", "--verify", branch], repo)
        if r.returncode == 0:
            return branch

    return None


def is_merge_commit(repo: Path, sha: str) -> bool:
    """Check if a commit is a merge commit by counting its parents."""
    out = git_ok(["cat-file", "-p", sha], repo)
    if not out:
        return False
    parent_count = sum(1 for line in out.splitlines() if line.startswith("parent "))
    return parent_count > 1


def get_first_parent_commits(
    repo: Path,
    branch: str | None = None,
    max_count: int = 500,
) -> list[str]:
    """Get commit SHAs along first-parent lineage."""
    cmd = ["log", "--first-parent", "--format=%H", f"--max-count={max_count}"]
    if branch:
        cmd.append(branch)
    out = git_ok(cmd, repo)
    if not out:
        return []
    return [line.strip() for line in out.strip().splitlines() if line.strip()]


def get_commit_info(repo: Path, sha: str) -> dict | None:
    """Get subject and changed files for a commit. Returns None for merges."""
    if is_merge_commit(repo, sha):
        return None

    out = git_ok(["show", "--no-patch", "--format=%s", sha], repo)
    if not out:
        return None
    subject = out.strip()

    out = git_ok(["diff-tree", "--no-commit-id", "-r", "--name-only", sha], repo)
    files = [f for f in (out or "").strip().splitlines() if f.strip()]

    return {"sha": sha, "subject": subject, "files": files}


def get_diff(repo: Path, base: str, head: str) -> str | None:
    """Get the full diff between two commits. Read-only."""
    out = git_ok(
        ["diff", "--no-color", "--no-ext-diff", base, head],
        repo,
    )
    return out


def get_changed_files(repo: Path, base: str, head: str) -> list[str]:
    """Get list of files changed between two commits."""
    out = git_ok(["diff", "--name-only", base, head], repo)
    if not out:
        return []
    return [f for f in out.strip().splitlines() if f.strip()]


def has_binary_files(repo: Path, base: str, head: str) -> bool:
    """Check if any changed files are binary."""
    out = git_ok(["diff", "--numstat", base, head], repo)
    if not out:
        return False
    for line in out.strip().splitlines():
        if line.startswith("-\t-\t"):
            return True
    return False


# ── Repo discovery ────────────────────────────────────────────────────────────


def find_repos(repo_dirs: list[str]) -> list[Path]:
    """Find git repositories under the given directories.

    Handles:
    - Direct repos: DIR is itself a git repo
    - Nested repos: ~/git-repos/github.com/owner/repo

    Deduplicates by both resolved filesystem path and remote URL.
    Filters out repos with no commits on their default branch.
    """
    candidates = []
    for d in repo_dirs:
        root = Path(d).expanduser()
        if not root.exists():
            continue

        if (root / ".git").exists():
            candidates.append(root)
            continue

        # Walk up to 4 levels deep (host/owner/repo or just owner/repo)
        for depth in range(1, 5):
            pattern = "/".join(["*"] * depth)
            for candidate in root.glob(pattern):
                if (candidate / ".git").exists():
                    candidates.append(candidate)

    # Deduplicate by resolved path first
    seen_paths: set[Path] = set()
    path_deduped = []
    for r in candidates:
        resolved = r.resolve()
        if resolved not in seen_paths:
            seen_paths.add(resolved)
            path_deduped.append(r)

    # Deduplicate by remote URL (keep the first occurrence)
    seen_urls: set[str] = set()
    deduped = []
    n_url_dupes = 0
    for r in path_deduped:
        url = get_remote_url(r)
        if url:
            if url in seen_urls:
                n_url_dupes += 1
                continue
            seen_urls.add(url)
        deduped.append(r)

    if n_url_dupes:
        print(
            f"  Removed {n_url_dupes} duplicate(s) (same remote URL).",
            file=sys.stderr,
        )

    # Filter out repos with too few commits to produce useful training data
    min_repo_commits = 10
    filtered = []
    n_small = 0
    for r in deduped:
        branch = get_default_branch(r)
        commits = get_first_parent_commits(r, branch, max_count=min_repo_commits)
        if len(commits) >= min_repo_commits:
            filtered.append(r)
        else:
            n_small += 1

    if n_small:
        print(
            f"  Filtered {n_small} repo(s) with <{min_repo_commits} commits.",
            file=sys.stderr,
        )

    return sorted(filtered, key=lambda p: str(p))


# ── Main harvest logic ────────────────────────────────────────────────────────


def get_all_commits(repo: Path, max_count: int = 5000) -> list[str]:
    """Get all commit SHAs in the repo across all branches, topo-sorted."""
    out = git_ok(
        ["rev-list", "--all", "--topo-order", f"--max-count={max_count}"],
        repo,
    )
    if not out:
        return []
    return [line.strip() for line in out.strip().splitlines() if line.strip()]


def get_parent_chain(repo: Path, start: str, length: int) -> list[str] | None:
    """Get a first-parent chain of `length` commits starting from `start` (inclusive).

    Returns None if the chain is shorter than requested.
    """
    out = git_ok(
        ["rev-list", "--first-parent", f"--max-count={length}", start],
        repo,
    )
    if not out:
        return None
    chain = [line.strip() for line in out.strip().splitlines() if line.strip()]
    if len(chain) < length:
        return None
    return chain


def harvest_one(
    repo: Path,
    rng: random.Random,
    min_commits: int,
    max_commits: int,
    max_files: int,
    max_diff_chars: int,
    all_branches: bool = False,
) -> dict | None:
    """Try to generate one training example from a repo.

    All operations are read-only.
    Returns None on failure.
    """
    if all_branches:
        # Pick a random commit from anywhere in the repo
        all_shas = get_all_commits(repo)
        if not all_shas:
            return None

        n = rng.randint(min_commits, max_commits)
        # Need n+1 commits in the chain (n to unwind + 1 for the base)
        start = rng.choice(all_shas)
        chain = get_parent_chain(repo, start, n + 1)
        if chain is None:
            return None

        head_sha = chain[0]
        base_sha = chain[-1]
        # The commits being unwound are chain[0..n-1], base is chain[n]
        commit_shas = chain[:-1]
        branch = None  # unknown / multi-branch
    else:
        branch = get_default_branch(repo)
        commits = get_first_parent_commits(repo, branch)
        if len(commits) < min_commits + 1:
            return None

        n = rng.randint(min_commits, min(max_commits, len(commits) - 1))
        max_head_idx = len(commits) - n - 1
        if max_head_idx < 0:
            return None
        head_idx = rng.randint(0, max_head_idx)
        head_sha = commits[head_idx]
        base_sha = commits[head_idx + n]
        commit_shas = commits[head_idx : head_idx + n]

    # Get changed files
    changed_files = get_changed_files(repo, base_sha, head_sha)
    if not changed_files:
        return None
    if len(changed_files) > max_files:
        return None

    # Skip if binary files are involved
    if has_binary_files(repo, base_sha, head_sha):
        return None

    # Get the diff
    diff = get_diff(repo, base_sha, head_sha)
    if not diff:
        return None
    if len(diff) > max_diff_chars:
        return None

    # Get info about each original commit in the range
    original_commits = []
    for sha in commit_shas:
        info = get_commit_info(repo, sha)
        if info:
            original_commits.append(info)

    # Skip if all commits were merges
    if not original_commits:
        return None

    # Detect languages
    languages = detect_languages(changed_files)

    # Mine .gitignore patterns from the base commit
    existing_gitignore = mine_gitignore(repo, base_sha)

    # Get repo URL
    repo_url = get_remote_url(repo) or str(repo)

    result = {
        "repo_url": repo_url,
        "branch": branch,
        "base_commit": base_sha,
        "head_commit": head_sha,
        "n_commits": n,
        "diff": diff,
        "changed_files": changed_files,
        "original_commits": original_commits,
        "existing_gitignore": existing_gitignore,
        "languages": languages,
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate git-smart-commit training inputs from local repos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-dirs",
        nargs="+",
        required=True,
        help="Directories to scan for git repos (supports shell globs)",
    )
    parser.add_argument(
        "--output",
        default="training-inputs.jsonl",
        help="Output JSONL file (default: training-inputs.jsonl)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of examples to generate (default: 1000)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=20,
        help="Skip commits touching more than this many files (default: 20)",
    )
    parser.add_argument(
        "--max-diff-chars",
        type=int,
        default=200000,
        help="Skip diffs larger than this (default: 200000)",
    )
    parser.add_argument(
        "--min-commits",
        type=int,
        default=1,
        help="Minimum commits to unwind (default: 1)",
    )
    parser.add_argument(
        "--max-commits",
        type=int,
        default=8,
        help="Maximum commits to unwind (default: 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--max-per-repo",
        type=int,
        default=None,
        help="Max examples per repo (default: unlimited). "
        "Useful to prevent large repos from dominating the dataset.",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Run 'git fetch --all' on each repo before harvesting",
    )
    parser.add_argument(
        "--all-branches",
        action="store_true",
        help="Sample from all local branches, not just the default branch",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just discover repos and print stats, don't generate",
    )

    args = parser.parse_args()
    rng = random.Random(args.seed)

    # Expand globs in repo-dirs
    expanded_dirs = []
    for pattern in args.repo_dirs:
        matches = glob.glob(os.path.expanduser(pattern), recursive=True)
        if matches:
            expanded_dirs.extend(matches)
        else:
            expanded_dirs.append(os.path.expanduser(pattern))

    print(
        f"Scanning for repos in {len(expanded_dirs)} director(ies)...",
        file=sys.stderr,
    )
    repos = find_repos(expanded_dirs)
    print(f"Found {len(repos)} repositories.", file=sys.stderr)

    if not repos:
        print("No repositories found. Check --repo-dirs paths.", file=sys.stderr)
        sys.exit(1)

    # Fetch if requested
    if args.fetch:
        print("Fetching all remotes...", file=sys.stderr)
        for i, repo in enumerate(repos):
            url = get_remote_url(repo) or str(repo)
            print(f"  [{i + 1}/{len(repos)}] {url}", file=sys.stderr, end="")
            if fetch_all(repo):
                print(" ✓", file=sys.stderr)
            else:
                print(" ✗ (fetch failed, using existing)", file=sys.stderr)

    if args.dry_run:
        if args.all_branches:
            print(
                f"\n{'Repo':<55} {'Commits':>8}  {'Default':<12}  Languages",
                file=sys.stderr,
            )
            print("─" * 110, file=sys.stderr)
        else:
            print(
                f"\n{'Repo':<60} {'Commits':>8}  {'Branch':<12}  Languages",
                file=sys.stderr,
            )
            print("─" * 110, file=sys.stderr)

        for r in repos:
            url = get_remote_url(r) or str(r)
            branch = get_default_branch(r) or "?"

            if args.all_branches:
                all_shas = get_all_commits(r)
                total_commits = len(all_shas)
            else:
                commits = get_first_parent_commits(r, branch, max_count=500)
                total_commits = len(commits)

            # Language detection from default branch
            default_commits = get_first_parent_commits(r, branch, max_count=500)
            if len(default_commits) > 1:
                recent_files = get_changed_files(
                    r,
                    default_commits[-1],
                    default_commits[0],
                )
            else:
                recent_files = []
            langs = detect_languages(recent_files)

            url_display = url[:58] if len(url) > 58 else url
            print(
                f"  {url_display:<60} {total_commits:>6}  "
                f"{branch:<12}  {', '.join(langs) or '-'}",
                file=sys.stderr,
            )
        sys.exit(0)

    # Generate examples, sampling randomly across repos for diversity
    generated = 0
    attempts = 0
    max_attempts = args.count * 20
    repo_counts: dict[str, int] = {}  # repo path -> examples generated
    max_per_repo = args.max_per_repo
    available_repos = list(repos)

    with open(args.output, "w") as f:
        while generated < args.count and attempts < max_attempts and available_repos:
            attempts += 1
            repo = rng.choice(available_repos)
            repo_key = str(repo)

            example = harvest_one(
                repo,
                rng,
                min_commits=args.min_commits,
                max_commits=args.max_commits,
                max_files=args.max_files,
                max_diff_chars=args.max_diff_chars,
                all_branches=args.all_branches,
            )

            if example is None:
                continue

            f.write(json.dumps(example) + "\n")
            generated += 1
            repo_counts[repo_key] = repo_counts.get(repo_key, 0) + 1

            # Remove repo from pool if it hit the cap
            if max_per_repo and repo_counts[repo_key] >= max_per_repo:
                available_repos = [r for r in available_repos if str(r) != repo_key]
                if not available_repos:
                    print(
                        f"  All repos exhausted (hit --max-per-repo={max_per_repo}).",
                        file=sys.stderr,
                    )

            if generated % 50 == 0:
                print(
                    f"  Generated {generated}/{args.count} "
                    f"({attempts} attempts, {len(available_repos)} repos remaining)",
                    file=sys.stderr,
                )

    print(
        f"\nDone. Generated {generated} examples in {attempts} attempts.",
        file=sys.stderr,
    )
    print(f"Output: {args.output}", file=sys.stderr)

    # Print stats
    if generated > 0:
        sizes = []
        lang_counts: dict[str, int] = {}
        commit_counts: list[int] = []
        repo_url_counts: dict[str, int] = {}
        with open(args.output) as f:
            for line in f:
                data = json.loads(line)
                sizes.append(len(line))
                commit_counts.append(data["n_commits"])
                for lang in data.get("languages", []):
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1
                url = data.get("repo_url", "?")
                repo_url_counts[url] = repo_url_counts.get(url, 0) + 1

        print("\nStats:", file=sys.stderr)
        print(f"  Examples: {generated}", file=sys.stderr)
        print(f"  Unique repos used: {len(repo_url_counts)}", file=sys.stderr)
        print(f"  Avg line size: {sum(sizes) // len(sizes):,} chars", file=sys.stderr)
        print(
            f"  Commits unwound: min={min(commit_counts)} "
            f"max={max(commit_counts)} "
            f"avg={sum(commit_counts) / len(commit_counts):.1f}",
            file=sys.stderr,
        )
        print(f"  Languages: {json.dumps(lang_counts, indent=4)}", file=sys.stderr)

        # Show top repos by example count
        top_repos = sorted(
            repo_url_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:15]
        print("  Top repos:", file=sys.stderr)
        for url, count in top_repos:
            url_display = url[:55] if len(url) > 55 else url
            print(f"    {url_display:<58} {count:>4}", file=sys.stderr)


if __name__ == "__main__":
    main()
