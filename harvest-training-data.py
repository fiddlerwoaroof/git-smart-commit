#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
harvest-training-data: Generate git-smart-commit training inputs from local repos.

Walks existing local git repositories, picks random commits, rewinds N commits
along first-parent, captures the diff (including new files), optionally injects
contextual noise files, and writes ready-to-use JSONL training inputs.

SAFE: Never modifies your working tree. All operations are read-only git
commands (log, diff, diff-tree, show). No checkout, no stash, no worktree.

The expensive frontier model step is NOT done here — this just produces inputs.

Usage:
    harvest-training-data [--repo-dirs DIR ...] [--output FILE] [--count N]
                          [--max-files MAX] [--max-diff-chars MAX]
                          [--min-commits MIN] [--max-commits MAX]
                          [--seed SEED] [--noise-prob PROB]
                          [--fetch] [--dry-run]

Each output line is a JSON object:
{
    "repo_url": "github.com/owner/repo",
    "base_commit": "abc123...",
    "head_commit": "def456...",
    "n_commits": 3,
    "diff": "...",
    "noise_files": {"path": "content", ...},
    "changed_files": ["a.py", "b.py"],
    "original_commits": [{"sha": "...", "subject": "...", "files": [...]}],
    "existing_gitignore": ["*.pyc", "__pycache__/", ...],
    "languages": ["python", "rust"]
}
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
    ".kt": "java",       # kotlin, same ecosystem
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
    ".toml": "rust",      # could be generic, but often Cargo.toml
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


def detect_languages(changed_files: list[str]) -> list[str]:
    """Detect programming languages from file extensions."""
    langs = set()
    for f in changed_files:
        ext = Path(f).suffix.lower()
        if ext in EXTENSION_TO_LANGUAGE:
            langs.add(EXTENSION_TO_LANGUAGE[ext])
        # Check for known filenames too
        name = Path(f).name.lower()
        if name in ("cargo.toml", "cargo.lock"):
            langs.add("rust")
        elif name in ("pom.xml", "build.gradle", "build.gradle.kts"):
            langs.add("java")
        elif name in ("gemfile", "rakefile"):
            langs.add("ruby")
        elif name in ("setup.py", "pyproject.toml", "requirements.txt"):
            langs.add("python")
        elif name in ("project.clj", "deps.edn"):
            langs.add("clojure")
        elif name in ("package.json", "yarn.lock"):
            langs.add("javascript")
        elif name in ("go.mod", "go.sum"):
            langs.add("go")
        elif name == "flake.nix":
            langs.add("nix")
    return sorted(langs)


# ── Noise generation ──────────────────────────────────────────────────────────

# Language/ecosystem-specific noise patterns
NOISE_BY_LANGUAGE = {
    "python": [
        ("__pycache__/{stem}.cpython-311.pyc", b""),
        (".mypy_cache/3.11/{stem}.data.json", b'{{"mtime": 0}}'),
        (".pytest_cache/v/cache/lastfailed", b"{}"),
        (".ruff_cache/content", b""),
        ("{parent}/.ipynb_checkpoints/{stem}-checkpoint.ipynb", b'{{"cells":[]}}'),
        (".venv/lib/python3.11/site-packages/pip/__init__.py", b""),
    ],
    "java": [
        ("{parent}/target/classes/{stem}.class", b"\xca\xfe\xba\xbe"),
        (".idea/workspace.xml", b'<?xml version="1.0"?>\n<project version="4"/>\n'),
        (".idea/.gitignore", b""),
        (".idea/modules.xml", b'<?xml version="1.0"?>\n'),
        ("{parent}/{stem}.class", b"\xca\xfe\xba\xbe"),
        ("build/classes/java/main/App.class", b"\xca\xfe\xba\xbe"),
        (".gradle/caches/journal-1/file-access.bin", b""),
    ],
    "common-lisp": [
        ("{parent}/{stem}.fasl", b""),
        ("{parent}/{stem}.fas", b""),
        ("{parent}/{stem}.lx64fsl", b""),
    ],
    "clojure": [
        ("target/classes/clojure/core.class", b"\xca\xfe\xba\xbe"),
        (".cpcache/orchard.edn", b"{}"),
        (".nrepl-port", b"12345"),
        (".lsp/sqlite.db", b""),
    ],
    "ruby": [
        (".bundle/config", b'---\nBUNDLE_PATH: "vendor/bundle"\n'),
        ("vendor/bundle/ruby/3.2.0/gems/.keep", b""),
        ("coverage/.resultset.json", b'{{"coverage": {{}}}}'),
        ("tmp/pids/server.pid", b"12345"),
    ],
    "rust": [
        ("target/debug/.fingerprint/.keep", b""),
        ("target/debug/build/.keep", b""),
        ("target/debug/deps/.keep", b""),
        ("target/.rustc_info.json", b'{{"host": "x86_64"}}'),
    ],
    "javascript": [
        ("node_modules/.package-lock.json", b'{{"lockfileVersion": 3}}'),
        ("node_modules/.cache/.keep", b""),
        ("dist/bundle.js", b"// generated\nvar a=1;\n"),
        (".next/cache/.keep", b""),
        ("coverage/lcov.info", b""),
    ],
    "typescript": [
        ("node_modules/.package-lock.json", b'{{"lockfileVersion": 3}}'),
        ("dist/index.js", b'"use strict";\n'),
        (".tsbuildinfo", b'{{"program":{{}}}}'),
    ],
    "go": [
        ("vendor/modules.txt", b""),
    ],
    "haskell": [
        (".stack-work/dist/.keep", b""),
        ("dist-newstyle/cache/.keep", b""),
    ],
    "nix": [
        ("result", b""),  # symlink, but content doesn't matter for training
    ],
}

# Editor-specific noise
EDITOR_NOISE = {
    "vim": [
        (".{name}.swp", b""),
        (".{name}.swo", b""),
        ("{parent}/.netrwhist", b""),
        ("Session.vim", b""),
        ("{parent}/tags", b"!_TAG_FILE_FORMAT\t2\n"),
    ],
    "emacs": [
        ("{parent}/#{name}#", b""),
        ("{parent}/.#{name}", b""),
        ("{parent}/{name}~", b""),
        (".projectile", b""),
        ("TAGS", b"\x0c\n"),
    ],
    "intellij": [
        (".idea/workspace.xml", b'<?xml version="1.0"?>\n<project version="4"/>\n'),
        (".idea/misc.xml", b'<?xml version="1.0"?>\n'),
        (".idea/.gitignore", b""),
        (".idea/dictionaries/edwlan.xml", b"<component/>"),
        ("{parent}/{stem}.iml", b""),
        ("out/production/classes/.keep", b""),
    ],
    "vscode": [
        (".vscode/settings.json", b'{{"editor.fontSize": 14}}\n'),
        (".vscode/launch.json", b'{{"version": "0.2.0"}}\n'),
        (".vscode/.browse.db", b""),
    ],
}

# OS-level noise (always possible)
OS_NOISE = [
    (".DS_Store", b"\x00\x00\x00\x01Bud1"),
    ("Thumbs.db", b""),
    (".directory", b"[Desktop Entry]\nIcon=folder\n"),
    (".env", b"SECRET_KEY=changeme\nDATABASE_URL=postgres://localhost/dev\n"),
    (".env.local", b"API_KEY=test123\n"),
]

# Editors to simulate, weighted by usage
EDITORS = ["vim", "emacs", "intellij", "vscode"]


def generate_noise(
    changed_files: list[str],
    languages: list[str],
    rng: random.Random,
) -> dict[str, str]:
    """Generate contextual noise files based on languages and editors in use."""
    count = rng.randint(1, 5)
    noise = {}

    # Build candidate pool
    candidates = []

    # Add language-specific noise
    for lang in languages:
        candidates.extend(NOISE_BY_LANGUAGE.get(lang, []))

    # Add editor noise (pick 1-2 editors)
    n_editors = rng.randint(1, 2)
    editors = rng.sample(EDITORS, min(n_editors, len(EDITORS)))
    for editor in editors:
        candidates.extend(EDITOR_NOISE.get(editor, []))

    # Always include OS noise as candidates
    candidates.extend(OS_NOISE)

    if not candidates:
        return {}

    # Pick a reference file for template expansion
    ref_files = [f for f in changed_files if Path(f).suffix] or changed_files
    if not ref_files:
        return {}

    selected = rng.sample(candidates, min(count, len(candidates)))

    for template, content in selected:
        ref = rng.choice(ref_files)
        ref_path = Path(ref)

        parent = str(ref_path.parent) if str(ref_path.parent) != "." else "."

        # Expand templates
        try:
            path = template.format(
                name=ref_path.name,
                stem=ref_path.stem,
                parent=parent,
            )
        except (KeyError, IndexError):
            continue

        # Don't shadow real files
        if path in changed_files:
            continue

        noise[path] = content.decode("utf-8", errors="replace") if content else ""

    return noise


# ── Gitignore mining ──────────────────────────────────────────────────────────

def mine_gitignore(repo: Path, commit: str) -> list[str]:
    """Extract .gitignore patterns from the repo at a given commit.

    Checks the root .gitignore and any nested .gitignore files.
    """
    patterns = []

    # List all .gitignore files at this commit
    out = git_ok(
        ["ls-tree", "-r", "--name-only", commit],
        repo,
    )
    if not out:
        return []

    gitignore_files = [
        f for f in out.strip().splitlines()
        if Path(f).name == ".gitignore"
    ]

    for gi_path in gitignore_files:
        content = git_ok(["show", f"{commit}:{gi_path}"], repo)
        if not content:
            continue
        for line in content.splitlines():
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            # Prefix with directory if it's a nested .gitignore
            parent = str(Path(gi_path).parent)
            if parent != ".":
                patterns.append(f"{parent}/{line}")
            else:
                patterns.append(line)

    return patterns


# ── Git helpers ───────────────────────────────────────────────────────────────

def git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run a git command, return CompletedProcess."""
    return subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
    )


def git_ok(args: list[str], cwd: Path) -> str | None:
    """Run git command, return stdout on success, None on failure."""
    r = git(args, cwd)
    return r.stdout if r.returncode == 0 else None


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
            url = url[len(prefix):]
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
    """Get the default branch (main/master) for the repo."""
    # Try HEAD
    out = git_ok(["symbolic-ref", "--short", "HEAD"], repo)
    if out:
        return out.strip()

    # Try common names
    for branch in ["main", "master", "develop"]:
        r = git(["rev-parse", "--verify", branch], repo)
        if r.returncode == 0:
            return branch

    return None


def get_first_parent_commits(
    repo: Path, branch: str | None = None, max_count: int = 500,
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
    """Get subject and changed files for a commit."""
    out = git_ok(["show", "--no-patch", "--format=%s", sha], repo)
    if not out:
        return None
    subject = out.strip()

    # Skip merge commits
    if subject.startswith("Merge "):
        return None

    out = git_ok(["diff-tree", "--no-commit-id", "-r", "--name-only", sha], repo)
    files = [f for f in (out or "").strip().splitlines() if f.strip()]

    return {"sha": sha, "subject": subject, "files": files}


def get_diff(repo: Path, base: str, head: str) -> str | None:
    """Get the full diff between two commits, including new files.

    This is a pure read-only operation — no checkout needed.
    """
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
    - Bare repos (skipped)
    """
    repos = []
    for d in repo_dirs:
        root = Path(d).expanduser()
        if not root.exists():
            continue

        if (root / ".git").exists():
            repos.append(root)
            continue

        # Walk up to 4 levels deep (host/owner/repo or just owner/repo)
        for depth in range(1, 5):
            pattern = "/".join(["*"] * depth)
            for candidate in root.glob(pattern):
                if (candidate / ".git").exists():
                    repos.append(candidate)

    # Deduplicate by resolved path
    seen = set()
    deduped = []
    for r in repos:
        resolved = r.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(r)

    return sorted(deduped, key=lambda p: str(p))


# ── Main harvest logic ────────────────────────────────────────────────────────

def harvest_one(
    repo: Path,
    rng: random.Random,
    min_commits: int,
    max_commits: int,
    max_files: int,
    max_diff_chars: int,
    noise_prob: float,
) -> dict | None:
    """Try to generate one training example from a repo.

    All operations are read-only. Returns None on failure.
    """
    branch = get_default_branch(repo)
    commits = get_first_parent_commits(repo, branch)
    if len(commits) < min_commits + 1:
        return None

    # Pick how many commits to unwind
    n = rng.randint(min_commits, min(max_commits, len(commits) - 1))

    # Pick a random starting point (head), ensuring we have n commits behind it
    max_head_idx = len(commits) - n - 1
    if max_head_idx < 0:
        return None
    head_idx = rng.randint(0, max_head_idx)
    head_sha = commits[head_idx]
    base_sha = commits[head_idx + n]

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
    for i in range(head_idx, head_idx + n):
        info = get_commit_info(repo, commits[i])
        if info:
            original_commits.append(info)

    # Skip if all commits were merge commits (filtered out by get_commit_info)
    if not original_commits:
        return None

    # Detect languages
    languages = detect_languages(changed_files)

    # Mine .gitignore patterns from the base commit
    existing_gitignore = mine_gitignore(repo, base_sha)

    # Generate noise
    noise = {}
    if rng.random() < noise_prob:
        noise = generate_noise(changed_files, languages, rng)

    # Get repo URL
    repo_url = get_remote_url(repo) or str(repo)

    return {
        "repo_url": repo_url,
        "base_commit": base_sha,
        "head_commit": head_sha,
        "n_commits": n,
        "diff": diff,
        "noise_files": noise,
        "changed_files": changed_files,
        "original_commits": original_commits,
        "existing_gitignore": existing_gitignore,
        "languages": languages,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate git-smart-commit training inputs from local repos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-dirs", nargs="+", required=True,
        help="Directories to scan for git repos (supports shell globs)",
    )
    parser.add_argument(
        "--output", default="training-inputs.jsonl",
        help="Output JSONL file (default: training-inputs.jsonl)",
    )
    parser.add_argument(
        "--count", type=int, default=1000,
        help="Number of examples to generate (default: 1000)",
    )
    parser.add_argument(
        "--max-files", type=int, default=20,
        help="Skip commits touching more than this many files (default: 20)",
    )
    parser.add_argument(
        "--max-diff-chars", type=int, default=200000,
        help="Skip diffs larger than this (default: 200000)",
    )
    parser.add_argument(
        "--min-commits", type=int, default=1,
        help="Minimum commits to unwind (default: 1)",
    )
    parser.add_argument(
        "--max-commits", type=int, default=8,
        help="Maximum commits to unwind (default: 8)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--noise-prob", type=float, default=0.3,
        help="Probability of injecting noise files (default: 0.3)",
    )
    parser.add_argument(
        "--fetch", action="store_true",
        help="Run 'git fetch --all' on each repo before harvesting",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Just discover repos and print stats, don't generate",
    )

    args = parser.parse_args()
    rng = random.Random(args.seed)

    # Expand globs in repo-dirs
    expanded_dirs = []
    for pattern in args.repo_dirs:
        matches = glob.glob(os.path.expanduser(pattern))
        if matches:
            expanded_dirs.extend(matches)
        else:
            expanded_dirs.append(os.path.expanduser(pattern))

    print(f"Scanning for repos in {len(expanded_dirs)} director(ies)...", file=sys.stderr)
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
            print(f"  [{i+1}/{len(repos)}] {url}", file=sys.stderr, end="")
            if fetch_all(repo):
                print(" ✓", file=sys.stderr)
            else:
                print(" ✗ (fetch failed, using existing)", file=sys.stderr)

    if args.dry_run:
        print(
            f"\n{'Repo':<60} {'Commits':>8}  {'Branch':<12}  Languages",
            file=sys.stderr,
        )
        print("─" * 110, file=sys.stderr)
        for r in repos:
            url = get_remote_url(r) or str(r)
            branch = get_default_branch(r) or "?"
            commits = get_first_parent_commits(r, branch, max_count=500)

            # Quick language detection from recent files
            if len(commits) > 1:
                recent_files = get_changed_files(r, commits[-1], commits[0])
            else:
                recent_files = []
            langs = detect_languages(recent_files)

            url_display = url[:58] if len(url) > 58 else url
            print(
                f"  {url_display:<60} {len(commits):>6}  "
                f"{branch:<12}  {', '.join(langs) or '-'}",
                file=sys.stderr,
            )
        sys.exit(0)

    # Generate examples, sampling randomly across repos for diversity
    generated = 0
    attempts = 0
    max_attempts = args.count * 20

    with open(args.output, "w") as f:
        while generated < args.count and attempts < max_attempts:
            attempts += 1
            repo = rng.choice(repos)

            example = harvest_one(
                repo,
                rng,
                min_commits=args.min_commits,
                max_commits=args.max_commits,
                max_files=args.max_files,
                max_diff_chars=args.max_diff_chars,
                noise_prob=args.noise_prob,
            )

            if example is None:
                continue

            f.write(json.dumps(example) + "\n")
            generated += 1

            if generated % 50 == 0:
                print(
                    f"  Generated {generated}/{args.count} "
                    f"({attempts} attempts across {len(repos)} repos)",
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
        n_with_noise = 0
        lang_counts: dict[str, int] = {}
        commit_counts: list[int] = []
        with open(args.output) as f:
            for line in f:
                data = json.loads(line)
                sizes.append(len(line))
                if data.get("noise_files"):
                    n_with_noise += 1
                commit_counts.append(data["n_commits"])
                for lang in data.get("languages", []):
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1

        print(f"\nStats:", file=sys.stderr)
        print(f"  Examples: {generated}", file=sys.stderr)
        print(
            f"  With noise: {n_with_noise} ({100*n_with_noise//generated}%)",
            file=sys.stderr,
        )
        print(f"  Avg line size: {sum(sizes)//len(sizes):,} chars", file=sys.stderr)
        print(
            f"  Commits unwound: min={min(commit_counts)} "
            f"max={max(commit_counts)} "
            f"avg={sum(commit_counts)/len(commit_counts):.1f}",
            file=sys.stderr,
        )
        print(
            f"  Languages: {json.dumps(lang_counts, indent=4)}", file=sys.stderr,
        )


if __name__ == "__main__":
    main()
