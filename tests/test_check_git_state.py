"""Tests for GitAnalyzer.check_git_state — rebase/merge/cherry-pick detection.

Regression test for a stale-REBASE_HEAD false positive:

    .git/REBASE_HEAD is a ref git writes during rebase to record the commit
    being applied. It can be left behind after a successful or aborted rebase,
    and is not a reliable indicator that a rebase is currently in progress.
    The authoritative signals are the ``rebase-merge/`` or ``rebase-apply/``
    directories — if neither exists, there is no active rebase.
"""

import importlib.util
import importlib.machinery
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parent.parent / "git-smart-commit"


def _load_script():
    """Import git-smart-commit as a module (it has no .py extension)."""
    loader = importlib.machinery.SourceFileLoader("gsc_script", str(_SCRIPT))
    spec = importlib.util.spec_from_loader("gsc_script", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gsc():
    return _load_script()


@pytest.fixture
def clean_repo(tmp_path):
    """A fresh git repo with a single commit so HEAD points to a branch."""
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    # Minimal identity so commit succeeds even in CI
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "t@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test"],
        check=True,
    )
    (tmp_path / "README").write_text("hi\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "README"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-q", "-m", "init"],
        check=True,
    )
    return tmp_path


class TestCheckGitState:
    def test_clean_repo_passes(self, gsc, clean_repo):
        analyzer = gsc.GitAnalyzer(clean_repo, client=None)
        analyzer.check_git_state()  # should not raise or exit

    def test_stale_rebase_head_alone_is_ignored(self, gsc, clean_repo):
        """A leftover REBASE_HEAD file without rebase-merge/ or rebase-apply/
        must NOT be reported as an in-progress rebase.

        REBASE_HEAD can linger after git rebase completes or aborts in edge
        cases (older git, ungraceful termination, rerere). The real signal
        is the presence of the rebase-merge or rebase-apply directory.
        """
        (clean_repo / ".git" / "REBASE_HEAD").write_text("deadbeef\n")
        analyzer = gsc.GitAnalyzer(clean_repo, client=None)
        analyzer.check_git_state()  # should not raise or exit

    def test_active_rebase_merge_detected(self, gsc, clean_repo):
        """A real rebase-in-progress (rebase-merge/ dir) must still be caught."""
        (clean_repo / ".git" / "rebase-merge").mkdir()
        analyzer = gsc.GitAnalyzer(clean_repo, client=None)
        with pytest.raises(SystemExit):
            analyzer.check_git_state()

    def test_active_rebase_apply_detected(self, gsc, clean_repo):
        """A real am-based rebase (rebase-apply/ dir) must still be caught."""
        (clean_repo / ".git" / "rebase-apply").mkdir()
        analyzer = gsc.GitAnalyzer(clean_repo, client=None)
        with pytest.raises(SystemExit):
            analyzer.check_git_state()

    def test_merge_head_detected(self, gsc, clean_repo):
        (clean_repo / ".git" / "MERGE_HEAD").write_text("deadbeef\n")
        analyzer = gsc.GitAnalyzer(clean_repo, client=None)
        with pytest.raises(SystemExit):
            analyzer.check_git_state()

    def test_cherry_pick_head_detected(self, gsc, clean_repo):
        (clean_repo / ".git" / "CHERRY_PICK_HEAD").write_text("deadbeef\n")
        analyzer = gsc.GitAnalyzer(clean_repo, client=None)
        with pytest.raises(SystemExit):
            analyzer.check_git_state()
