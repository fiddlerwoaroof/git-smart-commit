"""Unit tests for pure functions in git-smart-commit.

These tests cover the logic that has no LLM dependency and can be verified
against known inputs and outputs.  The functions are extracted from the script
source using ast+compile so we don't need all of its heavyweight dependencies
(httpx, textual, steropes).
"""

import re
import sys
import textwrap
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Extract and compile just the pure-function section of the script.
# We snip out the imports that would require httpx/pydantic/textual and
# only keep the fragments needed by the functions under test.
# ---------------------------------------------------------------------------

_SCRIPT = Path(__file__).parent.parent / "git-smart-commit"


def _extract_function(src: str, name: str) -> str:
    """Extract a top-level function definition from source text."""
    pattern = rf"^(def {re.escape(name)}\b.*?)(?=\ndef |\nclass |\Z)"
    m = re.search(pattern, src, re.DOTALL | re.MULTILINE)
    if not m:
        raise RuntimeError(f"Function {name!r} not found in script")
    return m.group(1)


def _load_pure_functions() -> types.ModuleType:
    """Compile only the pure helper functions from the script."""
    src = _SCRIPT.read_text()

    fn_names = ["split_hunks", "_parse_hunk_oldrange", "_is_binary_diff"]
    stubs = "\n\n".join(_extract_function(src, n) for n in fn_names)

    minimal_src = "\n".join(
        [
            "import re",
            "import textwrap",
            "from pathlib import Path",
            "",
            stubs,
        ]
    )

    mod = types.ModuleType("gsc_pure")
    mod.__file__ = str(_SCRIPT)
    exec(compile(minimal_src, "<gsc_pure>", "exec"), mod.__dict__)
    return mod


_gsc = _load_pure_functions()

split_hunks = _gsc.split_hunks
_parse_hunk_oldrange = _gsc._parse_hunk_oldrange
_is_binary_diff = _gsc._is_binary_diff


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIMPLE_DIFF = textwrap.dedent("""\
    diff --git a/foo.py b/foo.py
    index abc..def 100644
    --- a/foo.py
    +++ b/foo.py
    @@ -1,3 +1,4 @@
     line1
    +line2_new
     line3
     line4
    @@ -10,3 +11,2 @@
     contextA
    -removed
     contextB
""")

MULTI_FILE_DIFF = textwrap.dedent("""\
    diff --git a/a.py b/a.py
    index 000..111 100644
    --- a/a.py
    +++ b/a.py
    @@ -1,2 +1,3 @@
     old
    +new
     end
    diff --git a/b.py b/b.py
    index 222..333 100644
    --- a/b.py
    +++ b/b.py
    @@ -5,2 +5,1 @@
     keep
    -drop
""")


# ---------------------------------------------------------------------------
# split_hunks
# ---------------------------------------------------------------------------


class TestSplitHunks:
    def test_single_hunk_returns_one_element(self):
        diff = textwrap.dedent("""\
            diff --git a/f.py b/f.py
            --- a/f.py
            +++ b/f.py
            @@ -1,1 +1,2 @@
             existing
            +added
        """)
        result = split_hunks(diff)
        assert len(result) == 1
        assert "@@ -1,1 +1,2 @@" in result[0]

    def test_two_hunks_in_one_file(self):
        result = split_hunks(SIMPLE_DIFF)
        assert len(result) == 2
        assert "@@ -1,3 +1,4 @@" in result[0]
        assert "@@ -10,3 +11,2 @@" in result[1]

    def test_multi_file_diff_splits_correctly(self):
        result = split_hunks(MULTI_FILE_DIFF)
        assert len(result) == 2
        assert "a.py" in result[0]
        assert "b.py" in result[1]

    def test_each_hunk_keeps_file_header(self):
        result = split_hunks(SIMPLE_DIFF)
        for hunk in result:
            assert "diff --git" in hunk
            assert "--- " in hunk
            assert "+++ " in hunk

    def test_empty_diff_returns_one_element(self):
        result = split_hunks("")
        assert len(result) == 1
        assert result[0] == ""

    def test_no_at_sign_treated_as_single_hunk(self):
        raw = "diff --git a/x b/x\n--- a/x\n+++ b/x\n plain line\n"
        result = split_hunks(raw)
        assert len(result) == 1

    def test_hunk_content_preserved(self):
        result = split_hunks(SIMPLE_DIFF)
        assert "+line2_new" in result[0]
        assert "-removed" in result[1]


# ---------------------------------------------------------------------------
# _parse_hunk_oldrange
# ---------------------------------------------------------------------------


class TestParseHunkOldrange:
    def test_basic_range(self):
        patch = "diff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -10,5 +10,6 @@\n context\n"
        assert _parse_hunk_oldrange(patch) == (10, 5)

    def test_single_line_no_count(self):
        # @@ -7 +7,2 @@ means start=7, count=1 (omitted means 1)
        patch = "@@ -7 +7,2 @@\n line\n"
        assert _parse_hunk_oldrange(patch) == (7, 1)

    def test_count_zero(self):
        # @@ -5,0 +5,1 @@ is a pure insertion
        patch = "@@ -5,0 +5,1 @@\n+new line\n"
        assert _parse_hunk_oldrange(patch) == (5, 0)

    def test_no_at_signs_returns_none(self):
        assert _parse_hunk_oldrange("no hunk here") is None

    def test_malformed_header_returns_none(self):
        assert _parse_hunk_oldrange("@@ bad format @@\n") is None

    def test_uses_first_at_sign_only(self):
        patch = "@@ -1,3 +1,4 @@\n context\n@@ -20,2 +21,2 @@\n other\n"
        assert _parse_hunk_oldrange(patch) == (1, 3)


# ---------------------------------------------------------------------------
# _is_binary_diff
# ---------------------------------------------------------------------------


class TestIsBinaryDiff:
    def test_binary_files_line(self):
        assert _is_binary_diff("Binary files a/img.png and b/img.png differ")

    def test_git_binary_patch(self):
        assert _is_binary_diff("GIT binary patch\nliteral 1234\n...")

    def test_normal_diff_is_not_binary(self):
        assert not _is_binary_diff(SIMPLE_DIFF)

    def test_empty_is_not_binary(self):
        assert not _is_binary_diff("")

    def test_partial_binary_line_not_matched(self):
        # "Binary files" present but doesn't end with "differ"
        assert not _is_binary_diff("Binary files a/x and b/x changed")
