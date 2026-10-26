"""Repository hygiene: nothing machine-specific, thesis-internal or from the old working copy in tracked files."""
import os
import re
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATTERNS = {
    "absolute home path": re.compile(r"/home/\w+"),
    "cluster node name": re.compile(r"bioinf-|euryale|stheno|yamata"),
    "old package name": re.compile(r"\blejepa\.|lejepa/"),
    "thesis-internal reference": re.compile(r"METHODOLOGY|\bsec \d+|§\d+|FIGURES\.md|thesis_writing_notes"),
    "German umlaut": re.compile(r"[äöüÄÖÜß]"),
}
ALLOW = {"tests/test_hygiene.py"}


def tracked_files():
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("not a git checkout")
    return [f for f in out.stdout.split() if f not in ALLOW and not f.endswith((".png", ".pdb", ".pt"))]


@pytest.mark.parametrize("name", list(PATTERNS))
def test_no_forbidden_text(name):
    hits = []
    for f in tracked_files():
        try:
            text = open(os.path.join(REPO, f), encoding="utf-8").read()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if PATTERNS[name].search(line):
                hits.append(f"{f}:{i}: {line.strip()[:100]}")
    assert not hits, f"{name}:\n" + "\n".join(hits[:20])
