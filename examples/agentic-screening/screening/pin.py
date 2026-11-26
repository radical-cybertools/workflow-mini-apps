"""The AUTONMI commit this example is tested on, and the check of the commit a run imports.

The framework is installed separately (the README's recipe), so nothing else ties the example to
the API it was written for. `AUTONMI_COMMIT` is that commit; the README names the same one, and a
test keeps the two equal. A run warns on stderr when the framework it imports is at another
commit, at an unknown one, or has changed files, and records both commits in
`<run-dir>/autonmi.json`. The check is advisory: it never stops a run.

Nothing here imports the framework, so its tests run without it.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

# AUTONMI's main with its public API (`autonmi.mcp`, `ModelSettings`), 26 Nov 2026
AUTONMI_COMMIT = "cdda559fcd78dbf42671305c3fdb0b6fa97d451b"

_HEX = re.compile(r"[0-9a-f]{7,40}")


def commit_of(package: Path, name: str | None = None) -> dict:
    """The commit of the package in directory `package`, and where it was read (`from`):

    - `git`: the package's own clone (an editable install), and whether the files under the package
      differ from that commit (`dirty`);
    - `.git-commit`: a file beside the package, or above its `src/`, for a copy made without git
      (`git archive <commit> | tar -x -C DIR; echo <commit> > DIR/.git-commit`);
    - `direct_url.json`: the installer's record of a package installed from git (`name`'s
      distribution).

    `commit` is None when none of them says; `dirty` is None when it cannot be known. A file that
    cannot be read or parsed counts as absent."""
    name = name or package.name

    def git(*args: str) -> str:
        cmd = ["git", "-C", str(package), *args]
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()

    try:
        git("ls-files", "--error-unmatch", "__init__.py")  # its own clone, not a repository above
        changed = git("status", "--porcelain", "--untracked-files=no", "--", ".")
        return {"commit": git("rev-parse", "HEAD").lower(), "dirty": bool(changed), "from": "git"}
    except (OSError, subprocess.SubprocessError):
        pass
    for root in (package.parent, package.parent.parent):  # beside the package, or above `src/`
        try:
            words = (root / ".git-commit").read_text().split()
        except (OSError, ValueError):  # absent, unreadable, or not text
            continue
        commit = words[0].lower() if words else ""
        if _HEX.fullmatch(commit):
            return {"commit": commit, "dirty": None, "from": ".git-commit"}
    try:
        for dist in metadata.packages_distributions().get(name, []):
            record = json.loads(metadata.distribution(dist).read_text("direct_url.json") or "{}")
            commit = str((record.get("vcs_info") or {}).get("commit_id") or "").lower()
            if _HEX.fullmatch(commit):
                return {"commit": commit, "dirty": None, "from": "direct_url.json"}
    except (OSError, ValueError, AttributeError, ImportError):  # a malformed or missing record
        pass
    return {"commit": None, "dirty": None, "from": None}


def check(got: dict, where: Path | str | None = None, pin: str = AUTONMI_COMMIT) -> dict:
    """`got` (from `commit_of`) against the pin, for the run's record: `pin`, and `matches_pin`,
    true when the commit is the pinned one (a short commit matches its full one) and its files are
    not known to differ. One warning on stderr when it does not match; `where`, the package's
    directory, is named in the warning only, as run records name no local path."""
    commit = got.get("commit") or ""
    same = len(commit) >= 7 and pin.startswith(commit)
    out = {**got, "pin": pin, "matches_pin": same and got.get("dirty") is not True}
    if not out["matches_pin"]:
        if same:
            what = f"at the pinned {pin[:12]}, but with changed files"
        else:
            what = f"at {commit[:12] or 'an unknown commit'}, not at the pinned {pin[:12]}"
        print(
            f"[screening] warning: the autonmi package imported{f' ({where})' if where else ''} is "
            f"{what} (AUTONMI_COMMIT in screening/pin.py)",
            file=sys.stderr,
            flush=True,
        )
    return out


def autonmi_pin() -> dict:
    """The check of the autonmi package this process imports."""
    import autonmi

    package = Path(autonmi.__file__).resolve().parent
    return check(commit_of(package, "autonmi"), where=package)
