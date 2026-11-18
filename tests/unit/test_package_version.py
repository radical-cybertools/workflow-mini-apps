"""RunStarted versions: '<version>@<commit>' for git and editable installs."""

import json

from wfMiniAPI.telemetry import capability
from wfMiniAPI.telemetry.capability import _git_head, _package

SHA = "0123456789abcdef0123456789abcdef01234567"


def repo(tmp_path, head="ref: refs/heads/main", loose=True):
    git = tmp_path / "repo" / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text(head + "\n")
    if loose:
        (git / "refs" / "heads" / "main").write_text(SHA + "\n")
    else:
        (git / "packed-refs").write_text(f"# pack-refs\n{SHA} refs/heads/main\n")
    return git.parent


def test_loose_ref(tmp_path):
    assert _git_head(repo(tmp_path)) == SHA


def test_packed_ref(tmp_path):
    assert _git_head(repo(tmp_path, loose=False)) == SHA


def test_detached_head(tmp_path):
    assert _git_head(repo(tmp_path, head=SHA)) == SHA


def test_worktree(tmp_path):
    main = repo(tmp_path)
    wt_git = main / ".git" / "worktrees" / "wt"
    wt_git.mkdir(parents=True)
    (wt_git / "HEAD").write_text("ref: refs/heads/main\n")
    (wt_git / "commondir").write_text("../..\n")
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {wt_git}\n")
    assert _git_head(wt) == SHA


def test_not_a_checkout(tmp_path):
    assert _git_head(tmp_path) == ""


class FakeDist:
    version = "0.6.0"

    def __init__(self, direct_url):
        self._direct_url = direct_url

    def read_text(self, name):
        return json.dumps(self._direct_url) if name == "direct_url.json" else None


def test_package_editable(tmp_path, monkeypatch):
    url = {"url": f"file://{repo(tmp_path)}", "dir_info": {"editable": True}}
    monkeypatch.setattr(capability.metadata, "distribution", lambda _: FakeDist(url))
    assert _package("x") == "0.6.0@0123456"


def test_package_vcs(monkeypatch):
    url = {"url": "https://x", "vcs_info": {"vcs": "git", "commit_id": SHA}}
    monkeypatch.setattr(capability.metadata, "distribution", lambda _: FakeDist(url))
    assert _package("x") == "0.6.0@0123456"


def test_package_plain(monkeypatch):
    monkeypatch.setattr(capability.metadata, "distribution", lambda _: FakeDist({}))
    assert _package("x") == "0.6.0"
