"""Equivalence checks: in-scope workflows behave exactly like their goldens.

See tests/equivalence/README.md.  The DeepDriveSim case needs the 0c4f0c2
export in .equiv-work/ (or WFMINI_EQUIV_DDSIM) and is skipped without it.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EQUIV = Path(__file__).resolve().parent
sys.path.insert(0, str(EQUIV))

import compare  # noqa: E402
import harness  # noqa: E402

CASES = [(name, mode) for name, wf in harness.WORKFLOWS.items() for mode in wf.modes]


def _static_check(*args):
    return subprocess.run(
        [sys.executable, str(EQUIV / "static_check.py"), "check", *args],
        capture_output=True,
        text=True,
    )


def test_static_check():
    proc = _static_check()
    assert proc.returncode == 0, proc.stdout


def test_static_check_detects_core_edit(tmp_path):
    shutil.copytree(harness.REPO / "examples", tmp_path / "examples", symlinks=True)
    target = tmp_path / harness.WORKFLOWS["motif_4"].script
    text = target.read_text()
    target.write_text(text.replace("size=matmul_dim)", "size=matmul_dim + 1)", 1))
    proc = _static_check("--repo", str(tmp_path))
    assert proc.returncode == 1
    assert "core segment 'core' changed" in proc.stdout


@pytest.mark.parametrize("name,mode", CASES, ids=[f"{n}-{m}" for n, m in CASES])
def test_matches_golden(name, mode):
    if harness.WORKFLOWS[name].kind == "ddsim" and not harness.DDSIM_BASELINE.exists():
        pytest.skip(f"DeepDriveSim baseline not found at {harness.DDSIM_BASELINE}")
    golden = json.loads(harness.golden_path(name, mode).read_text())
    summary, facts = harness.run(name, mode, telemetry=True)
    diff = compare.first_difference(golden, summary)
    assert diff is None, diff
    if harness.WORKFLOWS[name].kind == "tutorial":
        return  # task scripts only: no workflow edge, no MiniApp
    # the MiniApp edge: one verified report, run events, every task span scoped
    (report,) = facts["reports"]
    assert report["telemetry_complete"], report["reasons"]
    assert report["events"].get("wfminiapi.RunStarted") == 1
    assert report["events"].get("wfminiapi.RunEnded") == 1
    scopes = set(facts["task_span_scopes"])
    if harness.WORKFLOWS[name].kind == "ddsim":
        # replica @flow.block ids: only the innermost id is recorded (UP-17)
        assert scopes and all(s and s.startswith("block.") for s in scopes), scopes
    else:
        assert scopes == {name}, scopes


MOTIFS = ["motif_2", "motif_3_simple", "motif_3_parallel", "motif_4", "motif_6"]


@pytest.mark.parametrize("name", MOTIFS)
def test_outputs_deleted_unless_kept(name, tmp_path):
    """With outputs.keep false, the script deletes every .h5 file the run wrote."""
    tiny = (harness.CONFIGS / harness.WORKFLOWS[name].config).read_text()
    assert "keep: true" in tiny
    config = tmp_path / "cfg.yaml"
    config.write_text(tiny.replace("keep: true", "keep: false"))
    summary = harness.run(name, "real", config=config)
    kept = harness.run(name, "real")
    assert summary["exit_code"] == kept["exit_code"]
    assert any(path.endswith(".h5") for path, _ in kept["files"])
    assert not [path for path, _ in summary["files"] if path.endswith(".h5")]


def test_harness_detects_behaviour_change(tmp_path):
    shutil.copytree(harness.REPO / "examples", tmp_path / "examples", symlinks=True)
    target = tmp_path / harness.WORKFLOWS["motif_4"].script
    text = target.read_text()
    target.write_text(text.replace("size=matmul_dim)", "size=matmul_dim + 1)", 1))
    env = dict(os.environ, WFMINI_EQUIV_TREE=str(tmp_path))
    code = (
        "import sys, json; sys.path.insert(0, %r); import harness; "
        "print(json.dumps(harness.run('motif_4', 'dry')))" % str(EQUIV)
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    golden = json.loads(harness.golden_path("motif_4", "dry").read_text())
    assert compare.first_difference(golden, json.loads(proc.stdout)) is not None


def test_baseline_patch_applies_to_a_copy(tmp_path):
    """The baseline fixes apply to a copy of the 55cd20f files; the source stays untouched."""
    source = tmp_path / "export"
    patched = [
        line[len("+++ b/") :].strip()
        for line in harness.BASELINE_PATCH.read_text().splitlines()
        if line.startswith("+++ b/")
    ]
    originals = {}
    for script in patched:
        proc = subprocess.run(
            ["git", "-C", str(harness.REPO), "show", f"55cd20f:{script}"],
            capture_output=True,
            text=True,
        )
        if proc.returncode:
            pytest.skip("55cd20f is not in this clone's history")
        (source / script).parent.mkdir(parents=True, exist_ok=True)
        (source / script).write_text(proc.stdout)
        originals[script] = proc.stdout
    target = tmp_path / "patched"
    assert harness._patched_copy(source, harness.BASELINE_PATCH, target) == target
    for script, text in originals.items():
        assert "filename_suffix=" in (target / script).read_text(), script
        assert (source / script).read_text() == text
    (target / "marker").write_text("")
    harness._patched_copy(source, harness.BASELINE_PATCH, target)  # same patch: reused
    assert (target / "marker").exists()
    (target / ".patch.sha").write_text("stale")
    harness._patched_copy(source, harness.BASELINE_PATCH, target)  # changed patch: rebuilt
    assert not (target / "marker").exists()
