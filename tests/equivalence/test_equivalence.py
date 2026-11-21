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
