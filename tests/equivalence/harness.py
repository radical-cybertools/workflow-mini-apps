"""Run an in-scope workflow under the equivalence shim and summarise what it did.

The summary is the unit of comparison between the baseline and any later
change (goldens).  It holds only behaviour that must not change:

* the task/block call graph (name, arguments, dependency edges);
* per task execution: the ordered kernel calls with their arguments;
* manager-side kernel calls;
* files written (path and size, real mode only);
* exit code and the failure point.

Logs, timestamps, pids and thread names are not part of it.
"""

from __future__ import annotations

import collections
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SHIM = HERE / "shim"
CONFIGS = HERE / "configs"
GOLDENS = HERE / "goldens"
WORK = REPO / ".equiv-work"
DDSIM_BASELINE = (WORK / "ddsim-0c4f0c2").resolve()
DDSIM_PATCH = HERE / "ddsim" / "run_workflow.patch"
# integrated DeepDriveSim = baseline export + DDSIM_PATCH (the edge change); override with
# WFMINI_EQUIV_DDSIM to check another tree (e.g. the DeepDriveSim worktree)
DDSIM_INTEGRATED = Path(os.environ.get("WFMINI_EQUIV_DDSIM", WORK / "ddsim-wfm")).resolve()
PYTHON = Path(os.environ.get("WFMINI_EQUIV_PYTHON", sys.executable))
TIMEOUT_S = int(os.environ.get("WFMINI_EQUIV_TIMEOUT", "900"))
# tree whose examples are run (default: this repository); lets a test run a modified copy
TREE = Path(os.environ.get("WFMINI_EQUIV_TREE", REPO)).resolve()
# optional directory put first on PYTHONPATH so a different wfMiniAPI is imported
LIB = os.environ.get("WFMINI_EQUIV_LIB")
# goldens are captured on the baseline: workflow-mini-apps 55cd20f + the KR-01 guard
BASELINE = WORK / "wfmini-55cd20f"
# "concurrent" (default): deterministic, compared with the goldens.  "dragon": the real
# Dragon backend under the `dragon` launcher; parallel, so baseline and integrated runs
# are compared with each other as multisets (run_equiv.py pair), not with goldens.
BACKEND = os.environ.get("WFMINI_EQUIV_BACKEND", "concurrent")
DRAGON = Path(os.environ.get("WFMINI_EQUIV_DRAGON", PYTHON.parent / "dragon"))


@dataclass(frozen=True)
class Workflow:
    name: str
    kind: str  # "example" | "ddsim" | "tutorial"
    script: str  # path relative to its root
    config: str | None = None  # tiny config in configs/
    args: tuple[str, ...] = ()
    modes: tuple[str, ...] = ("dry", "real")
    scripts: tuple[tuple[str, tuple[str, ...]], ...] = field(default=())


EXAMPLES = {
    "motif_2": "examples/motif_2/miniapp_pipeline.py",
    "motif_3_simple": "examples/motif_3/miniapp_simple.py",
    "motif_3_parallel": "examples/motif_3/miniapp_parallel.py",
    "motif_4": "examples/motif_4/miniapp_async.py",
    "motif_6": "examples/motif_6/miniapp_adaptive.py",
    "dragon_mpi": "examples/dragon-mpi/miniapp_rhapsody.py",
}

TUTORIAL_ARGS = {
    "simulation": (
        "--N_atoms",
        "64",
        "--grid_size",
        "8",
        "--n_steps",
        "2",
        "--neighbor_freq",
        "1",
        "--log_freq",
        "1",
        "--n_force",
        "2",
        "--n_fft",
        "1",
        "--n_int",
        "2",
        "--device",
        "cpu",
    ),
    "training": (
        "--num_epochs",
        "2",
        "--num_sample",
        "64",
        "--batch_size",
        "16",
        "--dense_dim_in",
        "32",
        "--dense_dim_out",
        "16",
        "--log_freq",
        "1",
        "--num_mult",
        "2",
        "--device",
        "cpu",
    ),
    "active_learn": (
        "--num_sample",
        "64",
        "--batch_size",
        "16",
        "--dense_dim_in",
        "32",
        "--dense_dim_out",
        "16",
        "--top_k",
        "2",
        "--num_mult",
        "2",
        "--device",
        "cpu",
    ),
}

WORKFLOWS: dict[str, Workflow] = {
    **{
        name: Workflow(
            name, "example", path, config=f"{name}.tiny.yaml", args=("--config", "cfg.yaml")
        )
        for name, path in EXAMPLES.items()
    },
    "ddsim": Workflow(
        "ddsim",
        "ddsim",
        "workflows/miniapps_workflow/run_workflow.py",
        config="ddsim.tiny.yaml",
        args=("--config_file", "cfg.yaml"),
        modes=("real",),
    ),
    "tutorial": Workflow(
        "tutorial", "tutorial", "examples/tutorial", scripts=tuple(TUTORIAL_ARGS.items())
    ),
}


def tutorial_dir(tree: Path = TREE) -> Path:
    for candidate in ("examples/tutorial", "examples/Tutorial/tasks"):
        if (tree / candidate).is_dir():
            return tree / candidate
    raise FileNotFoundError("Tutorial task scripts not found")


def example_path(workflow: Workflow, tree: Path = TREE) -> Path:
    return tree / workflow.script


def integrated_ddsim() -> Path:
    """The integrated DeepDriveSim tree, built from the baseline export and DDSIM_PATCH."""
    if "WFMINI_EQUIV_DDSIM" in os.environ or not DDSIM_BASELINE.exists():
        return DDSIM_INTEGRATED
    stamp = DDSIM_INTEGRATED / ".patch.sha"
    digest = __import__("hashlib").sha256(DDSIM_PATCH.read_bytes()).hexdigest()
    if not (stamp.exists() and stamp.read_text() == digest):
        shutil.rmtree(DDSIM_INTEGRATED, ignore_errors=True)
        shutil.copytree(DDSIM_BASELINE, DDSIM_INTEGRATED)
        subprocess.run(
            ["patch", "-p1", "-s", "-i", str(DDSIM_PATCH)], cwd=DDSIM_INTEGRATED, check=True
        )
        stamp.write_text(digest)
    return DDSIM_INTEGRATED


# ---------------------------------------------------------------------------- #
# running
def _env(
    trace_dir: Path, mode: str, extra_path: list[Path], lib: str | None = LIB
) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "WFMINI_EQUIV": "1",
            "WFMINI_EQUIV_KERNELS": mode,
            "WFMINI_EQUIV_OUT": str(trace_dir),
            "PYTHONPATH": os.pathsep.join(
                str(p) for p in [SHIM, *([lib] if lib else []), *extra_path]
            ),
            "PYTHONHASHSEED": "0",
            "WFMINI_EQUIV_BACKEND": BACKEND,
            "DDSIM_BACKEND": BACKEND,
        }
    )
    if BACKEND == "dragon":  # the launcher starts `python` from PATH
        env["PATH"] = os.pathsep.join([str(PYTHON.parent), env.get("PATH", "")])
    return env


def _launch(script: Path) -> list[str]:
    if BACKEND == "dragon":
        return [str(DRAGON), "-s", str(script)]
    return [str(PYTHON), str(script)]


def _run(cmd: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    """Run one workflow process; a run that outlives TIMEOUT_S gets returncode "timeout".

    A hung run is killed with every process it started (Dragon's runtime and workers),
    not just the launcher.  (The Dragon launcher does not start in a new session, so the
    tree is collected with psutil instead of a process group.)
    """
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        out, err = proc.communicate(timeout=TIMEOUT_S)
        return subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    except subprocess.TimeoutExpired:
        import psutil  # installed with rhapsody-py[telemetry]

        tree = psutil.Process(proc.pid).children(recursive=True)
        for p in [*tree, psutil.Process(proc.pid)]:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
        out, err = proc.communicate()
        if BACKEND == "dragon":
            subprocess.run([str(DRAGON.parent / "dragon-cleanup")], capture_output=True)
        return subprocess.CompletedProcess(cmd, "timeout", out, err)


def run(
    name: str,
    mode: str,
    keep: Path | None = None,
    baseline: bool = False,
    telemetry: bool = False,
):
    """Run one workflow in one mode and return its summary.

    ``baseline=True`` runs the pristine baseline tree and library (used to capture goldens).
    """
    tree = BASELINE if baseline else TREE
    lib = str(BASELINE / "wfMiniAPI" / "src") if baseline else LIB
    workflow = WORKFLOWS[name]
    WORK.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"{name}.{mode}.", dir=WORK))
    trace_dir = tmp / "_trace"
    run_dir = tmp / "run"
    run_dir.mkdir()
    try:
        if workflow.kind == "tutorial":
            procs = []
            for script, args in workflow.scripts:
                cmd = [
                    str(PYTHON),
                    str(tutorial_dir(tree) / f"{script}.py"),
                    "--experiment_dir",
                    str(run_dir),
                    "--task_index",
                    "0",
                    *args,
                ]
                procs.append(_run(cmd, run_dir, _env(trace_dir, mode, [], lib)))
            exit_code = next((p.returncode for p in procs if p.returncode), 0)
            stderr = "\n".join(p.stderr for p in procs)
        else:
            if workflow.config:
                shutil.copy(CONFIGS / workflow.config, run_dir / "cfg.yaml")
            if workflow.kind == "ddsim":
                root = DDSIM_BASELINE if baseline else integrated_ddsim()
                script, extra = root / workflow.script, [root]
            else:
                script, extra = example_path(workflow, tree), []
            proc = _run(
                [*_launch(script), *workflow.args],
                run_dir,
                _env(trace_dir, mode, extra, lib),
            )
            exit_code, stderr = proc.returncode, proc.stderr
            (tmp / "stdout.txt").write_text(proc.stdout)
            (tmp / "stderr.txt").write_text(proc.stderr)
        summary = summarise(workflow, mode, trace_dir, run_dir, exit_code, stderr)
        facts = telemetry_facts(run_dir) if telemetry else None
    finally:
        if keep is None:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            shutil.rmtree(keep, ignore_errors=True)
            shutil.move(str(tmp), keep)
    return (summary, facts) if telemetry else summary


# ---------------------------------------------------------------------------- #
# telemetry produced by the integrated (MiniApp) edge; not part of the goldens
def telemetry_facts(run_dir: Path) -> dict:
    """What the MiniApp edge recorded: report, run events, task-span scope tags."""
    reports = sorted(run_dir.rglob("*.wfminiapi.report.json"))
    checkpoints = sorted(run_dir.rglob("*.telemetry.jsonl"))
    facts = {"reports": [json.loads(p.read_text()) for p in reports], "task_span_scopes": []}
    for path in checkpoints:
        for line in path.read_text().splitlines():
            if '"section": "span"' not in line:
                continue
            rec = json.loads(line)
            if rec.get("name") == "task":
                attrs = rec.get("attributes") or {}
                facts["task_span_scopes"].append(attrs.get("asyncflow.workflow_id"))
    return facts


# ---------------------------------------------------------------------------- #
# summarising
_TMP_RE = re.compile(r"/[^\s'\"]*?/\.equiv-work/[^/\s'\"]+/run")


def _norm(value):
    """Replace per-run temporary paths so summaries are comparable."""
    if isinstance(value, str):
        for root in (DDSIM_BASELINE, DDSIM_INTEGRATED):
            value = value.replace(str(root), "<DDSIM>")
        value = value.replace(str(PYTHON), "<PYTHON>")
        return _TMP_RE.sub("<RUN>", value)
    if isinstance(value, list):
        return [_norm(v) for v in value]
    if isinstance(value, dict):
        return {k: _norm(v) for k, v in value.items()}
    return value


def _error_tail(stderr: str) -> str | None:
    lines = [ln for ln in stderr.strip().splitlines() if ln.strip()]
    for line in reversed(lines):
        if re.match(r"^[A-Za-z_.]*(Error|Exception|Interrupt)\b", line):
            return _norm(line.strip())
    return None


def _load_traces(trace_dir: Path) -> list[list[dict]]:
    traces = []
    for path in sorted(trace_dir.glob("trace.*.jsonl")):
        traces.append([json.loads(line) for line in path.read_text().splitlines() if line])
    return traces


def _kernel(rec: dict) -> dict:
    out = {"name": rec["name"], "args": rec["args"], "outcome": rec["outcome"]}
    if "error" in rec:
        out["error"] = rec["error"]
    return out


def _process_summary(records: list[dict]) -> dict:
    """Split one process trace into calls, task executions and manager kernels."""
    calls, executions, manager_kernels, violations = [], [], [], []
    open_by_thread: dict[str, list[dict]] = collections.defaultdict(list)
    argv = None
    for rec in sorted(records, key=lambda r: r["seq"]):
        kind = rec["kind"]
        if kind == "process":
            argv = rec["argv"]
        elif kind == "violation":
            violations.append(rec["what"])
        elif kind == "call":
            calls.append({k: rec[k] for k in ("comp", "name", "call", "args", "kwargs")})
        elif kind == "begin":
            execution = {"comp": rec["comp"], "name": rec["name"], "kernels": []}
            open_by_thread[rec["thread"]].append(execution)
            executions.append(execution)
        elif kind == "end":
            stack = open_by_thread[rec["thread"]]
            execution = stack.pop() if stack else {"name": rec["name"]}
            execution["outcome"] = rec["outcome"]
            if "error" in rec:
                execution["error"] = rec["error"]
            if rec.get("result") is not None:
                execution["result"] = rec["result"]
        elif kind == "kernel" and rec["depth"] == 0:  # top-level calls only
            stack = open_by_thread[rec["thread"]]
            (stack[-1]["kernels"] if stack else manager_kernels).append(_kernel(rec))
    return {
        "violations": violations,
        "argv": argv,
        "calls": calls,
        "executions": executions,
        "manager_kernels": manager_kernels,
    }


def _files(run_dir: Path) -> list[list]:
    out = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "cfg.yaml":
            rel = path.relative_to(run_dir).as_posix()
            if rel.startswith(("telemetry/", "telemetry-output/", "asyncflow.session.")):
                continue
            if rel.startswith(("runinfo/", "ddict_orc_")):  # Dragon runtime state, not output
                continue
            out.append([rel, path.stat().st_size])
    return out


def summarise(
    workflow: Workflow, mode: str, trace_dir: Path, run_dir: Path, exit_code: int, stderr: str
) -> dict:
    processes = [_process_summary(t) for t in _load_traces(trace_dir)]
    summary = {
        "workflow": workflow.name,
        "mode": mode,
        "exit_code": exit_code,
        "error_tail": _error_tail(stderr) if exit_code else None,
        "files": _files(run_dir) if mode == "real" else [],
    }
    if workflow.kind == "example" and BACKEND == "dragon":
        summary.update(_parallel_summary(workflow, processes, exit_code))
    elif workflow.kind == "example":
        (main,) = processes
        executions = main["executions"]
        if exit_code:
            # After the first failure, how many already-queued tasks AsyncFlow still
            # starts before it shuts down is timing-dependent: compare up to it.
            first = next(
                (i for i, e in enumerate(executions) if e.get("outcome") != "ok"),
                len(executions) - 1,
            )
            executions = executions[: first + 1]
        summary.update(
            calls=main["calls"], executions=executions, manager_kernels=main["manager_kernels"]
        )
    elif workflow.kind == "tutorial":
        summary["scripts"] = [
            {"script": Path(p["argv"][0]).name, "kernels": p["manager_kernels"]}
            for p in sorted(processes, key=lambda p: Path(p["argv"][0]).name)
        ]
    else:  # ddsim: conditional equivalence (timing-dependent steering)
        summary.update(_ddsim_summary(processes))
    violations = sorted({v for p in processes for v in p["violations"]})
    if violations:  # absent when clean, so goldens keep their shape
        summary["violations"] = violations
    return _norm(summary)


def _multiset(items):
    counter = collections.Counter(json.dumps(i, sort_keys=True) for i in items)
    return [[json.loads(k), n] for k, n in sorted(counter.items())]


_CALL_NO = re.compile(r"#\d+$")


def _call_shape(value):
    """A recorded call argument with call numbers dropped (they follow parallel timing)."""
    if isinstance(value, dict) and set(value) == {"future_of"}:
        return {"future_of": _CALL_NO.sub("", value["future_of"])}
    if isinstance(value, list):
        return [_call_shape(v) for v in value]
    if isinstance(value, dict):
        return {k: _call_shape(v) for k, v in value.items()}
    return value


def _parallel_summary(workflow: Workflow, processes: list[dict], exit_code: int) -> dict:
    """An example on a parallel backend: task bodies run in worker processes.

    Calls and executions are multisets.  After a failure, how many other tasks still
    run before shutdown depends on timing, so a failing run is reduced to the set of
    distinct executions that failed (the failure point) and the calls made before it
    cannot be compared either; only the call kinds that were made are kept.
    """
    script = Path(workflow.script).name
    manager = next(p for p in processes if p["argv"] and p["argv"][0].endswith(script))
    calls = [
        {
            "comp": c["comp"],
            "name": c["name"],
            "args": _call_shape(c["args"]),
            "kwargs": _call_shape(c["kwargs"]),
        }
        for c in manager["calls"]
    ]
    executions = [e for p in processes for e in p["executions"]]
    if exit_code:
        failed = [e for e in executions if e.get("outcome") not in (None, "ok")]
        return {
            "call_kinds": [list(k) for k in sorted({(c["comp"], c["name"]) for c in calls})],
            "failed_executions": [e for e, _ in _multiset(failed)],
            "manager_kernels": manager["manager_kernels"],
        }
    return {
        "calls": _multiset(calls),
        "executions": _multiset(executions),
        "manager_kernels": manager["manager_kernels"],
    }


def _ddsim_summary(processes: list[dict]) -> dict:
    manager = next(p for p in processes if p["argv"][0].endswith("run_workflow.py"))
    tasks = [p for p in processes if p is not manager]

    commands = [
        {"name": e["name"], "command": _norm(e.get("result"))}
        for e in manager["executions"]
        if e["comp"] == "executable"
    ]
    script_traces = [
        {
            "script": Path(p["argv"][0]).name,
            "argv": _norm(p["argv"][1:]),
            "kernels": p["manager_kernels"],
        }
        for p in tasks
    ]
    return {
        "call_names": _multiset({"comp": c["comp"], "name": c["name"]} for c in manager["calls"]),
        "commands": _multiset(commands),
        "script_traces": _multiset(script_traces),
        "manager_kernels": manager["manager_kernels"],
    }


def golden_path(name: str, mode: str) -> Path:
    return GOLDENS / name / f"{mode}.json"


def dump(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
