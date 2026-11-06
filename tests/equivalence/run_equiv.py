"""Run workflows under the equivalence shim and capture or check goldens.

python tests/equivalence/run_equiv.py capture [--workflow W] [--mode M]   # on the baseline
python tests/equivalence/run_equiv.py check   [--workflow W] [--mode M] [--keep]
WFMINI_EQUIV_BACKEND=dragon python tests/equivalence/run_equiv.py pair [--workflow W] [--mode M]

`pair` runs the baseline and the integrated tree now, on the same host and backend, and
compares the two summaries; it also checks the integrated run's MiniApp report.  It is
the check for parallel backends (Dragon, a GPU node), where no golden applies.
"""

from __future__ import annotations

import argparse
import json
import sys

import compare
import harness


def selected(args) -> list[tuple[str, str]]:
    names = [args.workflow] if args.workflow else list(harness.WORKFLOWS)
    if args.action == "pair":  # task scripts only: nothing runs on a backend
        names = [n for n in names if harness.WORKFLOWS[n].kind != "tutorial"]
    pairs = []
    for name in names:
        modes = harness.WORKFLOWS[name].modes
        pairs += [(name, m) for m in modes if args.mode in (None, m)]
    return pairs


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["capture", "check", "pair"])
    ap.add_argument("--workflow", choices=list(harness.WORKFLOWS))
    ap.add_argument("--mode", choices=["dry", "real"])
    ap.add_argument(
        "--keep", action="store_true", help="keep run directories under .equiv-work/last/"
    )
    args = ap.parse_args(argv)

    if args.action == "pair":
        return pair(selected(args), args.keep)
    failed = 0
    for name, mode in selected(args):
        keep = harness.WORK / "last" / f"{name}.{mode}" if args.keep else None
        summary = harness.run(name, mode, keep=keep, baseline=args.action == "capture")
        golden = harness.golden_path(name, mode)
        if args.action == "capture":
            harness.dump(summary, golden)
            print(f"{name:18s} {mode:4s} captured (exit {summary['exit_code']})")
            continue
        if not golden.exists():
            print(f"{name:18s} {mode:4s} NO GOLDEN")
            failed += 1
            continue
        diff = compare.first_difference(json.loads(golden.read_text()), summary)
        print(f"{name:18s} {mode:4s} {'identical' if diff is None else 'DIFFERENT ' + diff}")
        failed += diff is not None
    return 1 if failed else 0


# The preserved shared-file bug (U5): parallel tasks read and write the same ./data_0.h5,
# so on a parallel backend HDF5's file lock can make any of them fail, at random.
RACE = "unable to lock file"
ATTEMPTS = 3


def raced(summary: dict) -> bool:
    return any(RACE in (e.get("error") or "") for e in summary.get("failed_executions", []))


def pair(pairs: list[tuple[str, str]], keep: bool) -> int:
    failed = 0
    for name, mode in pairs:
        last = harness.WORK / "last" / harness.BACKEND
        for attempt in range(1, ATTEMPTS + 1):
            base = harness.run(
                name, mode, keep=last / f"{name}.{mode}.baseline" if keep else None, baseline=True
            )
            summary, facts = harness.run(
                name, mode, keep=last / f"{name}.{mode}" if keep else None, telemetry=True
            )
            if not (raced(base) or raced(summary)):
                break
        else:
            print(
                f"{name:18s} {mode:4s} INCONCLUSIVE: the shared ./data_0.h5 lock race "
                f"(preserved bug) hit all {ATTEMPTS} attempts",
                flush=True,
            )
            failed += 1
            continue
        note = "" if attempt == 1 else f"; attempt {attempt}, earlier ones hit the h5 lock race"
        if base["exit_code"] == "timeout" and summary["exit_code"] != "timeout":
            # Known difference: after a task failure the baseline core never reaches its
            # own flow.shutdown(), and on Dragon nothing else shuts the backend down, so
            # the process never exits.  MiniApp's exit shuts the engine down.
            note += "; baseline hangs after the failure, integrated exits"
            base = {**base, "exit_code": summary["exit_code"]}
        diff = compare.first_difference(base, summary)
        problem = diff and "DIFFERENT " + diff or telemetry_problem(name, facts)
        tag = f"exit {summary['exit_code']}{note}"
        print(f"{name:18s} {mode:4s} {problem or 'identical, report complete'} ({tag})", flush=True)
        failed += problem is not None
    return 1 if failed else 0


def telemetry_problem(name: str, facts: dict) -> str | None:
    """The MiniApp facts asserted by test_equivalence.py, as a message."""
    if len(facts["reports"]) != 1:
        return f"TELEMETRY {len(facts['reports'])} reports"
    report = facts["reports"][0]
    if not report["telemetry_complete"]:
        return f"TELEMETRY incomplete: {report['reasons']}"
    if (
        report["events"].get("wfminiapi.RunStarted") != 1
        or report["events"].get("wfminiapi.RunEnded") != 1
    ):
        return f"TELEMETRY run events {report['events']}"
    scopes = set(facts["task_span_scopes"])
    expected = (
        bool(scopes) and all(s and s.startswith("block.") for s in scopes)
        if harness.WORKFLOWS[name].kind == "ddsim"
        else scopes == {name}
    )
    return None if expected else f"TELEMETRY task-span scopes {sorted(map(str, scopes))}"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
