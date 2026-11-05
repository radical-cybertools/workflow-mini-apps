"""Run workflows under the equivalence shim and capture or check goldens.

python tests/equivalence/run_equiv.py capture [--workflow W] [--mode M]   # on the baseline
python tests/equivalence/run_equiv.py check   [--workflow W] [--mode M] [--keep]
"""

from __future__ import annotations

import argparse
import json
import sys

import compare
import harness


def selected(args) -> list[tuple[str, str]]:
    names = [args.workflow] if args.workflow else list(harness.WORKFLOWS)
    pairs = []
    for name in names:
        modes = harness.WORKFLOWS[name].modes
        pairs += [(name, m) for m in modes if args.mode in (None, m)]
    return pairs


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["capture", "check"])
    ap.add_argument("--workflow", choices=list(harness.WORKFLOWS))
    ap.add_argument("--mode", choices=["dry", "real"])
    ap.add_argument(
        "--keep", action="store_true", help="keep run directories under .equiv-work/last/"
    )
    args = ap.parse_args(argv)

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


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
