"""Verify a run again, offline, from its files.

    python -m wfMiniAPI.verify <checkpoint dir of the run's engine> [--session <id>]

Reads the run's manifest, written by the run's ``MiniApp.verify()`` beside its report, and
verifies the run again: the run engine's checkpoint, every part's report, each component's
verifier and each check named by import path (those that are not keep the verdict recorded
in the run). Prints the report as JSON; the exit status is its ``exit_hint``: 0 complete and
valid, 3 telemetry incomplete, 4 complete but a check failed.
"""

from __future__ import annotations

import argparse
import json
import sys

from .telemetry.run import jsonable, verify_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m wfMiniAPI.verify", description=__doc__)
    parser.add_argument("directory", help="the checkpoint directory of the run's engine")
    parser.add_argument(
        "--session", help="the run engine's session id, when the directory holds several runs"
    )
    args = parser.parse_args(argv)
    report = verify_run(args.directory, session=args.session)
    print(json.dumps(jsonable(report.to_dict()), indent=1))
    return report.exit_hint


if __name__ == "__main__":
    sys.exit(main())
