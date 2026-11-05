"""Compare two equivalence summaries; report the first difference."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def first_difference(a, b, path: str = "$") -> str | None:
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                return f"{path}.{key}: present only in {'golden' if key in a else 'run'}"
            diff = first_difference(a[key], b[key], f"{path}.{key}")
            if diff:
                return diff
        return None
    if isinstance(a, list):
        for i, (x, y) in enumerate(zip(a, b)):
            diff = first_difference(x, y, f"{path}[{i}]")
            if diff:
                return diff
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        return None
    if a != b:
        return f"{path}: {a!r} != {b!r}"
    return None


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: compare.py GOLDEN.json RUN.json", file=sys.stderr)
        return 2
    golden, run = (json.loads(Path(p).read_text()) for p in argv)
    diff = first_difference(golden, run)
    print("identical" if diff is None else f"DIFFERENT {diff}")
    return 0 if diff is None else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
