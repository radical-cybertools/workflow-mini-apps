"""Static equivalence check (E1): core text is byte-identical, edges change only as allowed.

    python tests/equivalence/static_check.py build   # write core_manifest.json from the tree
    python tests/equivalence/static_check.py check   # verify the tree against the manifest

The manifest stores, per example, the sha256 of every core segment and the edge
lines verbatim; per whole-file entry, the file's sha256.  ``check`` needs neither
git nor the baseline checkout, so it also runs in CI.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EDGE = HERE / "edge.toml"
MANIFEST = HERE / "core_manifest.json"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _first(lines: list[str], patterns: list[str], start: int) -> int | None:
    for pattern in patterns:
        rx = re.compile(pattern)
        for i in range(start, len(lines)):
            if rx.search(lines[i]):
                return i
    return None


def segments(text: str, specs: list[dict]) -> tuple[dict[str, str], list[str]]:
    """Return {segment: text} and the remaining (edge) lines, in order."""
    lines = text.splitlines(keepends=True)
    taken = [False] * len(lines)
    found = {}
    for spec in specs:
        if "start_at" in spec:
            begin = _first(lines, spec["start_at"], 0)
        else:
            anchor = _first(lines, spec["start_after"], 0)
            begin = None if anchor is None else anchor + 1
        if begin is None:
            raise ValueError(f"segment {spec['name']!r}: start anchor not found")
        if "end_at" in spec:
            end = _first(lines, spec["end_at"], begin)
            end = None if end is None else end + 1
        elif "end_before" in spec:
            end = _first(lines, spec["end_before"], begin)
        else:
            end = len(lines)
        if end is None:
            raise ValueError(f"segment {spec['name']!r}: end anchor not found")
        found[spec["name"]] = "".join(lines[begin:end])
        for i in range(begin, end):
            taken[i] = True
    edge = [ln.rstrip("\n") for ln, t in zip(lines, taken) if not t]
    return found, edge


def _roots(args) -> dict[str, Path]:
    if args.ddsim is None:  # the integrated tree: 0c4f0c2 export + ddsim/run_workflow.patch
        sys.path.insert(0, str(HERE))
        import harness

        args.ddsim = str(harness.integrated_ddsim())
    return {"repo": Path(args.repo).resolve(), "ddsim": Path(args.ddsim).resolve()}


def build(args) -> int:
    spec = tomllib.loads(EDGE.read_text())
    roots = _roots(args)
    manifest = {"examples": {}, "whole": {}}
    for ex in spec["example"]:
        segs, edge = segments((roots["repo"] / ex["path"]).read_text(), spec["example_segment"])
        manifest["examples"][ex["id"]] = {
            "segments": {name: _sha(text) for name, text in segs.items()},
            "edge_lines": edge,
        }
    for group in spec["whole"]:
        root = roots[group["root"]]
        manifest["whole"][group["id"]] = {p: _sha((root / p).read_text()) for p in group["paths"]}
    MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"wrote {MANIFEST.relative_to(REPO)}")
    return 0


def _allowed(line: str, patterns: list[str]) -> bool:
    return any(re.fullmatch(p, line.strip()) for p in patterns)


def check(args) -> int:
    spec = tomllib.loads(EDGE.read_text())
    manifest = json.loads(MANIFEST.read_text())
    roots = _roots(args)
    renames = spec.get("renames", {})
    problems = []

    for ex in spec["example"]:
        allow_removed = spec["defaults"]["allow_removed"] + ex.get("allow_removed", [])
        allow_added = spec["defaults"]["allow_added"] + ex.get("allow_added", [])
        text = (roots["repo"] / renames.get(ex["path"], ex["path"])).read_text()
        try:
            segs, edge = segments(text, spec["example_segment"])
        except ValueError as exc:
            problems.append(f"{ex['id']}: {exc}")
            continue
        want = manifest["examples"][ex["id"]]
        for name, digest in want["segments"].items():
            if _sha(segs.get(name, "")) != digest:
                problems.append(f"{ex['id']}: core segment {name!r} changed")
        for line in difflib.ndiff(want["edge_lines"], edge):
            tag, body = line[:2], line[2:]
            if tag == "- " and not _allowed(body, allow_removed):
                problems.append(f"{ex['id']}: edge line removed but not allowed: {body!r}")
            if tag == "+ " and not _allowed(body, allow_added):
                problems.append(f"{ex['id']}: edge line added but not allowed: {body!r}")

    for group in spec["whole"]:
        root = roots[group["root"]]
        if group["root"] == "ddsim" and not root.exists():
            print(f"skip {group['id']}: {root} not found")
            continue
        for path, digest in manifest["whole"][group["id"]].items():
            current = root / renames.get(path, path)
            if not current.exists() and (root / path).exists():  # tree from before the move
                current = root / path
            if not current.exists():
                problems.append(f"{group['id']}: {current} missing")
            elif _sha(current.read_text()) != digest:
                problems.append(f"{group['id']}: {current.relative_to(root)} changed")

    for p in problems:
        print("FAIL", p)
    print("static check:", "OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["build", "check"])
    ap.add_argument("--repo", default=str(REPO), help="workflow-mini-apps tree to check")
    ap.add_argument(
        "--ddsim",
        default=None,
        help="DeepDriveSim tree to check (default: the 0c4f0c2 export with "
        "tests/equivalence/ddsim/run_workflow.patch applied)",
    )
    args = ap.parse_args(argv)
    return build(args) if args.action == "build" else check(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
