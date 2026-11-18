"""Contract check (E2): every in-scope workflow attaches MiniApp in its edge.

    python tests/equivalence/contract_check.py [--repo PATH] [--ddsim PATH]

For each example the AST must show, inside ``async def run_miniapp``:

* ``<flow> = await WorkflowEngine.create(...)``;
* ``<telemetry> = await <flow>.start_telemetry(...)``;
* ``async with MiniApp(<flow>, <telemetry>), <flow>.workflow_scope(...)`` whose body
  awaits ``miniapp_workflow(...)``;

and ``async def miniapp_workflow`` must be a module-level function.  DeepDriveSim's
``run_workflow.py`` must enter ``MiniApp(<engine>, <telemetry>)`` together with a
``workflow_scope`` around its replicas.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import harness  # noqa: E402


def _awaited_call(node):
    if isinstance(node, ast.Await) and isinstance(node.value, ast.Call):
        return node.value
    return None


def _assignments(func) -> dict[str, ast.Call]:
    """name -> awaited call it is bound from (``x = await f(...)``)."""
    out = {}
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, call = node.targets[0], _awaited_call(node.value)
            if isinstance(target, ast.Name) and call is not None:
                out[target.id] = call
    return out


def _is_attr_call(call: ast.Call, owner: str, attr: str) -> bool:
    f = call.func
    return (
        isinstance(f, ast.Attribute)
        and f.attr == attr
        and isinstance(f.value, ast.Name)
        and f.value.id == owner
    )


def _miniapp_with(func, flow: str, telemetry: str):
    """The ``async with`` that enters MiniApp(flow, telemetry) and flow.workflow_scope(...)."""
    for node in ast.walk(func):
        if not isinstance(node, ast.AsyncWith):
            continue
        calls = [i.context_expr for i in node.items if isinstance(i.context_expr, ast.Call)]
        has_app = any(
            isinstance(c.func, ast.Name)
            and c.func.id == "MiniApp"
            and [a.id for a in c.args[:2] if isinstance(a, ast.Name)] == [flow, telemetry]
            for c in calls
        )
        has_scope = any(_is_attr_call(c, flow, "workflow_scope") for c in calls)
        if has_app and has_scope:
            return node
    return None


def check_example(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.AsyncFunctionDef)}
    problems = []
    if "miniapp_workflow" not in funcs:
        problems.append("no module-level `async def miniapp_workflow`")
    workflow = funcs.get("run_miniapp")
    if workflow is None:
        return problems + ["no module-level `async def run_miniapp`"]
    bound = _assignments(workflow)
    flow = next(
        (
            n
            for n, c in bound.items()
            if isinstance(c.func, ast.Attribute) and c.func.attr == "create"
        ),
        None,
    )
    if flow is None:
        return problems + ["no `<flow> = await WorkflowEngine.create(...)`"]
    telemetry = next(
        (n for n, c in bound.items() if _is_attr_call(c, flow, "start_telemetry")), None
    )
    if telemetry is None:
        return problems + [f"no `<telemetry> = await {flow}.start_telemetry(...)`"]
    with_node = _miniapp_with(workflow, flow, telemetry)
    if with_node is None:
        return problems + [
            f"no `async with MiniApp({flow}, {telemetry}), {flow}.workflow_scope(...)`"
        ]
    calls_core = any(
        (c := _awaited_call(n)) is not None
        and isinstance(c.func, ast.Name)
        and c.func.id == "miniapp_workflow"
        for stmt in with_node.body
        for n in ast.walk(stmt)
    )
    if not calls_core:
        problems.append("the MiniApp block does not await miniapp_workflow(...)")
    return problems


def check_ddsim(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)]
    for func in funcs:
        bound = _assignments(func)
        for engine, call in bound.items():
            if not (isinstance(call.func, ast.Attribute) and call.func.attr == "create"):
                continue
            telemetry = next(
                (n for n, c in bound.items() if _is_attr_call(c, engine, "start_telemetry")), None
            )
            if telemetry and _miniapp_with(func, engine, telemetry):
                return []
    return ["no `async with MiniApp(<engine>, <telemetry>), <engine>.workflow_scope(...)`"]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=str(harness.REPO))
    ap.add_argument(
        "--ddsim", default=None, help="integrated DeepDriveSim tree (skipped if missing)"
    )
    args = ap.parse_args(argv)
    problems = []
    for wf, rel in harness.EXAMPLES.items():
        problems += [f"{wf}: {p}" for p in check_example(Path(args.repo) / rel)]
    ddsim_root = Path(args.ddsim) if args.ddsim else harness.integrated_ddsim()
    ddsim = ddsim_root / harness.WORKFLOWS["ddsim"].script
    if ddsim.exists():
        problems += [f"ddsim: {p}" for p in check_ddsim(ddsim)]
    else:
        print(f"skip ddsim: {ddsim} not found")
    for p in problems:
        print("FAIL", p)
    print("contract check:", "OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
