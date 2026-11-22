"""Contract check (E2): every in-scope workflow attaches MiniApp in its edge.

    python tests/equivalence/contract_check.py [--repo PATH] [--ddsim PATH]

For each example the AST must show, inside ``async def run_miniapp``:

* ``<flow> = await WorkflowEngine.create(...)``;
* ``<telemetry> = await <flow>.start_telemetry(...)``;
* ``<app> = MiniApp(<flow>, <telemetry>)``;
* ``async with <app>, <flow>.workflow_scope(...)`` whose body awaits
  ``miniapp_workflow(...)``, directly inside a ``try`` whose ``finally`` is the script's
  shutdown (design B): a nested ``try`` that awaits ``<flow>.shutdown()``, with a
  ``finally`` that awaits ``<telemetry>.stop()`` and then calls ``<app>.verify()``;

and ``async def miniapp_workflow`` must be a module-level function that no longer shuts
the engine down.  DeepDriveSim's ``run_workflow.py`` must have the same shape around its
replicas (it has no ``miniapp_workflow``).
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


def _plain_assignments(func) -> dict[str, ast.Call]:
    """name -> call it is bound from without await (``x = f(...)``)."""
    out = {}
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
        ):
            out[node.targets[0].id] = node.value
    return out


def _is_attr_call(call: ast.Call, owner: str, attr: str) -> bool:
    f = call.func
    return (
        isinstance(f, ast.Attribute)
        and f.attr == attr
        and isinstance(f.value, ast.Name)
        and f.value.id == owner
    )


def _miniapp_app(func, flow: str, telemetry: str) -> str | None:
    """The name bound to ``MiniApp(<flow>, <telemetry>)``."""
    for name, call in _plain_assignments(func).items():
        if (
            isinstance(call.func, ast.Name)
            and call.func.id == "MiniApp"
            and [a.id for a in call.args[:2] if isinstance(a, ast.Name)] == [flow, telemetry]
        ):
            return name
    return None


def _is_awaited_attr_call(stmt, owner: str, attr: str) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and (c := _awaited_call(stmt.value)) is not None
        and _is_attr_call(c, owner, attr)
    )


def _is_attr_call_stmt(stmt, owner: str, attr: str) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and _is_attr_call(stmt.value, owner, attr)
    )


def _miniapp_with(func, flow: str, telemetry: str):
    """The ``async with <app>, <flow>.workflow_scope(...)`` in design B's try/finally.

    Returns (node, problem): node is the ``async with``; problem names what is missing.
    """
    app = _miniapp_app(func, flow, telemetry)
    if app is None:
        return None, f"no `<app> = MiniApp({flow}, {telemetry})`"
    for node in ast.walk(func):
        if not isinstance(node, ast.Try):
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.AsyncWith):
                continue
            items = [i.context_expr for i in stmt.items]
            has_app = any(isinstance(e, ast.Name) and e.id == app for e in items)
            has_scope = any(
                isinstance(e, ast.Call) and _is_attr_call(e, flow, "workflow_scope") for e in items
            )
            if not (has_app and has_scope):
                continue
            # finally: try: await flow.shutdown() / finally: await telemetry.stop(); app.verify()
            inner = node.finalbody[0] if len(node.finalbody) == 1 else None
            ok = (
                isinstance(inner, ast.Try)
                and len(inner.body) == 1
                and _is_awaited_attr_call(inner.body[0], flow, "shutdown")
                and len(inner.finalbody) == 2
                and _is_awaited_attr_call(inner.finalbody[0], telemetry, "stop")
                and _is_attr_call_stmt(inner.finalbody[1], app, "verify")
            )
            if not ok:
                return None, (
                    f"the try around `async with {app}, ...` must end with `finally: try: "
                    f"await {flow}.shutdown() finally: await {telemetry}.stop(); {app}.verify()`"
                )
            return stmt, None
    return None, f"no `async with {app}, {flow}.workflow_scope(...)` directly inside a try"


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
    with_node, problem = _miniapp_with(workflow, flow, telemetry)
    if with_node is None:
        return problems + [problem]
    calls_core = any(
        (c := _awaited_call(n)) is not None
        and isinstance(c.func, ast.Name)
        and c.func.id == "miniapp_workflow"
        for stmt in with_node.body
        for n in ast.walk(stmt)
    )
    if not calls_core:
        problems.append("the MiniApp block does not await miniapp_workflow(...)")
    core = funcs.get("miniapp_workflow")
    if core is not None and any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "shutdown"
        for n in ast.walk(core)
    ):
        problems.append("miniapp_workflow still shuts the engine down (the script does)")
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
            if telemetry:
                node, problem = _miniapp_with(func, engine, telemetry)
                return [] if node is not None else [problem]
    return ["no `<engine> = await WorkflowEngine.create(...)` with start_telemetry()"]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=str(harness.REPO))
    ap.add_argument(
        "--ddsim", default=None, help="integrated DeepDriveSim tree (skipped if missing)"
    )
    args = ap.parse_args(argv)
    problems = []
    for wf, rel in {**harness.EXAMPLES, **harness.AGENT_EXAMPLES}.items():
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
