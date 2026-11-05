"""Equivalence-harness shim (test-only; never shipped).

Loaded at interpreter start-up because ``tests/equivalence/shim`` is put on
``PYTHONPATH`` by the harness.  It does nothing unless ``WFMINI_EQUIV=1``.

What it does when active:

* records an independent trace (JSONL, one file per process) of
    - every top-level ``wfMiniAPI.kernel`` call (name, bound args, outcome),
    - every AsyncFlow task/block call (name, arguments, dependency edges),
    - task/block body begin and end;
* ``WFMINI_EQUIV_KERNELS=dry`` records kernel calls without executing them;
* makes the unmodified examples runnable on a CPU host, deterministically:
    - ``rhapsody.backends.DragonExecutionBackendV3`` (an ImportError on
      RHAPSODY main) becomes a one-worker ``ConcurrentExecutionBackend``,
    - ``multiprocessing.set_start_method("dragon")`` becomes a no-op,
    - ``random`` and NumPy are seeded.

The shim is applied identically to the baseline and to the integrated code,
so whatever it adds cancels out in the comparison.
"""

import os

if os.environ.get("WFMINI_EQUIV") == "1":
    import asyncio
    import functools
    import importlib.abc
    import importlib.util
    import inspect
    import itertools
    import json
    import random
    import sys
    import threading

    _OUT_DIR = os.environ.get("WFMINI_EQUIV_OUT", ".")
    _DRY = os.environ.get("WFMINI_EQUIV_KERNELS", "real") == "dry"
    _SEED = int(os.environ.get("WFMINI_EQUIV_SEED", "1234"))

    _lock = threading.Lock()
    _seq = itertools.count()
    _local = threading.local()
    _trace_path = os.path.join(_OUT_DIR, f"trace.{os.getpid()}.jsonl")
    os.makedirs(_OUT_DIR, exist_ok=True)
    _trace = open(_trace_path, "a", buffering=1, encoding="utf-8")

    def _record(kind, **fields):
        with _lock:
            rec = {
                "seq": next(_seq),
                "kind": kind,
                "thread": threading.current_thread().name,
                **fields,
            }
            _trace.write(json.dumps(rec, default=repr) + "\n")

    def _plain(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (list, tuple)):
            return [_plain(v) for v in value]
        if isinstance(value, dict):
            return {str(k): _plain(v) for k, v in value.items()}
        return f"<{type(value).__name__}>"

    # ------------------------------------------------------------------ #
    # determinism and CPU runnability
    random.seed(_SEED)

    import multiprocessing

    _orig_set_start_method = multiprocessing.set_start_method

    def _set_start_method(method, *args, **kwargs):
        if method == "dragon":
            _record("shim", what="set_start_method(dragon) skipped")
            return None
        return _orig_set_start_method(method, *args, **kwargs)

    multiprocessing.set_start_method = _set_start_method

    # ------------------------------------------------------------------ #
    # post-import hooks
    _hooks = {}

    class _HookLoader(importlib.abc.Loader):
        def __init__(self, loader, hook):
            self._loader, self._hook = loader, hook

        def create_module(self, spec):
            return self._loader.create_module(spec)

        def exec_module(self, module):
            self._loader.exec_module(module)
            self._hook(module)

    class _HookFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            hook = _hooks.get(name)
            if hook is None:
                return None
            for finder in sys.meta_path:
                if finder is self:
                    continue
                spec = finder.find_spec(name, path, target)
                if spec is not None:
                    if spec.loader is not None:
                        spec.loader = _HookLoader(spec.loader, hook)
                    return spec
            return None

    sys.meta_path.insert(0, _HookFinder())

    def _on_numpy(module):
        module.random.seed(_SEED)

    def _on_rhapsody_backends(module):
        if hasattr(module, "DragonExecutionBackendV3"):
            return
        from concurrent.futures import ThreadPoolExecutor

        concurrent_backend = module.ConcurrentExecutionBackend

        def DragonExecutionBackendV3(*args, **kwargs):  # noqa: N802
            _record("shim", what="DragonExecutionBackendV3 -> Concurrent(max_workers=1)")
            return concurrent_backend(ThreadPoolExecutor(max_workers=1))

        module.DragonExecutionBackendV3 = DragonExecutionBackendV3

    # ------------------------------------------------------------------ #
    # independent kernel recorder
    def _wrap_kernel(name, func):
        sig = inspect.signature(func)

        @functools.wraps(func)
        def recorder(*args, **kwargs):
            depth = getattr(_local, "depth", 0)
            try:
                bound = sig.bind_partial(*args, **kwargs)
                call_args = {k: _plain(v) for k, v in bound.arguments.items()}
            except TypeError:
                call_args = {"args": _plain(args), "kwargs": _plain(kwargs)}
            if _DRY and depth == 0:
                _record("kernel", name=name, depth=depth, args=call_args, outcome="dry")
                return None
            _local.depth = depth + 1
            try:
                result = func(*args, **kwargs)
            except BaseException as exc:
                _record(
                    "kernel",
                    name=name,
                    depth=depth,
                    args=call_args,
                    outcome="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            finally:
                _local.depth = depth
            _record("kernel", name=name, depth=depth, args=call_args, outcome="ok")
            return result

        recorder.__wfmini_equiv_wrapped__ = True
        return recorder

    def _on_kernel(module):
        for name in module.list_kernels():
            func = getattr(module, name, None)
            if callable(func) and not getattr(func, "__wfmini_equiv_wrapped__", False):
                setattr(module, name, _wrap_kernel(name, func))

    # ------------------------------------------------------------------ #
    # AsyncFlow task/block recorder
    _future_ids = {}
    _call_ids = itertools.count()

    def _describe_arg(value):
        if isinstance(value, asyncio.Future) and id(value) in _future_ids:
            return {"future_of": _future_ids[id(value)]}
        return _plain(value)

    def _wrap_body(func, comp):
        name = func.__name__

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def body(*args, **kwargs):
                _record("begin", comp=comp, name=name)
                try:
                    result = await func(*args, **kwargs)
                except BaseException as exc:
                    _record(
                        "end",
                        comp=comp,
                        name=name,
                        outcome="error",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    raise
                _record(
                    "end",
                    comp=comp,
                    name=name,
                    outcome="ok",
                    result=result if comp == "executable" else None,
                )
                return result

            return body
        return func

    def _wrap_call(decorated, name, comp):
        @functools.wraps(decorated)
        def call(*args, **kwargs):
            call_id = f"{name}#{next(_call_ids)}"
            _record(
                "call",
                comp=comp,
                name=name,
                call=call_id,
                args=[_describe_arg(a) for a in args],
                kwargs={k: _describe_arg(v) for k, v in kwargs.items()},
            )
            fut = decorated(*args, **kwargs)
            if isinstance(fut, asyncio.Future):
                _future_ids[id(fut)] = call_id
            return fut

        return call

    def _wrap_decorator(decorator, comp):
        @functools.wraps(decorator)
        def outer(possible_func=None, **options):
            if possible_func is None:
                inner = decorator(**options)
                return lambda f: _wrap_call(inner(_wrap_body(f, comp)), f.__name__, comp)
            return _wrap_call(
                decorator(_wrap_body(possible_func, comp), **options), possible_func.__name__, comp
            )

        return outer

    def _on_workflow_manager(module):
        engine = module.WorkflowEngine
        if getattr(engine, "__wfmini_equiv_wrapped__", False):
            return
        orig_init = engine.__init__

        @functools.wraps(orig_init)
        def __init__(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            # Reseed after all imports: installed packages may consume the global
            # stream at import time, which would make goldens environment-dependent.
            random.seed(_SEED)
            if "numpy" in sys.modules:
                sys.modules["numpy"].random.seed(_SEED)
            _record("shim", what="reseeded after WorkflowEngine creation")
            self.function_task = _wrap_decorator(self.function_task, "function")
            self.executable_task = _wrap_decorator(self.executable_task, "executable")
            self.block = _wrap_decorator(self.block, "block")

        engine.__init__ = __init__
        engine.__wfmini_equiv_wrapped__ = True

    _hooks.update(
        {
            "numpy": _on_numpy,
            "rhapsody.backends": _on_rhapsody_backends,
            "wfMiniAPI.kernel": _on_kernel,
            "radical.asyncflow.workflow_manager": _on_workflow_manager,
        }
    )

    _record("process", argv=sys.argv, pid=os.getpid(), dry=_DRY, seed=_SEED)
