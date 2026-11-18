"""Equivalence-harness shim (test-only; never shipped).

Loaded at interpreter start-up because ``tests/equivalence/shim`` is put on
``PYTHONPATH`` by the harness.  It does nothing unless ``WFMINI_EQUIV=1``.

What it does when active:

* records an independent trace (JSONL, one file per process) of
    - every top-level ``wfMiniAPI.kernel`` call (name, bound args, outcome),
    - every AsyncFlow task/block call (name, arguments, dependency edges),
    - task/block body begin and end;
* ``WFMINI_EQUIV_KERNELS=dry`` records kernel calls without executing them;
* ``WFMINI_EQUIV_BACKEND=dragon`` (run under the ``dragon`` launcher) keeps the real
  Dragon backend and only aliases the removed ``DragonExecutionBackendV3`` name to
  ``DragonExecutionBackend``, so the baseline runs unmodified; task order is then
  parallel, and the harness compares multisets.  Task bodies then run in long-lived
  Dragon workers, so each task body draws from its own ``random`` stream, seeded from
  its name and arguments (thread-local); its random output depends only on its inputs,
  not on which worker ran it before.  Otherwise (the default) the shim
* makes the examples runnable on a CPU host, deterministically:
    - ``rhapsody.backends.DragonExecutionBackend`` and ``...V3`` (an ImportError
      on RHAPSODY main) become a one-worker ``ConcurrentExecutionBackend``,
    - ``multiprocessing.set_start_method("dragon")`` becomes a no-op,
    - ``random`` and NumPy are seeded, and reseeded once the engine exists,
    - OpenTelemetry stops drawing from the global ``random`` stream (span and
      trace ids use ``os.urandom``; metric exemplars are off), so turning
      telemetry on cannot shift the workflow's own random numbers;
* asserts that nothing draws from ``random`` between the reseed and the first
  task call (recorded as a ``violation``).

The shim is applied identically to the baseline and to the integrated code,
so whatever it adds cancels out in the comparison.
"""

import os

if os.environ.get("WFMINI_EQUIV") == "1":
    import asyncio
    import functools
    import hashlib
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
    _ON_DRAGON = os.environ.get("WFMINI_EQUIV_BACKEND", "concurrent") == "dragon"

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
    os.environ.setdefault("OTEL_METRICS_EXEMPLAR_FILTER", "always_off")
    _reseeded_state = None  # random state right after the post-engine reseed

    import multiprocessing

    _orig_set_start_method = multiprocessing.set_start_method

    def _set_start_method(method, *args, **kwargs):
        if method == "dragon" and not _ON_DRAGON:
            _record("shim", what="set_start_method(dragon) skipped")
            return None
        return _orig_set_start_method(method, *args, **kwargs)

    multiprocessing.set_start_method = _set_start_method

    if _ON_DRAGON:
        # Module-level `random` functions read a thread-local per-task generator while a
        # task body runs, and the global one otherwise (manager code, the reseed).
        _global_rng = random._inst

        def _rng():
            return getattr(_local, "rng", None) or _global_rng

        def _per_task(name):
            return lambda *a, **k: getattr(_rng(), name)(*a, **k)

        for _name in (
            "random",
            "uniform",
            "randint",
            "randrange",
            "choice",
            "choices",
            "sample",
            "shuffle",
            "gauss",
            "normalvariate",
            "betavariate",
            "expovariate",
        ):
            setattr(random, _name, _per_task(_name))

    # Task bodies are pickled by value to Dragon workers, together with every global
    # they reference: they may reference module-level functions (pickled by reference)
    # but no objects such as `_local`.
    def _task_rng_enter(name, args, kwargs):
        outer = getattr(_local, "rng", None)
        if _ON_DRAGON:
            key = json.dumps([name, _plain(list(args)), _plain(kwargs)], sort_keys=True)
            seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
            _local.rng = random.Random(seed)
        return outer

    def _task_rng_exit(outer):
        _local.rng = outer

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

    def _seed_numpy():
        # Best effort: NumPy's RNG only affects data values, which are not compared.
        # Some environments load numpy lazily (importlib.util.LazyLoader), so no
        # import hook on numpy itself.
        try:
            sys.modules["numpy"].random.seed(_SEED)
        except Exception:
            pass

    def _on_rhapsody_backends(module):
        from concurrent.futures import ThreadPoolExecutor

        if _ON_DRAGON:
            module.DragonExecutionBackendV3 = module.DragonExecutionBackend
            _record("shim", what="DragonExecutionBackendV3 -> DragonExecutionBackend")
            return
        concurrent_backend = module.ConcurrentExecutionBackend

        def dragon_on_cpu(*args, **kwargs):
            _record("shim", what="Dragon backend -> Concurrent(max_workers=1)")
            return concurrent_backend(ThreadPoolExecutor(max_workers=1))

        # Replace both names: DragonExecutionBackend always exists in
        # rhapsody.backends (its constructor fails later without Dragon).
        module.DragonExecutionBackend = dragon_on_cpu
        module.DragonExecutionBackendV3 = dragon_on_cpu

    def _on_otel_id_generator(module):
        def span_id(self):
            return int.from_bytes(os.urandom(8), "big") or 1

        def trace_id(self):
            return int.from_bytes(os.urandom(16), "big") or 1

        module.RandomIdGenerator.generate_span_id = span_id
        module.RandomIdGenerator.generate_trace_id = trace_id

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
                outer_rng = _task_rng_enter(name, args, kwargs)
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
                finally:
                    _task_rng_exit(outer_rng)
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
            global _reseeded_state
            if _reseeded_state is not None:
                # first task/block call: nothing may have drawn from `random` since
                # the reseed (edge code, telemetry, MiniApp)
                if random.getstate() != _reseeded_state:
                    _record("violation", what="global random drawn between reseed and first call")
                _reseeded_state = None
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
            _seed_numpy()
            global _reseeded_state
            _reseeded_state = random.getstate()
            _record("shim", what="reseeded after WorkflowEngine creation")
            self.function_task = _wrap_decorator(self.function_task, "function")
            self.executable_task = _wrap_decorator(self.executable_task, "executable")
            self.block = _wrap_decorator(self.block, "block")

        engine.__init__ = __init__
        engine.__wfmini_equiv_wrapped__ = True

    _hooks.update(
        {
            "opentelemetry.sdk.trace.id_generator": _on_otel_id_generator,
            "rhapsody.backends": _on_rhapsody_backends,
            "wfMiniAPI.kernel": _on_kernel,
            "radical.asyncflow.workflow_manager": _on_workflow_manager,
        }
    )

    _record("process", argv=sys.argv, pid=os.getpid(), dry=_DRY, seed=_SEED)
