"""Optional, explicit preflight checks an edge can call before building a backend."""

from __future__ import annotations

import os

__all__ = ["BackendUnavailableError", "require_dragon"]

_HINT = "launch the workflow under Dragon, e.g. `dragon run_workflow.py`, or choose another backend"


class BackendUnavailableError(RuntimeError):
    """The requested execution backend cannot run in this process."""


def require_dragon() -> None:
    """Raise BackendUnavailableError unless Dragon is installed and this process runs under it.

    Cheap: imports Dragon's facts module and looks for the launch parameters the
    ``dragon`` launcher exports; it never starts a Dragon runtime or a worker pool.
    """
    try:
        from dragon.infrastructure import facts
    except ImportError as exc:
        raise BackendUnavailableError(
            f"Dragon is not installed ({exc}); install wfMiniAPI[dragon] or {_HINT}"
        ) from None
    launched = any(
        os.environ.get(facts.PREFIX + name) for name in facts.env_vars if name.startswith("GS_")
    )
    if not launched:
        raise BackendUnavailableError(
            f"this process was not started by the Dragon launcher: {_HINT}"
        )
