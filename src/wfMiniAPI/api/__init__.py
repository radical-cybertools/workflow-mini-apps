"""wfMiniAPI public API for workflow mini-apps.

    from wfMiniAPI.api import MiniApp

Importing this package imports AsyncFlow/RHAPSODY telemetry; task processes that only
need kernels import ``wfMiniAPI.kernel`` instead.
"""

from .app import AttachError, ComponentReport, MiniApp, MiniAppReport, PartRef, Run
from .preflight import BackendUnavailableError, require_dragon

__all__ = [
    "AttachError",
    "BackendUnavailableError",
    "ComponentReport",
    "MiniApp",
    "MiniAppReport",
    "PartRef",
    "Run",
    "require_dragon",
]
