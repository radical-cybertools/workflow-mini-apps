"""wfMiniAPI public API for workflow mini-apps.

    from wfMiniAPI.api import MiniApp

Importing this package imports AsyncFlow/RHAPSODY telemetry; task processes that only
need kernels import ``wfMiniAPI.kernel`` instead.
"""

from .app import AttachError, MiniApp, MiniAppReport
from .preflight import BackendUnavailableError, require_dragon

__all__ = ["AttachError", "BackendUnavailableError", "MiniApp", "MiniAppReport", "require_dragon"]
