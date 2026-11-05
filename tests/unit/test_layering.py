"""Task processes import the kernels without loading the workflow frameworks."""

import subprocess
import sys

FORBIDDEN = (
    "radical",
    "rhapsody",
    "opentelemetry",
    "dragon",
    "wfMiniAPI.api",
    "wfMiniAPI.telemetry",
)


def test_kernel_import_loads_no_framework():
    code = (
        "import sys, wfMiniAPI.kernel; "
        "print(sorted({m.split('.')[0] if not m.startswith('wfMiniAPI') else m "
        "for m in sys.modules}))"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    for name in FORBIDDEN:
        assert f"'{name}'" not in loaded, f"importing wfMiniAPI.kernel loaded {name}"
