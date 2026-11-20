"""The kernel library imports cleanly and keeps the surface existing workflows use."""

import subprocess
import sys

import pytest

# Kernel registration order before the Phase 1 split (wfMiniAPI/kernel.py @ 55cd20f).
KERNELS = [
    "sleep",
    "get_device_module",
    "writeSingleRank",
    "writeNonMPI",
    "writeWithMPI",
    "readNonMPI",
    "readWithMPI",
    "MPIallReduce",
    "MPIallGather",
    "dataCopyH2D",
    "dataCopyD2H",
    "matMulSimple2D",
    "matMulGeneral",
    "fft",
    "fftn",
    "axpy",
    "axpy_fast",
    "inplaceCompute",
    "generateRandomNumber",
    "scatterAdd",
    "top_k",
]


def _python(code):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)


def test_import_prints_nothing():
    assert _python("import wfMiniAPI.kernel").stdout == ""


def test_package_import_has_no_side_effects():
    out = _python("import sys, wfMiniAPI; print('wfMiniAPI.kernel' in sys.modules)")
    assert out.stdout.strip() == "False"


def test_compat_import_paths():
    import wfMiniAPI
    import wfMiniAPI.kernel as wf
    from wfMiniAPI import kernel, kernels

    assert wf is kernel is wfMiniAPI.kernel
    assert wf.matMulSimple2D is kernels.matMulSimple2D
    assert wfMiniAPI.registry.list_kernels is kernels.list_kernels


def test_registration_order_unchanged():
    from wfMiniAPI import kernel

    assert kernel.list_kernels() == KERNELS


@pytest.mark.parametrize(
    "call",
    [
        lambda k: k.matMulSimple2D("cpu", 8),
        lambda k: k.generateRandomNumber("cpu", 8),
        lambda k: k.scatterAdd("cpu", 10, 10),  # KR-06
        lambda k: k.top_k("cpu", 16, 4),
    ],
)
def test_cpu_kernels_run(call):
    from wfMiniAPI import kernel

    call(kernel)


def test_gpu_without_cupy_fails_as_before():
    from wfMiniAPI import kernel

    if kernel.CUPY_AVAILABLE:
        pytest.skip("CuPy is installed")
    with pytest.raises(ImportError, match="CuPy is not installed"):
        kernel.matMulSimple2D("gpu", 8)
