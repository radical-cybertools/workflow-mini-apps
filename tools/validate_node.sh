#!/usr/bin/env bash
# Validate the MiniApp edge on an HPC node: real Dragon backend, and a GPU when present.
#
#   tools/validate_node.sh [--ddsim PATH] [--python PYTHON] [--quick] [--setup | --offline]
#
# Run it from the repository root on a compute node (not a login node).  It
#   1. creates a venv with wfMiniAPI and the pinned AsyncFlow/RHAPSODY/Dragon (and CuPy
#      when `nvidia-smi` finds a GPU),
#   2. exports the baselines (workflow-mini-apps 55cd20f; DeepDriveSim 0c4f0c2 with --ddsim),
#   3. runs each workflow on the baseline and on this tree under `dragon` and compares them
#      (tests/equivalence/run_equiv.py pair), then the MiniApp lifecycle scenarios on Dragon,
#   4. writes everything to one tarball to send back.
#
# --ddsim PATH   a DeepDriveSim clone (adds the DeepDriveSim workflow)
# --python EXE   interpreter for the venv (default python3; 3.11-3.13)
# --quick        dry mode only and no lifecycle scenarios (a smoke test of the script)
# --setup        steps 1-2 only, CuPy included; run it where there is internet (a login node)
# --offline      no installs: use the venv --setup made (a compute node without internet)
# Environment:   CUDA_PACKAGE (default cupy-cuda12x) when the node has another CUDA major;
#                `none` installs no CuPy.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=python3
DDSIM=""
QUICK=0
SETUP=0
OFFLINE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --ddsim) DDSIM=$(cd "$2" && pwd); shift 2 ;;
        --python) PYTHON=$2; shift 2 ;;
        --quick) QUICK=1; shift ;;
        --setup) SETUP=1; shift ;;
        --offline) OFFLINE=1; shift ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [ "$SETUP" = 1 ] && [ "$OFFLINE" = 1 ]; then
    echo "--setup and --offline exclude each other" >&2; exit 2
fi
CUDA_PACKAGE=${CUDA_PACKAGE:-cupy-cuda12x}

cd "$REPO"
WORK=$REPO/.equiv-work
HOST=$(hostname -s)
STAMP=$(date +%Y%m%dT%H%M%S)
OUT=$WORK/validate-$HOST-$STAMP
VENV=$WORK/venv-validate
mkdir -p "$OUT"
exec > >(tee "$OUT/validate.log") 2>&1

GPU=0
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then GPU=1; fi
echo "host=$HOST gpu=$GPU quick=$QUICK ddsim=${DDSIM:-none} tree=$(git rev-parse --short HEAD)"

# 1. venv ------------------------------------------------------------------------
if [ "$OFFLINE" = 1 ]; then
    if [ ! -x "$VENV/bin/python" ]; then
        echo "no $VENV: run tools/validate_node.sh --setup on a node with internet first" >&2
        exit 2
    fi
else
    if [ ! -x "$VENV/bin/python" ]; then
        "$PYTHON" -m venv "$VENV"
        "$VENV/bin/pip" install -q --upgrade pip
    fi
    EXTRAS="workflow,dragon,io,mpi,test"
    "$VENV/bin/pip" install -q -e ".[$EXTRAS]"
    # --setup runs on a node without a GPU, for a GPU node
    if [ "$CUDA_PACKAGE" != none ] && { [ "$GPU" = 1 ] || [ "$SETUP" = 1 ]; }; then
        "$VENV/bin/pip" install -q "$CUDA_PACKAGE"
    fi
fi
PY=$VENV/bin/python
if [ "$GPU" = 1 ] && ! "$PY" -c "import cupy" 2>/dev/null; then
    echo "warning: CuPy is not in $VENV, so the GPU kernels will fail (rerun --setup)"
fi

# 2. baselines (docs/testing/equivalence.md) ----------------------------------------
if [ ! -d "$WORK/wfmini-55cd20f" ]; then
    mkdir -p "$WORK/wfmini-55cd20f"
    git archive 55cd20f | tar -x -C "$WORK/wfmini-55cd20f"
    sed -i '321s/^)$/) if CUPY_AVAILABLE else None/' \
        "$WORK/wfmini-55cd20f/wfMiniAPI/src/wfMiniAPI/kernel.py"
fi
if [ -n "$DDSIM" ] && [ ! -d "$WORK/ddsim-0c4f0c2" ]; then
    mkdir -p "$WORK/ddsim-0c4f0c2"
    git -C "$DDSIM" archive 0c4f0c2 | tar -x -C "$WORK/ddsim-0c4f0c2"
fi
if [ "$SETUP" = 1 ]; then
    echo "== setup done; on the compute node: tools/validate_node.sh --offline${DDSIM:+ --ddsim $DDSIM}"
    exit 0
fi

# environment record
{
    echo "## versions"; "$PY" -c "
from wfMiniAPI.telemetry.capability import _package
for d in ('wfMiniAPI', 'radical.asyncflow', 'rhapsody-py', 'dragonhpc', 'cupy-cuda12x', 'numpy'):
    print(d, _package(d))"
    echo "## pip freeze"; "$VENV/bin/pip" freeze
    echo "## nvidia-smi"; nvidia-smi 2>&1 || true
    echo "## lscpu"; lscpu 2>&1 | head -20 || true
} > "$OUT/environment.txt"

# 3. runs ------------------------------------------------------------------------
export PATH=$VENV/bin:$PATH WFMINI_EQUIV_PYTHON=$PY WFMINI_EQUIV_BACKEND=dragon
# a failing baseline never exits on Dragon (it skips flow.shutdown()); stop it after 300 s
export WFMINI_EQUIV_TIMEOUT=${WFMINI_EQUIV_TIMEOUT:-300}
status=0
MODES=("--mode" "dry")
[ "$QUICK" = 1 ] || MODES=()
rm -rf "$WORK/last/dragon"  # only this run goes into the tarball
echo "== baseline vs integrated on Dragon"
"$PY" tests/equivalence/run_equiv.py pair "${MODES[@]}" --keep || status=1
dragon-cleanup >/dev/null 2>&1 || true
if [ "$QUICK" = 0 ]; then
    echo "== MiniApp lifecycle scenarios on Dragon"
    WFMINI_TEST_DRAGON=1 "$PY" -m pytest -q -m dragon tests/integration || status=1
    dragon-cleanup >/dev/null 2>&1 || true
fi

# 4. tarball ------------------------------------------------------------------------
cp -r "$WORK/last/dragon" "$OUT/runs" 2>/dev/null || true
TAR=$REPO/validation-$HOST-$STAMP.tar.gz
tar -czf "$TAR" -C "$WORK" "$(basename "$OUT")"
echo "== result: $([ $status = 0 ] && echo PASS || echo FAIL); send $TAR"
exit $status
