#!/usr/bin/env bash
# WO-0 run_harness.sh - Deterministic harness runner.
#
# Sets environment variables for determinism and runs the ARC solver harness.
#
# Usage:
#   ./scripts/run_harness.sh [--upto-wo N] [--strict]
#
# Examples:
#   ./scripts/run_harness.sh --upto-wo 0              # Run WO-0 only
#   ./scripts/run_harness.sh --upto-wo 1 --strict    # Run up to WO-1, fail on error

set -euo pipefail

# Get script directory (works even when sourced or symlinked)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ============================================================================
# Determinism environment (contracts 05_contracts.md / Global)
# ============================================================================
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0

echo "[run_harness] Determinism env set:" >&2
echo "  OMP_NUM_THREADS=${OMP_NUM_THREADS}" >&2
echo "  OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}" >&2
echo "  MKL_NUM_THREADS=${MKL_NUM_THREADS}" >&2
echo "  NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS}" >&2
echo "  PYTHONHASHSEED=${PYTHONHASHSEED}" >&2

# ============================================================================
# Python path setup (add src/ to PYTHONPATH)
# ============================================================================
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
echo "[run_harness] PYTHONPATH=${PYTHONPATH}" >&2

# ============================================================================
# Activate venv if present
# ============================================================================
if [[ -d "${REPO_ROOT}/venv" ]]; then
    echo "[run_harness] Activating venv at ${REPO_ROOT}/venv" >&2
    source "${REPO_ROOT}/venv/bin/activate"
else
    echo "[run_harness] WARNING: No venv found at ${REPO_ROOT}/venv" >&2
    echo "[run_harness] Using system Python (may not match contracts)" >&2
fi

# ============================================================================
# Run harness
# ============================================================================
cd "${REPO_ROOT}"

# Default args
DATA_ROOT="${DATA_ROOT:-data}"
UPTO_WO=""
STRICT_FLAG=""
PROGRESS_FLAG="--progress"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --upto-wo)
            UPTO_WO="$2"
            shift 2
            ;;
        --strict)
            STRICT_FLAG="--strict"
            shift
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --progress)
            PROGRESS_FLAG="--progress"
            shift
            ;;
        --no-progress)
            PROGRESS_FLAG="--no-progress"
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--upto-wo N] [--strict] [--data-root DIR] [--progress|--no-progress]" >&2
            exit 1
            ;;
    esac
done

# Require --upto-wo
if [[ -z "${UPTO_WO}" ]]; then
    echo "Error: --upto-wo is required" >&2
    echo "Usage: $0 --upto-wo N [--strict] [--data-root DIR] [--progress|--no-progress]" >&2
    exit 1
fi

echo "[run_harness] Running: python -m arcsolver.harness --data-root ${DATA_ROOT} --upto-wo ${UPTO_WO} ${STRICT_FLAG} ${PROGRESS_FLAG}" >&2
exec python -m arcsolver.harness \
    --data-root "${DATA_ROOT}" \
    --upto-wo "${UPTO_WO}" \
    ${STRICT_FLAG} \
    ${PROGRESS_FLAG}
