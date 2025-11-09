#!/usr/bin/env bash
# Run only WO-6 (assumes WO-1 through WO-5 receipts already exist)

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

# Activate venv
source venv/bin/activate

# Set determinism environment
export PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Run WO-6 only
python -c "
import sys
from pathlib import Path
sys.path.insert(0, 'src')
from arcsolver.harness import run_stage_wo6
run_stage_wo6(Path('data'), strict=False, enable_progress=True)
"
