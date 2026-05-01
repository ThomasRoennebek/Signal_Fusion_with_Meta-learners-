#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# run_lr_sweep.sh — Execute the LR Sweep Benchmark for Stage 2
#
# Runs: anil, fomaml, maml × inner_steps {1,3,5} × inner_lr {0.001,0.005,0.01}
# That is 27 experiments total; each takes ~30-60 s on CPU.
#
# Usage:
#   cd "<path to Master Thesis>"
#   source .venv/bin/activate
#   bash experiments/run_lr_sweep.sh
#
#   # To force-rerun experiments that already have results:
#   bash experiments/run_lr_sweep.sh --force-rerun
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

FORCE_RERUN=""
for arg in "$@"; do
    if [ "$arg" = "--force-rerun" ]; then
        FORCE_RERUN="--force-rerun"
    fi
done

echo "==========================================="
echo "  Stage 2 — LR Sweep Benchmark"
echo "  Preset : lr_sweep_benchmark"
echo "  Config : experiments/stage2_config.yaml"
echo "==========================================="
echo ""

PYTHONPATH=src python experiments/run_stage2.py \
    --config experiments/stage2_config.yaml \
    --preset lr_sweep_benchmark \
    $FORCE_RERUN

echo ""
echo "Done. Results in: models/stage_2/experiments/"
