#!/usr/bin/env bash
# =============================================================================
# run_optimization_cycle.sh  Stage 2 Optimization Cycle 1
#
# Focused FOMAML / MAML LR sweep to resolve the collapse observed at
# lr=0.01 (k=2/2, Logistics, A_weak_only).
#
# Structure
# ---------
#   Part 1 (6 experiments)
#     Preset: optimization_cycle_ref
#     Methods: zero_shot, finetune
#     Steps: [1, 3, 5]   LR: default (lr-agnostic)
#     Purpose: clean reference floor for this benchmark round
#
#   Part 2 (27 experiments)
#     Preset: optimization_cycle_adaptive
#     Methods: anil, fomaml, maml
#     Steps: [1, 3, 5]   LR: [0.001, 0.005, 0.01]
#     Purpose: LR sweep to find stable operating points for FOMAML and MAML
#
#   Total: 33 experiments, meta_batch_size=4, n_repeats=20, --force-rerun
#
# Usage (from project root with .venv active)
# -------------------------------------------
#   source .venv/bin/activate
#   bash experiments/run_optimization_cycle.sh
#
#   # Skip reference methods if already cleanly run:
#   bash experiments/run_optimization_cycle.sh --adaptive-only
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG="$SCRIPT_DIR/stage2_config.yaml"
RUNNER="$SCRIPT_DIR/run_stage2.py"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"

ADAPTIVE_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --adaptive-only) ADAPTIVE_ONLY=true ;;
    esac
done

echo "=============================================================="
echo "  Stage 2 -- Optimization Cycle 1"
echo "  Target     : Logistics  |  k = 2 pos / 2 neg"
echo "  Init       : A_weak_only"
echo "  meta_batch : 4  |  n_repeats : 20  |  FORCE RERUN: YES"
echo "  Steps      : [1, 3, 5]"
echo "  LR (adapt) : [0.001, 0.005, 0.01]"
echo "=============================================================="
echo ""

START=$(date +%s)

if [ "$ADAPTIVE_ONLY" = false ]; then
    echo "Part 1 / 2 -- Reference methods (zero_shot, finetune)"
    echo "  Preset: optimization_cycle_ref  (6 experiments)"
    echo ""
    python "$RUNNER" \
        --config "$CONFIG" \
        --preset optimization_cycle_ref \
        --force-rerun
    echo ""
    echo "Part 1 complete."
    echo ""
fi

echo "Part 2 / 2 -- Adaptive methods with LR sweep (anil, fomaml, maml)"
echo "  Preset: optimization_cycle_adaptive  (27 experiments)"
echo ""
python "$RUNNER" \
    --config "$CONFIG" \
    --preset optimization_cycle_adaptive \
    --force-rerun

echo ""
echo "Part 2 complete."
echo ""

END=$(date +%s)
ELAPSED=$(( END - START ))
echo "=============================================================="
echo "  Optimization cycle complete in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
echo "  Artifacts : models/stage_2/experiments/"
echo "  Registry  : models/stage_2/experiments/experiment_summary.csv"
echo "  Next step : open notebooks/17.1_stage2_analysis_dashboard.ipynb"
echo "              and run all cells in the Optimization Cycle section"
echo "=============================================================="
