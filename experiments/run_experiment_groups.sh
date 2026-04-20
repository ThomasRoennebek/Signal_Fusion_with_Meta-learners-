#!/usr/bin/env bash
# run_experiment_groups.sh
# Run Stage 2 experiment groups A-E in the order recommended for the
# MAML/FOMAML optimization program.
#
# Recommended order: B -> E -> C -> D
#
# Usage (from project root):
#   bash experiments/run_experiment_groups.sh               # run B, E, C, D
#   bash experiments/run_experiment_groups.sh B             # Group B only
#   bash experiments/run_experiment_groups.sh B E C D       # explicit order
#   bash experiments/run_experiment_groups.sh --force-rerun B      # force re-run Group B
#   bash experiments/run_experiment_groups.sh --force-rerun B E C D
#
# Flags (must come before group letters):
#   --force-rerun   Ignore existing artifacts and re-run every experiment.
#                   Use this when you want fully clean, comparable results
#                   for a sweep (e.g. the Group B LR sweep).
#
# All groups respect skip_if_exists=true by default — already-completed
# experiments are skipped automatically unless --force-rerun is set.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG="$SCRIPT_DIR/stage2_config.yaml"
RUNNER="$SCRIPT_DIR/run_stage2.py"

cd "$PROJECT_ROOT"

# Locate the project Python (prefers .venv, falls back to system python3)
if [ -f ".venv/bin/python3" ] || [ -L ".venv/bin/python3" ]; then
    PYTHON=".venv/bin/python3"
elif [ -f ".venv/bin/python" ] || [ -L ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

echo "Using Python: $PYTHON  ($($PYTHON --version 2>&1))"
echo "Project root: $PROJECT_ROOT"
echo ""

# Parse optional --force-rerun flag (must be first argument if present)
FORCE_RERUN_FLAG=""
if [ "${1:-}" = "--force-rerun" ]; then
    FORCE_RERUN_FLAG="--force-rerun"
    shift
    echo "Force-rerun mode: existing artifacts will be overwritten."
    echo ""
fi

run_group() {
    local group="$1"
    local preset="$2"
    local description="$3"
    local est_time="$4"
    echo "============================================================"
    echo "  GROUP $group: $description"
    echo "  Preset: $preset  |  Est. time: $est_time"
    if [ -n "$FORCE_RERUN_FLAG" ]; then
        echo "  Mode: force-rerun (existing artifacts will be overwritten)"
    fi
    echo "============================================================"
    "$PYTHON" "$RUNNER" \
        --config "$CONFIG" \
        --preset "$preset" \
        $FORCE_RERUN_FLAG
    echo "  Group $group finished."
    echo ""
}

# Determine which groups to run (default: B E C D — recommended order)
GROUPS_TO_RUN="${*:-B E C D}"

for group in $GROUPS_TO_RUN; do
    case "$group" in
        A)
            run_group A clean_logistics_baseline \
                "All 5 methods, Logistics, A_weak_only, k=2/2, steps=[1,3,5] (already complete; will skip)" \
                "~0 s"
            ;;
        B)
            run_group B maml_fomaml_lr_sweep \
                "FOMAML/MAML x steps=[1,2,3,5] x lr=[0.0005,0.001,0.002,0.005,0.01] — 40 experiments" \
                "~3-6 h"
            ;;
        C)
            run_group C initialization_ablation \
                "ANIL/FOMAML/MAML x {A_weak_only, C_hybrid} x k=2/2 x steps=3 x lr=0.005 — 6 experiments (3 from B skip)" \
                "~30-60 min"
            ;;
        D)
            run_group D k_shot_sweep_logistics \
                "zero_shot/ANIL/FOMAML/MAML x k={1,2,3,4} x steps=3 — 16 experiments (4 skip)" \
                "~1-2 h"
            ;;
        E)
            run_group E richer_department_check \
                "zero_shot/ANIL/FOMAML/MAML x {Devices & Needles, Packaging Material} x k=2/2 x steps=3 x lr=0.005 — 8 experiments" \
                "~45-90 min"
            ;;
        *)
            echo "Unknown group: $group (expected A-E)"
            exit 1
            ;;
    esac
done

echo "All requested groups finished."
echo "Registry: models/stage_2/experiments/experiment_summary.csv"
echo "Open notebooks/17.1_stage2_analysis_dashboard.ipynb to analyze results."
