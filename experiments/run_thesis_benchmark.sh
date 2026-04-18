#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# run_thesis_benchmark.sh — Master's Thesis Stage 2 Main Benchmark
#
# Runs the full, clean, force-rerun thesis benchmark for Logistics:
#
#   Part 1 — Reference methods (zero_shot, finetune)
#     6 experiments: 2 methods × 3 steps × k=2/2, A_weak_only
#     No LR sweep (lr-agnostic or irrelevant for these methods)
#
#   Part 2 — Adaptive methods with LR sweep (anil, fomaml, maml)
#     27 experiments: 3 methods × 3 steps × 3 lr × k=2/2, A_weak_only
#     inner_lr ∈ {0.001, 0.005, 0.01}
#
#   TOTAL: 33 experiments, meta_batch_size=4, n_repeats=20
#
# Usage (from project root with venv active):
#   source .venv/bin/activate
#   bash experiments/run_thesis_benchmark.sh
#
#   # Skip the reference methods if already run cleanly:
#   bash experiments/run_thesis_benchmark.sh --adaptive-only
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG="$SCRIPT_DIR/stage2_config.yaml"
RUNNER="$SCRIPT_DIR/run_stage2.py"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"

# ── Parse args ───────────────────────────────────────────────────────────
ADAPTIVE_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --adaptive-only) ADAPTIVE_ONLY=true ;;
    esac
done

echo "═══════════════════════════════════════════════════════════════"
echo "  Stage 2 — Thesis Main Benchmark"
echo "  Config  : $CONFIG"
echo "  Presets : thesis_ref_methods + thesis_adaptive_lr_sweep"
echo "  Support : k=2/2 | Init: A_weak_only | meta_batch=4"
echo "  Steps   : [1, 3, 5] | LR (adaptive): [0.001, 0.005, 0.01]"
echo "  Repeats : 20 | FORCE RERUN: YES"
echo "═══════════════════════════════════════════════════════════════"
echo ""

START=$(date +%s)

# ── Part 1: Reference methods ────────────────────────────────────────────
if [ "$ADAPTIVE_ONLY" = false ]; then
    echo "▶ Part 1 / 2 — Reference methods (zero_shot, finetune)"
    echo "  6 experiments with --force-rerun"
    echo ""
    python "$RUNNER" \
        --config "$CONFIG" \
        --preset thesis_ref_methods \
        --force-rerun
    echo ""
    echo "✓ Part 1 complete."
    echo ""
fi

# ── Part 2: Adaptive methods with LR sweep ───────────────────────────────
echo "▶ Part 2 / 2 — Adaptive methods with LR sweep (anil, fomaml, maml)"
echo "  27 experiments with --force-rerun"
echo ""
python "$RUNNER" \
    --config "$CONFIG" \
    --preset thesis_adaptive_lr_sweep \
    --force-rerun

echo ""
echo "✓ Part 2 complete."
echo ""

# ── Summary ──────────────────────────────────────────────────────────────
END=$(date +%s)
ELAPSED=$(( END - START ))
echo "═══════════════════════════════════════════════════════════════"
echo "  Benchmark complete in $(( ELAPSED / 60 ))m $(( ELAPSED % 60 ))s"
echo "  Artifacts: models/stage_2/experiments/"
echo "  Registry : models/stage_2/experiments/experiment_summary.csv"
echo "═══════════════════════════════════════════════════════════════"
