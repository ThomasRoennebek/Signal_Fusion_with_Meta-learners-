"""
config.py — Central configuration: paths and shared hyperparameters.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]

DATA_RAW       = ROOT / "Data" / "raw"
DATA_INTERIM   = ROOT / "Data" / "interim"
DATA_PROCESSED = ROOT / "Data" / "processed"

MODELS_STAGE1  = ROOT / "models" / "stage_1"
MODELS_STAGE2  = ROOT / "models" / "stage_2"

REPORTS        = ROOT / "reports"
FIGURES        = REPORTS / "figures"
TABLES         = REPORTS / "tables"

# ── Random seed ────────────────────────────────────────────────────────────
SEED = 42



for path in [
    DATA_RAW,
    DATA_INTERIM,
    DATA_PROCESSED,
    MODELS_STAGE1,
    MODELS_STAGE2,
    REPORTS,
    FIGURES,
    TABLES,
]:
    path.mkdir(parents=True, exist_ok=True)
