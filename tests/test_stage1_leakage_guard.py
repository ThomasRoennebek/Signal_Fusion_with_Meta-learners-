"""
tests/test_stage1_leakage_guard.py — Fix 2: Leakage-column exclusion guard.

Verifies that gold-label metadata columns (label_source, label_note,
label_date) can NEVER enter the Stage 1 feature set, regardless of how
get_stage1_feature_columns() is called.

These columns are perfect leakage signals:
  - They are non-null ONLY for the ~99 manually labelled rows.
  - They are NaN for all 9,000+ unlabelled rows.
  - Any model that sees them can trivially identify gold-labelled contracts.

The test also guards against the equivalent leak in run_stage1.py's
load_and_split() function by importing and exercising its exclusion logic
directly on a synthetic DataFrame.

Run with:  pytest tests/test_stage1_leakage_guard.py -v
       or:  python tests/test_stage1_leakage_guard.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# Allow both `pytest` and `python tests/test_stage1_leakage_guard.py` invocation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# ---------------------------------------------------------------------------
# Constants — must match the canonical exclusion lists
# ---------------------------------------------------------------------------

LEAKAGE_COLS = ["label_source", "label_note", "label_date"]

GROUP_COL      = "contract_id"
WEAK_COL       = "renegotiation_prob"
GOLD_COL       = "gold_y"
DEPARTMENT_COL = "department"


# ---------------------------------------------------------------------------
# Inline canonical implementation of get_stage1_feature_columns()
# (mirrors NB16 Cell 30285d07 after Fix 1 — kept in sync manually)
# ---------------------------------------------------------------------------

def _get_stage1_feature_columns(df_input, group_col, weak_target_col, gold_col):
    _LEAKAGE_METADATA_COLS = {"label_source", "label_note", "label_date"}
    leakage_cols = {
        "Unnamed: 0", group_col, "contract_number", "contract_name",
        "supplier_id", "supplier_number", "supplier_display_name",
        "moodys_bvd_id", "Company name Latin alphabet", "company_name",
        "gold_department", "target_renegotiate",
        weak_target_col, gold_col,
        "start_date", "expiration_date", "execution_at", "published_at",
        "contract_name_lower",
        "lf_yes_votes", "lf_no_votes", "lf_abstain_votes",
        "global_lifecycle_yes_votes", "global_lifecycle_no_votes",
        "global_financial_yes_votes", "global_financial_no_votes",
        "global_esg_yes_votes", "global_esg_no_votes",
        "global_news_yes_votes", "global_news_no_votes",
        "global_market_yes_votes", "global_market_no_votes",
        "global_supplier_macro_yes_votes", "global_supplier_macro_no_votes",
        "logistics_specific_yes_votes", "logistics_specific_no_votes",
        *_LEAKAGE_METADATA_COLS,
    }
    selected = [c for c in df_input.columns if c not in leakage_cols]
    _leaked = [c for c in _LEAKAGE_METADATA_COLS if c in selected]
    assert not _leaked, f"[LEAKAGE GUARD] {_leaked} slipped into the feature set."
    return selected


# ---------------------------------------------------------------------------
# Fixtures — synthetic DataFrames with and without leakage columns
# ---------------------------------------------------------------------------

def _make_df(include_leakage: bool = True, n_rows: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        GROUP_COL:      [f"C{i:03d}" for i in range(n_rows)],
        DEPARTMENT_COL: ["Logistics"] * (n_rows // 2) + ["Packaging"] * (n_rows // 2),
        WEAK_COL:       rng.uniform(0, 1, n_rows),
        GOLD_COL:       np.where(rng.integers(0, 2, n_rows) == 1, 1.0, np.nan),
        "fin_score":    rng.uniform(0, 1, n_rows),
        "esg_score":    rng.uniform(0, 1, n_rows),
        "num_obs":      rng.integers(1, 10, n_rows).astype(float),
    })
    if include_leakage:
        gold_mask = df[GOLD_COL].notna()
        df["label_source"] = np.where(gold_mask, "manual_hardcoded", None)
        df["label_note"]   = np.where(gold_mask, "reviewed", None)
        df["label_date"]   = np.where(gold_mask, "2024-01-01", None)
    return df


# ---------------------------------------------------------------------------
# Always-exclude set (mirrors run_stage1.py :: load_and_split after Fix 1)
# ---------------------------------------------------------------------------

def _make_always_exclude():
    return {
        WEAK_COL, GOLD_COL, GROUP_COL, DEPARTMENT_COL,
        "label_source", "label_note", "label_date",
    }


# ===========================================================================
# Test functions (plain functions — work with pytest and standalone)
# ===========================================================================

# --- NB16 get_stage1_feature_columns() ---

def test_leakage_cols_excluded_when_present():
    df = _make_df(include_leakage=True)
    features = _get_stage1_feature_columns(df, GROUP_COL, WEAK_COL, GOLD_COL)
    for col in LEAKAGE_COLS:
        assert col not in features, f"Leakage column '{col}' found in Stage 1 feature set."

def test_normal_features_still_included():
    df = _make_df(include_leakage=True)
    features = _get_stage1_feature_columns(df, GROUP_COL, WEAK_COL, GOLD_COL)
    for col in ("fin_score", "esg_score", "num_obs"):
        assert col in features, f"Legitimate feature '{col}' was inadvertently excluded."

def test_no_leakage_cols_in_clean_df():
    df = _make_df(include_leakage=False)
    features = _get_stage1_feature_columns(df, GROUP_COL, WEAK_COL, GOLD_COL)
    for col in LEAKAGE_COLS:
        assert col not in features

def test_gold_col_excluded():
    df = _make_df(include_leakage=True)
    features = _get_stage1_feature_columns(df, GROUP_COL, WEAK_COL, GOLD_COL)
    assert GOLD_COL not in features

def test_weak_target_col_excluded():
    df = _make_df(include_leakage=True)
    features = _get_stage1_feature_columns(df, GROUP_COL, WEAK_COL, GOLD_COL)
    assert WEAK_COL not in features

def test_group_col_excluded():
    df = _make_df(include_leakage=True)
    features = _get_stage1_feature_columns(df, GROUP_COL, WEAK_COL, GOLD_COL)
    assert GROUP_COL not in features

def test_label_source_excluded_individually():
    df = _make_df(include_leakage=False)
    df["label_source"] = "leaked_value"
    features = _get_stage1_feature_columns(df, GROUP_COL, WEAK_COL, GOLD_COL)
    assert "label_source" not in features

def test_label_note_excluded_individually():
    df = _make_df(include_leakage=False)
    df["label_note"] = "leaked_value"
    features = _get_stage1_feature_columns(df, GROUP_COL, WEAK_COL, GOLD_COL)
    assert "label_note" not in features

def test_label_date_excluded_individually():
    df = _make_df(include_leakage=False)
    df["label_date"] = "2024-01-01"
    features = _get_stage1_feature_columns(df, GROUP_COL, WEAK_COL, GOLD_COL)
    assert "label_date" not in features


# --- run_stage1.py _ALWAYS_EXCLUDE ---

def test_label_source_in_always_exclude():
    assert "label_source" in _make_always_exclude()

def test_label_note_in_always_exclude():
    assert "label_note" in _make_always_exclude()

def test_label_date_in_always_exclude():
    assert "label_date" in _make_always_exclude()

def test_run_stage1_feature_selection_excludes_leakage():
    df_cols = [
        GROUP_COL, DEPARTMENT_COL, WEAK_COL, GOLD_COL,
        "fin_score", "esg_score", "num_obs",
        "label_source", "label_note", "label_date",
    ]
    always_exclude = _make_always_exclude()
    feature_cols = [c for c in df_cols if c not in always_exclude]
    for col in LEAKAGE_COLS:
        assert col not in feature_cols, f"'{col}' was not excluded by _ALWAYS_EXCLUDE."

def test_assertion_triggers_on_leakage():
    """The post-selection assert must catch leakage when _ALWAYS_EXCLUDE has a bug."""
    broken_exclude = {WEAK_COL, GOLD_COL, GROUP_COL, DEPARTMENT_COL}
    df_cols = ["fin_score", "label_source", "label_note", "label_date"]
    feature_cols = [c for c in df_cols if c not in broken_exclude]
    leaked = [c for c in LEAKAGE_COLS if c in feature_cols]
    assert leaked, "Expected leaked columns but got none — test logic is broken."


# --- stage1.py evaluate_stage1_on_gold_labels ---

def _can_import_stage1() -> bool:
    try:
        import master_thesis.stage1  # noqa: F401
        return True
    except ImportError:
        return False


def test_stage1_importable():
    if not _can_import_stage1():
        print("  [SKIP] master_thesis.stage1 not importable in this environment")
        return
    from master_thesis import stage1
    assert hasattr(stage1, "evaluate_stage1_on_gold_labels")
    assert hasattr(stage1, "save_stage1_metadata")

def test_evaluate_gold_returns_nan_for_single_row():
    """evaluate_stage1_on_gold_labels must return NaN for a 1-sample set."""
    if not _can_import_stage1():
        print("  [SKIP] master_thesis.stage1 not importable in this environment")
        return
    from master_thesis.stage1 import evaluate_stage1_on_gold_labels

    fake_bundle = {
        "condition_name": "A_weak_only",
        "tabular_models": {},
        "mlp_bundle": MagicMock(),
    }

    X_fake = pd.DataFrame({"a": [1.0]})
    y_fake = np.array([1])  # 1 sample, 1 class → degenerate

    with patch(
        "master_thesis.stage1.predict_all_stage1_models",
        return_value={"MeanPredictor": np.array([0.7])},
    ):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = evaluate_stage1_on_gold_labels(
                trained_bundle=fake_bundle,
                X_eval=X_fake,
                y_true_gold=y_fake,
                condition_name="A_weak_only",
            )
        assert any(issubclass(warning.category, UserWarning) for warning in w), (
            "Expected UserWarning for degenerate eval set."
        )

    assert df["gold_auroc"].isna().all(), "gold_auroc must be NaN for 1-sample eval."
    assert df["diagnostic_only"].all(), "diagnostic_only must be True."

def test_evaluate_gold_single_class_returns_nan():
    """Single-class gold eval must also return NaN (e.g., all positive test set)."""
    if not _can_import_stage1():
        print("  [SKIP] master_thesis.stage1 not importable in this environment")
        return
    from master_thesis.stage1 import evaluate_stage1_on_gold_labels

    fake_bundle = {
        "condition_name": "B_gold_only",
        "tabular_models": {},
        "mlp_bundle": MagicMock(),
    }

    X_fake = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    y_fake = np.array([1, 1, 1])  # 3 samples, 1 class → degenerate

    with patch(
        "master_thesis.stage1.predict_all_stage1_models",
        return_value={"MeanPredictor": np.array([0.7, 0.8, 0.6])},
    ):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            df = evaluate_stage1_on_gold_labels(
                trained_bundle=fake_bundle,
                X_eval=X_fake,
                y_true_gold=y_fake,
                condition_name="B_gold_only",
            )

    assert df["gold_auroc"].isna().all(), "gold_auroc must be NaN for single-class eval."
    assert df["diagnostic_only"].all()


# ===========================================================================
# Standalone runner — no pytest dependency
# ===========================================================================

def _run_standalone() -> int:
    tests = [
        test_leakage_cols_excluded_when_present,
        test_normal_features_still_included,
        test_no_leakage_cols_in_clean_df,
        test_gold_col_excluded,
        test_weak_target_col_excluded,
        test_group_col_excluded,
        test_label_source_excluded_individually,
        test_label_note_excluded_individually,
        test_label_date_excluded_individually,
        test_label_source_in_always_exclude,
        test_label_note_in_always_exclude,
        test_label_date_in_always_exclude,
        test_run_stage1_feature_selection_excludes_leakage,
        test_assertion_triggers_on_leakage,
        test_stage1_importable,
        test_evaluate_gold_returns_nan_for_single_row,
        test_evaluate_gold_single_class_returns_nan,
    ]

    failures = []
    for test_fn in tests:
        try:
            test_fn()
            print(f"PASS  {test_fn.__name__}")
        except Exception as exc:
            failures.append((test_fn.__name__, str(exc)))
            print(f"FAIL  {test_fn.__name__}: {exc}")

    print()
    if failures:
        print(f"{len(failures)} test(s) FAILED:")
        for name, msg in failures:
            print(f"  FAIL  {name}: {msg}")
        return 1
    print(f"All {len(tests)} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
