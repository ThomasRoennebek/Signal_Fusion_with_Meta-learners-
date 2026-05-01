"""
Tests for synthetic_diagnostics.py.

Validates per-class, per-feature, and overall behavior of:
- distance_to_nearest_real
- ks_per_feature
- categorical_mode_preservation
- compute_realism_diagnostics (entry point)

Uses small toy fixtures with known expected values so the tests
are deterministic and easy to debug.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from master_thesis.synthetic_diagnostics import (
    categorical_mode_preservation,
    compute_realism_diagnostics,
    distance_to_nearest_real,
    ks_per_feature,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _two_class_real() -> pd.DataFrame:
    """6 real rows: 3 class-1 (around marker=0), 3 class-0 (around marker=10)."""
    return pd.DataFrame([
        {"marker": 0.0, "extra": 0.0, "cat_feat": "x", "gold_y": 1},
        {"marker": 1.0, "extra": 0.0, "cat_feat": "x", "gold_y": 1},
        {"marker": 2.0, "extra": 0.0, "cat_feat": "y", "gold_y": 1},
        {"marker": 10.0, "extra": 0.0, "cat_feat": "p", "gold_y": 0},
        {"marker": 11.0, "extra": 0.0, "cat_feat": "p", "gold_y": 0},
        {"marker": 12.0, "extra": 0.0, "cat_feat": "q", "gold_y": 0},
    ])


def _two_class_synth_close() -> pd.DataFrame:
    """4 synth rows close to their real same-class neighbors."""
    return pd.DataFrame([
        {"marker": 0.5, "extra": 0.0, "cat_feat": "x", "gold_y": 1},
        {"marker": 1.5, "extra": 0.0, "cat_feat": "x", "gold_y": 1},
        {"marker": 10.5, "extra": 0.0, "cat_feat": "p", "gold_y": 0},
        {"marker": 11.5, "extra": 0.0, "cat_feat": "p", "gold_y": 0},
    ])


# ---------------------------------------------------------------------------
# distance_to_nearest_real
# ---------------------------------------------------------------------------

class TestDistanceToNearestReal:
    def test_matches_hand_computation(self):
        real = _two_class_real()
        synth = _two_class_synth_close()
        out = distance_to_nearest_real(synth, real, ["marker", "extra"])
        # Each synth row sits 0.5 from its nearest same-class real row.
        assert math.isclose(out["dist_to_nearest_real_mean"], 0.5, abs_tol=1e-9)
        assert math.isclose(out["dist_to_nearest_real_p95"], 0.5, abs_tol=1e-9)

    def test_per_class_breakdown(self):
        real = _two_class_real()
        synth = _two_class_synth_close()
        out = distance_to_nearest_real(synth, real, ["marker", "extra"])
        per_class = out["dist_to_nearest_real_per_class"]
        assert set(per_class.keys()) == {0, 1}
        for cls in (0, 1):
            assert math.isclose(per_class[cls]["mean"], 0.5, abs_tol=1e-9)
            assert per_class[cls]["n"] == 2

    def test_zero_distance_for_clones(self):
        """Synth row identical to a real row → distance 0."""
        real = _two_class_real()
        synth = real.iloc[:2].copy()  # exact clones of two class-1 rows
        out = distance_to_nearest_real(synth, real, ["marker", "extra"])
        assert out["dist_to_nearest_real_mean"] == 0.0

    def test_empty_synth_returns_nan(self):
        real = _two_class_real()
        synth = real.iloc[0:0].copy()
        out = distance_to_nearest_real(synth, real, ["marker", "extra"])
        assert math.isnan(out["dist_to_nearest_real_mean"])
        assert math.isnan(out["dist_to_nearest_real_p95"])

    def test_handles_nan_via_real_medians(self):
        real = _two_class_real()
        synth = pd.DataFrame([
            {"marker": float("nan"), "extra": 0.0, "cat_feat": "x", "gold_y": 1},
        ])
        out = distance_to_nearest_real(synth, real, ["marker", "extra"])
        # marker NaN filled with real-class-1 median = 1.0; nearest real has marker=1
        assert math.isclose(out["dist_to_nearest_real_mean"], 0.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# ks_per_feature
# ---------------------------------------------------------------------------

class TestKsPerFeature:
    def test_identical_distributions_ks_zero(self):
        real = _two_class_real()
        synth = real.copy()  # identical → KS should be 0 for each (feat, class)
        out = ks_per_feature(real, synth, ["marker", "extra"])
        assert out["ks_max"] == 0.0
        assert out["ks_median"] == 0.0

    def test_disjoint_distributions_ks_one(self):
        """Synth shifted far from real for class-1 → KS = 1.0 on `marker`."""
        real = _two_class_real()
        synth = pd.DataFrame([
            {"marker": 1000.0, "extra": 0.0, "cat_feat": "x", "gold_y": 1},
            {"marker": 1001.0, "extra": 0.0, "cat_feat": "x", "gold_y": 1},
            {"marker": 1002.0, "extra": 0.0, "cat_feat": "x", "gold_y": 1},
            {"marker": 10.0, "extra": 0.0, "cat_feat": "p", "gold_y": 0},
            {"marker": 11.0, "extra": 0.0, "cat_feat": "p", "gold_y": 0},
        ])
        out = ks_per_feature(real, synth, ["marker", "extra"])
        # Class-1 marker should hit KS = 1.0 (fully disjoint)
        assert out["ks_per_feature"]["marker"][1] == 1.0
        assert out["ks_max"] == 1.0

    def test_too_few_samples_returns_nan_for_pair(self):
        real = _two_class_real()
        synth = pd.DataFrame([
            {"marker": 0.5, "extra": 0.0, "cat_feat": "x", "gold_y": 1},  # only 1 row in class 1
        ])
        out = ks_per_feature(real, synth, ["marker"])
        # Single-sample → NaN at that (feat, class)
        assert math.isnan(out["ks_per_feature"]["marker"][1])


# ---------------------------------------------------------------------------
# categorical_mode_preservation
# ---------------------------------------------------------------------------

class TestCategoricalModePreservation:
    def test_full_preservation_when_in_modes(self):
        real = _two_class_real()
        # Real class-1 modes for cat_feat: {"x"} (count 2 vs 1 for "y")
        # Real class-0 modes for cat_feat: {"p"} (count 2 vs 1 for "q")
        synth = pd.DataFrame([
            {"marker": 0.5, "extra": 0.0, "cat_feat": "x", "gold_y": 1},
            {"marker": 1.5, "extra": 0.0, "cat_feat": "x", "gold_y": 1},
            {"marker": 10.5, "extra": 0.0, "cat_feat": "p", "gold_y": 0},
            {"marker": 11.5, "extra": 0.0, "cat_feat": "p", "gold_y": 0},
        ])
        out = categorical_mode_preservation(real, synth, ["cat_feat"])
        assert out["cat_mode_preservation_mean"] == 1.0
        assert out["cat_mode_preservation_min"] == 1.0

    def test_partial_preservation(self):
        real = _two_class_real()
        # Class-1 mode is "x". Half the synth class-1 rows are "y" (not in modes).
        synth = pd.DataFrame([
            {"marker": 0.5, "extra": 0.0, "cat_feat": "x", "gold_y": 1},
            {"marker": 1.5, "extra": 0.0, "cat_feat": "y", "gold_y": 1},
            {"marker": 10.5, "extra": 0.0, "cat_feat": "p", "gold_y": 0},
        ])
        out = categorical_mode_preservation(real, synth, ["cat_feat"])
        # Class 1: 1 of 2 in modes → 0.5. Class 0: 1 of 1 in modes → 1.0.
        per = out["cat_mode_preservation_per_feature"]["cat_feat"]
        assert per[1] == 0.5
        assert per[0] == 1.0
        assert out["cat_mode_preservation_min"] == 0.5

    def test_tied_modes_both_count(self):
        """When real has tied modes, ANY tied value is mode-preserving."""
        real = pd.DataFrame([
            {"cat_feat": "x", "gold_y": 1},
            {"cat_feat": "y", "gold_y": 1},
            {"cat_feat": "p", "gold_y": 0},
        ])
        synth = pd.DataFrame([
            {"cat_feat": "x", "gold_y": 1},
            {"cat_feat": "y", "gold_y": 1},
            {"cat_feat": "p", "gold_y": 0},
        ])
        out = categorical_mode_preservation(real, synth, ["cat_feat"])
        # Class 1: both x and y are modes (tied); both synth values are in modes → 1.0
        assert out["cat_mode_preservation_mean"] == 1.0


# ---------------------------------------------------------------------------
# compute_realism_diagnostics (entry point)
# ---------------------------------------------------------------------------

class TestComputeRealismDiagnostics:
    def test_flat_dict_keys(self):
        real = _two_class_real()
        synth = _two_class_synth_close()
        out = compute_realism_diagnostics(
            real_support_df=real,
            synth_support_df=synth,
            num_cols=["marker", "extra"],
            cat_cols=["cat_feat"],
        )
        for k in (
            "n_synthetic_rows",
            "dist_to_nearest_real_mean", "dist_to_nearest_real_p95",
            "ks_max", "ks_median",
            "cat_mode_preservation_mean", "cat_mode_preservation_min",
        ):
            assert k in out, f"Missing key: {k}"
        assert out["n_synthetic_rows"] == 4
        # No nested dicts unless include_breakdowns=True
        assert "dist_to_nearest_real_per_class" not in out
        assert "ks_per_feature" not in out

    def test_breakdowns_included_on_request(self):
        real = _two_class_real()
        synth = _two_class_synth_close()
        out = compute_realism_diagnostics(
            real_support_df=real,
            synth_support_df=synth,
            num_cols=["marker", "extra"],
            cat_cols=["cat_feat"],
            include_breakdowns=True,
        )
        assert "dist_to_nearest_real_per_class" in out
        assert "ks_per_feature" in out
        assert "cat_mode_preservation_per_feature" in out

    def test_empty_synth_returns_nan_scalars(self):
        real = _two_class_real()
        synth = real.iloc[0:0].copy()
        out = compute_realism_diagnostics(
            real_support_df=real,
            synth_support_df=synth,
            num_cols=["marker", "extra"],
            cat_cols=["cat_feat"],
        )
        assert out["n_synthetic_rows"] == 0
        assert math.isnan(out["dist_to_nearest_real_mean"])
        assert math.isnan(out["ks_max"])
        assert math.isnan(out["cat_mode_preservation_mean"])
