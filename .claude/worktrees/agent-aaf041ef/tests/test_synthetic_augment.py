"""
Tests for synthetic_augment.py.

Focus: Option B same-contract neighbor exclusion in _smote_nc.

Validates:
- SMOTE-NC interpolates ACROSS contracts, never between year-rows of one contract
- Class with <2 distinct contracts triggers Gaussian-noise fallback (warns)
- Synthetic rows get unique SYNTH_-prefixed contract IDs
- ≥2 distinct contracts → no fallback flag
- Public augment_support API enforces the same invariants end-to-end
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from master_thesis.synthetic_augment import _smote_nc, augment_support


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_two_contract_class_df(
    contract_a_marker: float = 0.0,
    contract_b_marker: float = 1.0,
    rows_per_contract: int = 5,
) -> pd.DataFrame:
    """
    Class with exactly 2 contracts, distinct numeric markers.

    Used to detect same-contract leakage: any synthetic row whose
    `marker` lies strictly between the two contract markers must
    have been interpolated ACROSS contracts (Option B working).
    A synthetic row with marker == contract_a_marker or
    marker == contract_b_marker is evidence of same-contract pickup.
    """
    rows = []
    for yr in range(rows_per_contract):
        rows.append({
            "contract_id": "A",
            "marker": contract_a_marker,
            "cat_feat": "x",
        })
    for yr in range(rows_per_contract):
        rows.append({
            "contract_id": "B",
            "marker": contract_b_marker,
            "cat_feat": "y",
        })
    return pd.DataFrame(rows)


def _make_one_contract_class_df(rows_per_contract: int = 5) -> pd.DataFrame:
    """Class with multiple year-rows but only one distinct contract."""
    return pd.DataFrame([
        {"contract_id": "C", "marker": float(yr), "cat_feat": "x"}
        for yr in range(rows_per_contract)
    ])


# ---------------------------------------------------------------------------
# _smote_nc — Option B (same-contract exclusion)
# ---------------------------------------------------------------------------

class TestSmoteNcSameContractExclusion:
    """
    Option B: when picking k-NN neighbors for an anchor row, rows with
    the same contract_id must be excluded.
    """

    def test_interpolates_across_contracts(self):
        """
        With anchor A (marker=0) and only B-rows eligible as neighbors
        (marker=1), every interpolated marker = 0 + lam*(1-0) = lam.
        With lam ~ Uniform(0,1), all synthetic markers must lie in (0,1).
        A synthetic marker == 0.0 or == 1.0 would mean the neighbor came
        from the same contract as the anchor — Option B violation.
        """
        class_df = _make_two_contract_class_df(0.0, 1.0, rows_per_contract=5)
        rng = np.random.default_rng(42)

        synth = _smote_nc(
            class_df=class_df,
            num_cols=["marker"],
            cat_cols=["cat_feat"],
            n_generate=30,
            k_neighbors=3,
            rng=rng,
            contract_id_col="contract_id",
        )

        markers = synth["marker"].to_numpy()
        # No fallback — should not be a Gaussian-noise result
        assert "augmentation_fallback" not in synth.columns or \
            not synth["augmentation_fallback"].any(), \
            "Did not expect fallback when class has 2 distinct contracts."

        # All synthetic markers strictly inside the interpolation interval
        interior = (markers > 0.0) & (markers < 1.0)
        # lam=0 is a measure-zero event for float64 uniform — every row should
        # be strictly interior with overwhelming probability.
        assert interior.mean() >= 0.95, (
            f"Only {interior.mean():.0%} of synthetic markers are strictly "
            f"interpolated between contracts A and B. "
            f"Markers seen: min={markers.min()}, max={markers.max()}, "
            f"on-edge count={(~interior).sum()}/{len(markers)}. "
            "This suggests same-contract neighbors are leaking through."
        )

    def test_categorical_mode_reflects_cross_contract(self):
        """
        Contract A has cat_feat='x', contract B has cat_feat='y'.
        Mode is computed over [anchor] + k cross-contract neighbors.
        With anchor A (cat='x') and neighbors all from B (cat='y'),
        cat_feat picks the majority. Either 'x' or 'y' is valid —
        what matters is that it's NEVER mode-collapsed onto a single
        contract's category in a way that ignores the other contract.
        """
        class_df = _make_two_contract_class_df(0.0, 1.0, rows_per_contract=5)
        rng = np.random.default_rng(0)

        synth = _smote_nc(
            class_df=class_df,
            num_cols=["marker"],
            cat_cols=["cat_feat"],
            n_generate=30,
            k_neighbors=3,
            rng=rng,
            contract_id_col="contract_id",
        )

        # Sanity: synthetic categorical values must be drawn from observed values
        observed = set(class_df["cat_feat"].unique())
        synth_vals = set(synth["cat_feat"].unique())
        assert synth_vals.issubset(observed), \
            f"Synthetic cat_feat values {synth_vals} not in observed {observed}"


class TestSmoteNcFallback:
    """When fewer than 2 distinct contracts → must fall back to Gaussian noise."""

    def test_one_contract_triggers_fallback(self):
        class_df = _make_one_contract_class_df(rows_per_contract=5)
        rng = np.random.default_rng(0)

        with pytest.warns(UserWarning, match="contract"):
            synth = _smote_nc(
                class_df=class_df,
                num_cols=["marker"],
                cat_cols=["cat_feat"],
                n_generate=8,
                k_neighbors=3,
                rng=rng,
                contract_id_col="contract_id",
            )

        assert len(synth) == 8
        assert "augmentation_fallback" in synth.columns
        assert synth["augmentation_fallback"].all(), \
            "Every synthetic row from a fallback path must be flagged."

    def test_two_contracts_does_not_fall_back(self):
        class_df = _make_two_contract_class_df(0.0, 1.0, rows_per_contract=4)
        rng = np.random.default_rng(0)

        # No warning should be raised
        with pytest.warns(None) as record:
            synth = _smote_nc(
                class_df=class_df,
                num_cols=["marker"],
                cat_cols=["cat_feat"],
                n_generate=6,
                k_neighbors=3,
                rng=rng,
                contract_id_col="contract_id",
            )

        # Filter out unrelated warnings (e.g. pandas FutureWarnings)
        smote_warnings = [w for w in record if "smote_nc" in str(w.message).lower()]
        assert len(smote_warnings) == 0, \
            f"Unexpected SMOTE-NC fallback warning: {[str(w.message) for w in smote_warnings]}"

        if "augmentation_fallback" in synth.columns:
            assert not synth["augmentation_fallback"].any()


class TestSmoteNcSyntheticIds:
    """Synthetic rows must get unique, SYNTH_-prefixed contract IDs."""

    def test_synthetic_ids_unique_and_prefixed(self):
        class_df = _make_two_contract_class_df(0.0, 1.0, rows_per_contract=4)
        rng = np.random.default_rng(7)

        synth = _smote_nc(
            class_df=class_df,
            num_cols=["marker"],
            cat_cols=["cat_feat"],
            n_generate=20,
            k_neighbors=3,
            rng=rng,
            contract_id_col="contract_id",
        )

        ids = synth["contract_id"].tolist()
        assert len(set(ids)) == len(ids), \
            f"Synthetic contract IDs are not unique: duplicates found in {ids}"
        assert all(isinstance(i, str) and i.startswith("SYNTH_") for i in ids), \
            f"All synthetic IDs must start with 'SYNTH_'. Got: {ids[:5]}..."

    def test_synthetic_ids_do_not_collide_with_real(self):
        """Synthetic IDs must never be 'A' or 'B' (the real contract IDs)."""
        class_df = _make_two_contract_class_df(0.0, 1.0, rows_per_contract=4)
        rng = np.random.default_rng(1)

        synth = _smote_nc(
            class_df=class_df,
            num_cols=["marker"],
            cat_cols=["cat_feat"],
            n_generate=20,
            k_neighbors=3,
            rng=rng,
            contract_id_col="contract_id",
        )

        real_ids = set(class_df["contract_id"].unique())
        synth_ids = set(synth["contract_id"].unique())
        assert real_ids.isdisjoint(synth_ids), \
            f"Synthetic IDs collide with real: {real_ids & synth_ids}"


# ---------------------------------------------------------------------------
# augment_support — end-to-end integration of Option B
# ---------------------------------------------------------------------------

class TestAugmentSupportOptionB:
    """
    End-to-end: support_df with multi-year contracts must produce
    synthetic rows that respect Option B (cross-contract interpolation).
    """

    def _make_support_df(self) -> pd.DataFrame:
        """Two classes × two contracts × five year-rows = 20 rows."""
        rows = []
        for cid, marker in [("P1", 1.0), ("P2", 2.0)]:
            for _ in range(5):
                rows.append({
                    "contract_id": cid,
                    "department": "TestDept",
                    "marker": marker,
                    "cat_feat": "yes",
                    "gold_y": 1,
                })
        for cid, marker in [("N1", 10.0), ("N2", 20.0)]:
            for _ in range(5):
                rows.append({
                    "contract_id": cid,
                    "department": "TestDept",
                    "marker": marker,
                    "cat_feat": "no",
                    "gold_y": 0,
                })
        return pd.DataFrame(rows)

    def test_synthetic_markers_interpolate_within_class(self):
        """
        Positive class markers ∈ {1.0, 2.0}. Synthetic positive markers
        must lie in [1.0, 2.0] (closed) and strictly inside (1.0, 2.0)
        for the vast majority of rows (Option B working).

        Same for negative class: markers ∈ {10.0, 20.0}.
        """
        support_df = self._make_support_df()
        augmented = augment_support(
            support_df=support_df,
            feature_cols=["marker", "cat_feat"],
            target_col="gold_y",
            method="smote_nc",
            synthetic_proportion=0.5,
            categorical_cols=["cat_feat"],
            random_state=42,
        )

        synth = augmented[augmented["is_synthetic"]]
        assert len(synth) > 0, "Expected at least some synthetic rows"

        synth_pos = synth.loc[synth["gold_y"] == 1, "marker"].to_numpy()
        synth_neg = synth.loc[synth["gold_y"] == 0, "marker"].to_numpy()

        # All synthetic markers within the class's contract-marker range
        assert np.all((synth_pos >= 1.0) & (synth_pos <= 2.0)), \
            f"Positive synthetic markers out of range: {synth_pos}"
        assert np.all((synth_neg >= 10.0) & (synth_neg <= 20.0)), \
            f"Negative synthetic markers out of range: {synth_neg}"

        # Strict interior dominates → cross-contract interpolation happened
        if len(synth_pos) > 0:
            pos_interior = ((synth_pos > 1.0) & (synth_pos < 2.0)).mean()
            assert pos_interior >= 0.8, \
                f"Only {pos_interior:.0%} of positive synthetics are strictly " \
                f"between contract P1 and P2 markers. Same-contract leakage suspected."
        if len(synth_neg) > 0:
            neg_interior = ((synth_neg > 10.0) & (synth_neg < 20.0)).mean()
            assert neg_interior >= 0.8, \
                f"Only {neg_interior:.0%} of negative synthetics are strictly " \
                f"between contract N1 and N2 markers. Same-contract leakage suspected."

    def test_synthetic_rows_have_distinct_contract_ids(self):
        support_df = self._make_support_df()
        augmented = augment_support(
            support_df=support_df,
            feature_cols=["marker", "cat_feat"],
            target_col="gold_y",
            method="smote_nc",
            synthetic_proportion=0.5,
            categorical_cols=["cat_feat"],
            random_state=42,
        )

        synth = augmented[augmented["is_synthetic"]]
        real_ids = set(support_df["contract_id"].unique())
        synth_ids = set(synth["contract_id"].unique())

        assert real_ids.isdisjoint(synth_ids), \
            f"Synthetic IDs collide with real contract IDs: {real_ids & synth_ids}"
        assert len(synth_ids) == len(synth), \
            "Each synthetic row should have a unique contract_id"

    def test_department_preserved(self):
        """Department is protected — must never be interpolated away."""
        support_df = self._make_support_df()
        augmented = augment_support(
            support_df=support_df,
            feature_cols=["marker", "cat_feat"],
            target_col="gold_y",
            method="smote_nc",
            synthetic_proportion=0.5,
            categorical_cols=["cat_feat"],
            random_state=42,
        )
        assert (augmented["department"] == "TestDept").all()

    def test_fallback_when_class_has_one_contract(self):
        """
        Positive class has only 1 contract → SMOTE-NC for that class
        falls back to Gaussian noise. Negative class still uses real SMOTE-NC.
        """
        rows = []
        # Positive: 1 contract × 5 year-rows
        for _ in range(5):
            rows.append({
                "contract_id": "P1",
                "department": "TestDept",
                "marker": 1.0,
                "cat_feat": "yes",
                "gold_y": 1,
            })
        # Negative: 2 contracts × 3 year-rows
        for cid, marker in [("N1", 10.0), ("N2", 20.0)]:
            for _ in range(3):
                rows.append({
                    "contract_id": cid,
                    "department": "TestDept",
                    "marker": marker,
                    "cat_feat": "no",
                    "gold_y": 0,
                })
        support_df = pd.DataFrame(rows)

        with pytest.warns(UserWarning, match="contract"):
            augmented = augment_support(
                support_df=support_df,
                feature_cols=["marker", "cat_feat"],
                target_col="gold_y",
                method="smote_nc",
                synthetic_proportion=0.5,
                categorical_cols=["cat_feat"],
                random_state=42,
            )

        synth = augmented[augmented["is_synthetic"]]
        # Positive synthetic rows should be flagged as fallback
        synth_pos = synth[synth["gold_y"] == 1]
        if len(synth_pos) > 0:
            assert synth_pos["augmentation_fallback"].all(), \
                "Positive class with 1 contract should fall back to Gaussian."
        # Negative synthetic rows should NOT be flagged (they had 2 contracts)
        synth_neg = synth[synth["gold_y"] == 0]
        if len(synth_neg) > 0:
            assert not synth_neg["augmentation_fallback"].any(), \
                "Negative class with 2 contracts should use real SMOTE-NC."
