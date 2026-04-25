"""
Regression test: synthetic rows must NEVER appear in a query set.

This is the single most important invariant of the synthetic-augmentation
pipeline.  If it ever breaks, the thesis evaluation is invalid because
synthetic rows would be evaluated as if they were real gold-labeled data.

The tests here are intentionally exhaustive across seeds and configurations
because the bug surface — accidentally letting a SYNTH_-prefixed contract_id
slip into the query — is the kind of regression a single-fixture test can
miss.  Run with:  ``pytest tests/test_no_synthetic_in_query.py -v``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src/ is on the path so the test runs both via pytest and as a
# stand-alone script (``python tests/test_no_synthetic_in_query.py``).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from master_thesis.episode_sampler import sample_episode_with_synthetic_support
from master_thesis.stage2 import VALID_AUGMENTATION_METHODS


FEATURE_COLS = ["fin_score", "esg_score", "supplier_sector"]


def _make_toy_dept_df(
    n_pos_contracts: int = 5,
    n_neg_contracts: int = 5,
    years_per_contract: int = 3,
    seed: int = 0,
    department_name: str = "TestDept",
) -> pd.DataFrame:
    """Build a small gold-labeled department DataFrame for testing.

    Each contract gets ``years_per_contract`` rows so the multi-row-per-contract
    case (which is what motivates the same-contract neighbor exclusion in
    SMOTE-NC) is exercised by every test.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_pos_contracts):
        cid = f"POS_{i:03d}"
        for yr in range(years_per_contract):
            rows.append({
                "contract_id": cid,
                "department": department_name,
                "gold_y": 1,
                "observation_year": 2020 + yr,
                "fin_score": float(rng.uniform(0.5, 1.0)),
                "esg_score": float(rng.uniform(0.3, 0.9)),
                "supplier_sector": "Pharma",
            })
    for i in range(n_neg_contracts):
        cid = f"NEG_{i:03d}"
        for yr in range(years_per_contract):
            rows.append({
                "contract_id": cid,
                "department": department_name,
                "gold_y": 0,
                "observation_year": 2020 + yr,
                "fin_score": float(rng.uniform(0.0, 0.5)),
                "esg_score": float(rng.uniform(0.1, 0.5)),
                "supplier_sector": "Tech",
            })
    return pd.DataFrame(rows)


def _synth_ids_in(df: pd.DataFrame, contract_id_col: str = "contract_id") -> set:
    """All contract_ids in df that look synthetic."""
    return {
        cid for cid in df[contract_id_col].astype(str).tolist()
        if cid.startswith("SYNTH_")
    }


# ---------------------------------------------------------------------------
# Single-cell tests — one configuration each, asserted against a fixture
# ---------------------------------------------------------------------------
class TestNoSyntheticInQuery:
    """Guard against synthetic leakage into the query set."""

    def test_smote_nc_no_leakage(self):
        dept_df = _make_toy_dept_df()
        episode = sample_episode_with_synthetic_support(
            dept_df=dept_df, feature_cols=FEATURE_COLS,
            department_name="TestDept",
            n_support_pos=2, n_support_neg=2,
            augmentation_method="smote_nc", synthetic_proportion=0.5,
            categorical_cols=["supplier_sector"],
            random_state=42,
        )
        query_df = episode["query_df"]
        support_df = episode["support_df"]

        # No synthetic rows in the query DataFrame.
        assert "is_synthetic" not in query_df.columns or not query_df["is_synthetic"].any(), (
            "Synthetic rows leaked into query_df."
        )

        # No SYNTH_ contract IDs in the query.
        leaked = _synth_ids_in(support_df) & set(query_df["contract_id"].astype(str).tolist())
        assert not leaked, f"Synthetic contract IDs leaked into query: {leaked}"

        # Support DOES contain synthetic rows.
        assert support_df["is_synthetic"].any(), (
            "Expected synthetic rows in support but found none."
        )

    def test_none_method_no_synthetic(self):
        dept_df = _make_toy_dept_df()
        episode = sample_episode_with_synthetic_support(
            dept_df=dept_df, feature_cols=FEATURE_COLS,
            department_name="TestDept",
            n_support_pos=2, n_support_neg=2,
            augmentation_method="none",
            random_state=42,
        )
        info = episode["augmentation_info"]
        assert info["n_synthetic_support_rows"] == 0
        assert info["method"] == "none"
        # No SYNTH_ ids should exist anywhere in the episode.
        assert not _synth_ids_in(episode["support_df"])
        assert not _synth_ids_in(episode["query_df"])

    def test_support_query_contract_disjoint(self):
        """Real support and query contracts must never overlap — with or without augmentation."""
        dept_df = _make_toy_dept_df()
        episode = sample_episode_with_synthetic_support(
            dept_df=dept_df, feature_cols=FEATURE_COLS,
            department_name="TestDept",
            n_support_pos=2, n_support_neg=2,
            augmentation_method="smote_nc", synthetic_proportion=0.75,
            categorical_cols=["supplier_sector"],
            random_state=99,
        )
        support_real_ids = set(
            episode["support_df"]
            .loc[~episode["support_df"]["is_synthetic"], "contract_id"]
            .astype(str)
            .tolist()
        )
        query_ids = set(episode["query_df"]["contract_id"].astype(str).tolist())
        overlap = support_real_ids & query_ids
        assert not overlap, f"Real support contracts overlap with query: {overlap}"

    def test_is_synthetic_mask_aligned(self):
        dept_df = _make_toy_dept_df()
        episode = sample_episode_with_synthetic_support(
            dept_df=dept_df, feature_cols=FEATURE_COLS,
            department_name="TestDept",
            n_support_pos=2, n_support_neg=2,
            augmentation_method="smote_nc", target_per_class=5,
            categorical_cols=["supplier_sector"],
            random_state=42,
        )
        sup_df = episode["support_df"]
        mask = episode["is_synthetic_mask"]
        # Lengths line up
        assert len(mask) == len(episode["X_support"]) == len(sup_df), (
            "is_synthetic_mask must match X_support and support_df row count."
        )
        # Real-then-synthetic ordering inside support
        n_real = int(episode["augmentation_info"]["n_real_support_rows"])
        assert (~mask[:n_real]).all(), "First n_real rows must be real."
        assert mask[n_real:].all(), "Tail rows must be synthetic."
        # Mask agrees with the support_df is_synthetic column
        assert (mask == sup_df["is_synthetic"].to_numpy(dtype=bool)).all()

    def test_query_unchanged_by_augmentation(self):
        """Augmentation must not perturb the query side at all."""
        from master_thesis.episode_sampler import sample_support_query_split

        dept_df = _make_toy_dept_df()
        ep_real = sample_support_query_split(
            dept_df=dept_df, feature_cols=FEATURE_COLS,
            n_support_pos=2, n_support_neg=2, random_state=42,
        )
        ep_aug = sample_episode_with_synthetic_support(
            dept_df=dept_df, feature_cols=FEATURE_COLS,
            n_support_pos=2, n_support_neg=2, random_state=42,
            augmentation_method="smote_nc", synthetic_proportion=0.5,
            categorical_cols=["supplier_sector"],
        )
        assert sorted(ep_real["query_ids"]) == sorted(ep_aug["query_ids"])
        assert ep_real["X_query"].shape == ep_aug["X_query"].shape
        assert (
            ep_real["query_df"]["contract_id"].tolist()
            == ep_aug["query_df"]["contract_id"].tolist()
        )

    def test_only_supported_methods(self):
        """Methods outside ``VALID_AUGMENTATION_METHODS`` must raise."""
        dept_df = _make_toy_dept_df()
        with pytest.raises((ValueError, KeyError)):
            sample_episode_with_synthetic_support(
                dept_df=dept_df, feature_cols=FEATURE_COLS,
                n_support_pos=2, n_support_neg=2, random_state=0,
                augmentation_method="not_a_real_method",
                synthetic_proportion=0.5,
                categorical_cols=["supplier_sector"],
            )


# ---------------------------------------------------------------------------
# Fuzz tests — many seeds × many configurations
# ---------------------------------------------------------------------------
class TestNoSyntheticInQueryFuzz:
    """Stress-test the leakage invariant across many seeds × configurations."""

    @pytest.mark.parametrize("seed", list(range(20)))
    # synthetic_proportion is constrained by the augment_support validator
    # to the half-open interval [0, 1).  Stay strictly inside that range.
    @pytest.mark.parametrize("synthetic_proportion", [0.25, 0.5, 0.75, 0.9])
    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_smote_nc_fuzz_no_leakage(self, seed, synthetic_proportion, k):
        dept_df = _make_toy_dept_df(seed=seed)
        episode = sample_episode_with_synthetic_support(
            dept_df=dept_df, feature_cols=FEATURE_COLS,
            n_support_pos=k, n_support_neg=k,
            augmentation_method="smote_nc",
            synthetic_proportion=synthetic_proportion,
            categorical_cols=["supplier_sector"],
            random_state=seed,
        )
        # 1. No synthetic id reaches the query.
        leaked = _synth_ids_in(episode["query_df"])
        assert not leaked, (
            f"seed={seed} k={k} prop={synthetic_proportion}: leaked {leaked}"
        )
        # 2. Real support contracts and query contracts are disjoint.
        sup = episode["support_df"]
        real_ids = set(sup.loc[~sup["is_synthetic"], "contract_id"].astype(str).tolist())
        qry_ids = set(episode["query_df"]["contract_id"].astype(str).tolist())
        assert not (real_ids & qry_ids), (
            f"seed={seed} k={k} prop={synthetic_proportion}: real/query overlap"
        )
        # 3. Mask alignment.
        assert len(episode["is_synthetic_mask"]) == len(sup)
        # 4. info counts match the dataframe.
        info = episode["augmentation_info"]
        assert info["n_real_support_rows"] == int((~sup["is_synthetic"]).sum())
        assert info["n_synthetic_support_rows"] == int(sup["is_synthetic"].sum())

    @pytest.mark.parametrize("seed", list(range(10)))
    @pytest.mark.parametrize("target_per_class", [3, 5, 8])
    def test_target_per_class_fuzz_no_leakage(self, seed, target_per_class):
        dept_df = _make_toy_dept_df(seed=seed)
        episode = sample_episode_with_synthetic_support(
            dept_df=dept_df, feature_cols=FEATURE_COLS,
            n_support_pos=2, n_support_neg=2,
            augmentation_method="smote_nc",
            target_per_class=target_per_class,
            categorical_cols=["supplier_sector"],
            random_state=seed,
        )
        leaked = _synth_ids_in(episode["query_df"])
        assert not leaked, f"seed={seed} target={target_per_class}: leaked {leaked}"

        sup = episode["support_df"]
        # Each class is filled up to exactly target_per_class rows (or to the
        # real-row count if it already exceeded the target).
        for cls in (0, 1):
            n_cls = int((sup["gold_y"] == cls).sum())
            assert n_cls >= target_per_class or n_cls == int(
                ((~sup["is_synthetic"]) & (sup["gold_y"] == cls)).sum()
            ), f"seed={seed} cls={cls}: got {n_cls} rows for target {target_per_class}"


# ---------------------------------------------------------------------------
# Stand-alone runner — works without pytest
# ---------------------------------------------------------------------------
def _run_standalone() -> int:
    """Run a representative subset without depending on pytest."""
    failures = []
    suite = TestNoSyntheticInQuery()
    for name in (
        "test_smote_nc_no_leakage",
        "test_none_method_no_synthetic",
        "test_support_query_contract_disjoint",
        "test_is_synthetic_mask_aligned",
        "test_query_unchanged_by_augmentation",
    ):
        try:
            getattr(suite, name)()
            print(f"PASS  {name}")
        except AssertionError as e:
            failures.append((name, str(e)))
            print(f"FAIL  {name}: {e}")
        except Exception as e:
            failures.append((name, f"{type(e).__name__}: {e}"))
            print(f"ERROR {name}: {type(e).__name__}: {e}")

    fuzz = TestNoSyntheticInQueryFuzz()
    for seed in range(5):
        for prop in (0.5, 0.9):
            for k in (1, 2):
                name = f"test_smote_nc_fuzz_no_leakage[seed={seed},prop={prop},k={k}]"
                try:
                    fuzz.test_smote_nc_fuzz_no_leakage(seed, prop, k)
                except AssertionError as e:
                    failures.append((name, str(e)))
                    print(f"FAIL  {name}: {e}")
                except Exception as e:
                    failures.append((name, f"{type(e).__name__}: {e}"))
                    print(f"ERROR {name}: {type(e).__name__}: {e}")
    if not failures:
        print()
        print(f"All checks passed. (VALID_AUGMENTATION_METHODS={VALID_AUGMENTATION_METHODS})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_standalone())
