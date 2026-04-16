"""
Tests for episode_sampler.py.

Validates:
- No support/query contract leakage
- Correct support counts (n_support_pos / n_support_neg)
- Empty query set raises ValueError
- Insufficient positive/negative contracts raise ValueError
- Repeated episodes from make_logistics_meta_test_split
- Department task table construction
- Department validity filtering
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from master_thesis.episode_sampler import (
    build_department_task_table,
    filter_valid_departments,
    make_logistics_meta_test_split,
    sample_meta_batch,
    sample_support_query_split,
    summarize_department_tasks,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_dept_df(
    department: str = "Logistics",
    n_pos_contracts: int = 6,
    n_neg_contracts: int = 6,
    rows_per_contract: int = 2,
    n_features: int = 4,
    seed: int = 0,
) -> pd.DataFrame:
    """Create a minimal department dataframe for testing."""
    rng = np.random.default_rng(seed)
    rows = []
    label = 1
    for contract_idx in range(n_pos_contracts + n_neg_contracts):
        if contract_idx >= n_pos_contracts:
            label = 0
        contract_id = f"C{contract_idx:04d}"
        for _ in range(rows_per_contract):
            row = {
                "contract_id": contract_id,
                "department": department,
                "observation_year": 2022,
                "gold_y": label,
            }
            for fi in range(n_features):
                row[f"feat_{fi}"] = rng.standard_normal()
            rows.append(row)
    return pd.DataFrame(rows)


def _make_multi_dept_df(departments=("Logistics", "IT", "Finance")) -> pd.DataFrame:
    frames = [_make_dept_df(dept, seed=i) for i, dept in enumerate(departments)]
    return pd.concat(frames, ignore_index=True)


FEAT_COLS = [f"feat_{i}" for i in range(4)]


# ---------------------------------------------------------------------------
# sample_support_query_split
# ---------------------------------------------------------------------------

class TestSampleSupportQuerySplit:
    def test_no_contract_leakage(self):
        dept_df = _make_dept_df()
        episode = sample_support_query_split(
            dept_df=dept_df,
            feature_cols=FEAT_COLS,
            department_name="Logistics",
            n_support_pos=2,
            n_support_neg=2,
            random_state=42,
        )
        support_ids = set(episode["support_df"]["contract_id"].tolist())
        query_ids = set(episode["query_df"]["contract_id"].tolist())
        intersection = support_ids & query_ids
        assert len(intersection) == 0, f"Contract leakage detected: {intersection}"

    def test_support_pos_count(self):
        dept_df = _make_dept_df(n_pos_contracts=5, n_neg_contracts=5)
        for n_pos in [1, 2, 3]:
            episode = sample_support_query_split(
                dept_df=dept_df,
                feature_cols=FEAT_COLS,
                department_name="Logistics",
                n_support_pos=n_pos,
                n_support_neg=n_pos,
                random_state=0,
            )
            support_pos = (episode["support_df"]["gold_y"] == 1).sum()
            support_neg = (episode["support_df"]["gold_y"] == 0).sum()
            # Support contracts, not rows: count unique positive contracts
            pos_contracts = episode["support_df"].loc[
                episode["support_df"]["gold_y"] == 1, "contract_id"
            ].nunique()
            neg_contracts = episode["support_df"].loc[
                episode["support_df"]["gold_y"] == 0, "contract_id"
            ].nunique()
            assert pos_contracts == n_pos, f"Expected {n_pos} pos contracts, got {pos_contracts}"
            assert neg_contracts == n_pos, f"Expected {n_pos} neg contracts, got {neg_contracts}"

    def test_insufficient_pos_raises(self):
        dept_df = _make_dept_df(n_pos_contracts=1, n_neg_contracts=5)
        with pytest.raises(ValueError, match="positive contracts"):
            sample_support_query_split(
                dept_df=dept_df,
                feature_cols=FEAT_COLS,
                department_name="Logistics",
                n_support_pos=2,
                n_support_neg=2,
                random_state=0,
            )

    def test_insufficient_neg_raises(self):
        dept_df = _make_dept_df(n_pos_contracts=5, n_neg_contracts=1)
        with pytest.raises(ValueError, match="negative contracts"):
            sample_support_query_split(
                dept_df=dept_df,
                feature_cols=FEAT_COLS,
                department_name="Logistics",
                n_support_pos=2,
                n_support_neg=2,
                random_state=0,
            )

    def test_empty_query_raises(self):
        # Only 1 pos + 1 neg contract total: support consumes all
        dept_df = _make_dept_df(n_pos_contracts=1, n_neg_contracts=1, rows_per_contract=1)
        with pytest.raises(ValueError):
            sample_support_query_split(
                dept_df=dept_df,
                feature_cols=FEAT_COLS,
                department_name="Logistics",
                n_support_pos=1,
                n_support_neg=1,
                random_state=0,
            )

    def test_query_is_remainder(self):
        dept_df = _make_dept_df(n_pos_contracts=4, n_neg_contracts=4)
        episode = sample_support_query_split(
            dept_df=dept_df,
            feature_cols=FEAT_COLS,
            department_name="Logistics",
            n_support_pos=2,
            n_support_neg=2,
            random_state=1,
        )
        total_ids = dept_df["contract_id"].nunique()
        support_ids = set(episode["support_df"]["contract_id"].unique())
        query_ids = set(episode["query_df"]["contract_id"].unique())
        # Support uses 4 contracts, query gets remaining 4
        assert len(support_ids) == 4
        assert len(query_ids) == total_ids - 4
        assert support_ids.isdisjoint(query_ids)


# ---------------------------------------------------------------------------
# make_logistics_meta_test_split
# ---------------------------------------------------------------------------

class TestMakeLogisticsMetaTestSplit:
    def test_returns_correct_n_repeats(self):
        task_df = _make_dept_df(n_pos_contracts=5, n_neg_contracts=5)
        episodes = make_logistics_meta_test_split(
            task_df=task_df,
            feature_cols=FEAT_COLS,
            target_department="Logistics",
            n_support_pos=2,
            n_support_neg=2,
            n_repeats=5,
        )
        assert len(episodes) == 5

    def test_no_leakage_across_repeats(self):
        task_df = _make_dept_df(n_pos_contracts=5, n_neg_contracts=5)
        episodes = make_logistics_meta_test_split(
            task_df=task_df,
            feature_cols=FEAT_COLS,
            target_department="Logistics",
            n_support_pos=2,
            n_support_neg=2,
            n_repeats=3,
        )
        for ep in episodes:
            s_ids = set(ep["support_df"]["contract_id"].tolist())
            q_ids = set(ep["query_df"]["contract_id"].tolist())
            assert s_ids.isdisjoint(q_ids), "Leakage within episode"

    def test_invalid_department_raises(self):
        task_df = _make_dept_df(department="Logistics")
        with pytest.raises(ValueError):
            make_logistics_meta_test_split(
                task_df=task_df,
                feature_cols=FEAT_COLS,
                target_department="NonExistentDept",
                n_support_pos=2,
                n_support_neg=2,
                n_repeats=3,
            )


# ---------------------------------------------------------------------------
# build_department_task_table
# ---------------------------------------------------------------------------

class TestBuildDepartmentTaskTable:
    def test_drops_missing_gold_labels(self):
        df = _make_multi_dept_df()
        # Set some rows' gold_y to NaN
        df.loc[:5, "gold_y"] = np.nan
        task_df = build_department_task_table(df, feature_cols=FEAT_COLS)
        assert task_df["gold_y"].notna().all()

    def test_drops_missing_department(self):
        df = _make_multi_dept_df()
        df.loc[:5, "department"] = np.nan
        task_df = build_department_task_table(df, feature_cols=FEAT_COLS)
        assert task_df["department"].notna().all()

    def test_binary_target(self):
        df = _make_multi_dept_df()
        task_df = build_department_task_table(df, feature_cols=FEAT_COLS)
        assert set(task_df["gold_y"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# filter_valid_departments
# ---------------------------------------------------------------------------

class TestFilterValidDepartments:
    def test_filters_departments_with_insufficient_pos(self):
        # Dept A: 5 pos / 5 neg — valid
        # Dept B: 1 pos / 5 neg — invalid for n_support_pos=2
        df_a = _make_dept_df("DeptA", n_pos_contracts=5, n_neg_contracts=5)
        df_b = _make_dept_df("DeptB", n_pos_contracts=1, n_neg_contracts=5)
        df = pd.concat([df_a, df_b], ignore_index=True)

        filtered, valid_depts, validity = filter_valid_departments(
            task_df=df,
            n_support_pos=2,
            n_support_neg=2,
            min_query_contracts=1,
        )
        assert "DeptA" in valid_depts
        assert "DeptB" not in valid_depts

    def test_validity_df_has_expected_columns(self):
        df = _make_multi_dept_df()
        _, _, validity_df = filter_valid_departments(
            task_df=df,
            n_support_pos=2,
            n_support_neg=2,
        )
        assert "valid_department" in validity_df.columns
        assert "validity_reason" in validity_df.columns


# ---------------------------------------------------------------------------
# sample_meta_batch
# ---------------------------------------------------------------------------

class TestSampleMetaBatch:
    def test_returns_correct_batch_size(self):
        df = _make_multi_dept_df(departments=("A", "B", "C", "D", "E"))
        valid_depts = ["A", "B", "C", "D", "E"]
        batch = sample_meta_batch(
            task_df=df,
            feature_cols=FEAT_COLS,
            valid_departments=valid_depts,
            meta_batch_size=3,
            n_support_pos=2,
            n_support_neg=2,
            random_state=42,
        )
        assert len(batch) == 3

    def test_no_leakage_within_batch_episodes(self):
        df = _make_multi_dept_df(departments=("A", "B", "C"))
        valid_depts = ["A", "B", "C"]
        batch = sample_meta_batch(
            task_df=df,
            feature_cols=FEAT_COLS,
            valid_departments=valid_depts,
            meta_batch_size=2,
            n_support_pos=2,
            n_support_neg=2,
            random_state=0,
        )
        for ep in batch:
            s_ids = set(ep["support_df"]["contract_id"].tolist())
            q_ids = set(ep["query_df"]["contract_id"].tolist())
            assert s_ids.isdisjoint(q_ids)
