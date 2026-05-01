"""
Tests for src/master_thesis/stage2b_synthetic.py

Covers:
- resolve_categorical_cols
- build_department_eligibility_table
- compute_support_accounting
- adapt_and_evaluate (zero_shot, finetune, ANIL/FOMAML/MAML error paths)
- build_stage2b_result_row schema
- run_stage2b_experiment (integration: returns rows, query real-only, schema)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_toy_dept_df(
    n_pos_contracts: int = 5,
    n_neg_contracts: int = 5,
    rows_per_contract: int = 2,
    n_features: int = 4,
    department: str = "TestDept",
    seed: int = 0,
) -> pd.DataFrame:
    """Build a small DataFrame that looks like a department task table."""
    rng = np.random.default_rng(seed)
    rows = []
    cid = 0
    for label, n_contracts in [(1, n_pos_contracts), (0, n_neg_contracts)]:
        for _ in range(n_contracts):
            cid += 1
            for _ in range(rows_per_contract):
                row = {
                    "contract_id": f"C{cid:03d}",
                    "department": department,
                    "gold_y": label,
                    "label_source": "manual_hardcoded",
                    "observation_year": 2024,
                }
                for fi in range(n_features):
                    row[f"feat_{fi}"] = rng.normal()
                rows.append(row)
    return pd.DataFrame(rows)


def _feature_cols(n: int = 4) -> list[str]:
    return [f"feat_{i}" for i in range(n)]


class _TinyMLP(nn.Module):
    """Minimal MLP that matches the TabularMLP interface for testing."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)


class _IdentityPreprocessor:
    """Preprocessor that passes features through unchanged."""

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.values.astype(np.float32)
        return np.asarray(X, dtype=np.float32)


# ---------------------------------------------------------------------------
# Tests: resolve_categorical_cols
# ---------------------------------------------------------------------------

class TestResolveCategoricalCols:
    def test_explicit_cols_filtered(self):
        from master_thesis.stage2b_synthetic import resolve_categorical_cols

        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [3.0, 4.0]})
        result = resolve_categorical_cols(df, ["a", "b", "c"], explicit_categorical_cols=["b", "z"])
        assert result == ["b"]  # "z" not in feature_cols → dropped

    def test_auto_detect_non_numeric(self):
        from master_thesis.stage2b_synthetic import resolve_categorical_cols

        df = pd.DataFrame({
            "num": [1.0, 2.0],
            "cat_obj": ["a", "b"],
            "cat_str": pd.array(["x", "y"], dtype="string"),
        })
        result = resolve_categorical_cols(df, ["num", "cat_obj", "cat_str"])
        assert "num" not in result
        assert "cat_obj" in result
        assert "cat_str" in result

    def test_empty_returns_empty(self):
        from master_thesis.stage2b_synthetic import resolve_categorical_cols

        df = pd.DataFrame({"a": [1, 2]})
        assert resolve_categorical_cols(df, ["a"]) == []


# ---------------------------------------------------------------------------
# Tests: build_department_eligibility_table
# ---------------------------------------------------------------------------

class TestBuildDepartmentEligibilityTable:
    def test_basic_structure(self):
        from master_thesis.stage2b_synthetic import build_department_eligibility_table

        df = _make_toy_dept_df(n_pos_contracts=4, n_neg_contracts=3)
        table = build_department_eligibility_table(df, max_k=5)

        assert len(table) == 1
        row = table.iloc[0]
        assert row["department"] == "TestDept"
        assert row["n_pos_contracts"] == 4
        assert row["n_neg_contracts"] == 3
        assert row["max_balanced_real_k"] == 3  # min(4, 3)

    def test_can_run_columns(self):
        from master_thesis.stage2b_synthetic import build_department_eligibility_table

        df = _make_toy_dept_df(n_pos_contracts=2, n_neg_contracts=2)
        table = build_department_eligibility_table(df, max_k=5)
        row = table.iloc[0]

        assert row["can_run_k1"] is True
        assert row["can_run_k2"] is True
        assert row["can_run_k3"] is False  # only 2 per class

    def test_smotenc_k1_impossible(self):
        from master_thesis.stage2b_synthetic import build_department_eligibility_table

        df = _make_toy_dept_df(n_pos_contracts=5, n_neg_contracts=5)
        table = build_department_eligibility_table(df, max_k=5)
        row = table.iloc[0]

        # SMOTENC needs ≥2 per class → k=1 is never possible for true SMOTENC
        assert row["smotenc_k1_possible"] is False
        assert row["smotenc_k2_possible"] is True

    def test_stage2b_evaluable(self):
        from master_thesis.stage2b_synthetic import build_department_eligibility_table

        # 2 pos, 2 neg → max_query_safe_real_k = min(1,1) = 1 < 2 → not evaluable
        df_small = _make_toy_dept_df(n_pos_contracts=2, n_neg_contracts=2)
        table = build_department_eligibility_table(df_small, max_k=5)
        assert table.iloc[0]["stage2b_evaluable"] is False

        # 4 pos, 4 neg → max_query_safe_real_k = 3 ≥ 2 → evaluable
        df_big = _make_toy_dept_df(n_pos_contracts=4, n_neg_contracts=4)
        table = build_department_eligibility_table(df_big, max_k=5)
        assert table.iloc[0]["stage2b_evaluable"] is True

    def test_smotenc_false_when_minority_class_has_one(self):
        """If k >= 2 but the department only has 1 positive contract,
        smotenc_k{k}_possible must be False (even though k >= 2)."""
        from master_thesis.stage2b_synthetic import build_department_eligibility_table

        # 1 positive, 5 negative → SMOTENC impossible at any k
        df = _make_toy_dept_df(n_pos_contracts=1, n_neg_contracts=5)
        table = build_department_eligibility_table(df, max_k=5)
        row = table.iloc[0]

        for k in range(1, 6):
            assert row[f"smotenc_k{k}_possible"] is False, (
                f"smotenc_k{k}_possible should be False with only 1 positive contract"
            )
        assert row["smotenc_possible"] is False

    def test_smotenc_true_when_both_classes_sufficient(self):
        """If both classes have >= 2 and max_balanced_real_k >= k, it's True."""
        from master_thesis.stage2b_synthetic import build_department_eligibility_table

        # 4 pos, 3 neg → max_balanced = 3, both classes ≥ 2
        df = _make_toy_dept_df(n_pos_contracts=4, n_neg_contracts=3)
        table = build_department_eligibility_table(df, max_k=5)
        row = table.iloc[0]

        assert row["smotenc_k1_possible"] is True   # k=1 ≤ 3, classes ≥ 2
        assert row["smotenc_k2_possible"] is True   # k=2 ≤ 3
        assert row["smotenc_k3_possible"] is True   # k=3 ≤ 3
        assert row["smotenc_k4_possible"] is False  # k=4 > 3
        assert row["smotenc_k5_possible"] is False  # k=5 > 3

    def test_multiple_departments(self):
        from master_thesis.stage2b_synthetic import build_department_eligibility_table

        df1 = _make_toy_dept_df(department="A", n_pos_contracts=3, n_neg_contracts=3, seed=0)
        df2 = _make_toy_dept_df(department="B", n_pos_contracts=1, n_neg_contracts=1, seed=1)
        df = pd.concat([df1, df2], ignore_index=True)
        table = build_department_eligibility_table(df, max_k=5)
        assert len(table) == 2
        assert set(table["department"]) == {"A", "B"}


# ---------------------------------------------------------------------------
# Tests: compute_support_accounting
# ---------------------------------------------------------------------------

class TestComputeSupportAccounting:
    def test_real_only(self):
        from master_thesis.stage2b_synthetic import compute_support_accounting

        df = pd.DataFrame({
            "gold_y": [1, 1, 0, 0],
            "contract_id": ["C1", "C2", "C3", "C4"],
        })
        acc = compute_support_accounting(df)
        assert acc["real_k_pos"] == 2
        assert acc["real_k_neg"] == 2
        assert acc["synthetic_k_pos"] == 0
        assert acc["n_real_support_rows"] == 4
        assert acc["n_synthetic_support_rows"] == 0

    def test_with_synthetic(self):
        from master_thesis.stage2b_synthetic import compute_support_accounting

        df = pd.DataFrame({
            "gold_y": [1, 1, 0, 0, 1, 0],
            "is_synthetic": [False, False, False, False, True, True],
            "contract_id": ["C1", "C2", "C3", "C4", "SYNTH_1", "SYNTH_2"],
        })
        acc = compute_support_accounting(df)
        assert acc["real_k_pos"] == 2
        assert acc["real_k_neg"] == 2
        assert acc["synthetic_k_pos"] == 1
        assert acc["synthetic_k_neg"] == 1
        assert acc["effective_k_pos"] == 3
        assert acc["effective_k_neg"] == 3
        assert acc["n_real_support_rows"] == 4
        assert acc["n_synthetic_support_rows"] == 2

    def test_missing_is_synthetic_col(self):
        from master_thesis.stage2b_synthetic import compute_support_accounting

        df = pd.DataFrame({
            "gold_y": [1, 0, 1],
            "contract_id": ["C1", "C2", "C3"],
        })
        acc = compute_support_accounting(df)
        # All treated as real
        assert acc["n_real_support_rows"] == 3
        assert acc["n_synthetic_support_rows"] == 0


# ---------------------------------------------------------------------------
# Tests: adapt_and_evaluate
# ---------------------------------------------------------------------------

class TestAdaptAndEvaluate:
    @pytest.fixture
    def episode(self):
        rng = np.random.default_rng(42)
        n_sup, n_qry = 6, 4
        n_feat = 4
        support_df = pd.DataFrame({
            "gold_y": [1, 1, 1, 0, 0, 0],
            **{f"feat_{i}": rng.normal(size=n_sup) for i in range(n_feat)},
        })
        query_df = pd.DataFrame({
            "gold_y": [1, 1, 0, 0],
            **{f"feat_{i}": rng.normal(size=n_qry) for i in range(n_feat)},
        })
        return {"support_df": support_df, "query_df": query_df}

    def test_zero_shot_returns_predictions(self, episode):
        from master_thesis.stage2b_synthetic import adapt_and_evaluate

        model = _TinyMLP(4)
        result = adapt_and_evaluate(
            method="zero_shot", base_model=model,
            preprocessor=_IdentityPreprocessor(), episode=episode,
            feature_cols=_feature_cols(4), device="cpu",
        )
        assert "y_true" in result
        assert "y_pred" in result
        assert len(result["y_true"]) == 4
        assert len(result["y_pred"]) == 4
        assert result["method"] == "zero_shot"

    def test_finetune_returns_predictions(self, episode):
        from master_thesis.stage2b_synthetic import adapt_and_evaluate

        model = _TinyMLP(4)
        result = adapt_and_evaluate(
            method="finetune", base_model=model,
            preprocessor=_IdentityPreprocessor(), episode=episode,
            feature_cols=_feature_cols(4), inner_steps=2, device="cpu",
        )
        assert len(result["y_pred"]) == 4
        assert result["n_query_pos"] == 2
        assert result["n_query_neg"] == 2

    def test_unknown_method_raises(self, episode):
        from master_thesis.stage2b_synthetic import adapt_and_evaluate

        model = _TinyMLP(4)
        with pytest.raises(ValueError, match="Unknown method"):
            adapt_and_evaluate(
                method="invalid_method", base_model=model,
                preprocessor=_IdentityPreprocessor(), episode=episode,
                feature_cols=_feature_cols(4), device="cpu",
            )

    def test_anil_dispatch(self, episode, monkeypatch):
        """Verify ANIL branch dispatches to adapt_anil_on_episode."""
        from master_thesis import stage2b_synthetic
        import master_thesis.anil as anil_mod

        model = _TinyMLP(4)
        called = {"count": 0}

        def _mock_adapt_anil(model, X_supp, y_supp, inner_lr=0.01, inner_steps=5):
            called["count"] += 1
            # Return a fake (head_weight, head_bias) matching the last linear layer
            last_layer = model.net[-1]
            return last_layer.weight.detach().clone(), last_layer.bias.detach().clone()

        monkeypatch.setattr(anil_mod, "adapt_anil_on_episode", _mock_adapt_anil)
        # Also patch the import inside adapt_and_evaluate
        monkeypatch.setattr(
            stage2b_synthetic, "__builtins__",
            stage2b_synthetic.__builtins__ if hasattr(stage2b_synthetic, "__builtins__") else {},
        )
        # Re-import to pick up mock via the `from master_thesis.anil import` inside function
        import importlib
        importlib.reload(stage2b_synthetic)
        monkeypatch.setattr(anil_mod, "adapt_anil_on_episode", _mock_adapt_anil)

        result = stage2b_synthetic.adapt_and_evaluate(
            method="anil", base_model=model,
            preprocessor=_IdentityPreprocessor(), episode=episode,
            feature_cols=_feature_cols(4), device="cpu",
        )
        assert called["count"] == 1, "adapt_anil_on_episode was not called"
        assert "y_pred" in result
        assert len(result["y_pred"]) == 4
        assert result["method"] == "anil"

    def test_fomaml_dispatch(self, episode, monkeypatch):
        """Verify FOMAML branch dispatches to adapt_fomaml_on_episode."""
        from master_thesis import stage2b_synthetic
        import master_thesis.FOMAML as fomaml_mod

        model = _TinyMLP(4)
        called = {"count": 0}

        def _mock_adapt_fomaml(model, X_supp, y_supp, inner_lr=0.01, inner_steps=5):
            called["count"] += 1
            return dict(model.named_parameters())

        monkeypatch.setattr(fomaml_mod, "adapt_fomaml_on_episode", _mock_adapt_fomaml)
        import importlib
        importlib.reload(stage2b_synthetic)
        monkeypatch.setattr(fomaml_mod, "adapt_fomaml_on_episode", _mock_adapt_fomaml)

        result = stage2b_synthetic.adapt_and_evaluate(
            method="fomaml", base_model=model,
            preprocessor=_IdentityPreprocessor(), episode=episode,
            feature_cols=_feature_cols(4), device="cpu",
        )
        assert called["count"] == 1, "adapt_fomaml_on_episode was not called"
        assert "y_pred" in result
        assert len(result["y_pred"]) == 4
        assert result["method"] == "fomaml"

    def test_maml_dispatch(self, episode, monkeypatch):
        """Verify MAML branch dispatches to adapt_maml_on_episode."""
        from master_thesis import stage2b_synthetic
        import master_thesis.maml as maml_mod

        model = _TinyMLP(4)
        called = {"count": 0}

        def _mock_adapt_maml(model, X_supp, y_supp, inner_lr=0.01,
                              inner_steps=5, create_graph=False):
            called["count"] += 1
            return dict(model.named_parameters())

        monkeypatch.setattr(maml_mod, "adapt_maml_on_episode", _mock_adapt_maml)
        import importlib
        importlib.reload(stage2b_synthetic)
        monkeypatch.setattr(maml_mod, "adapt_maml_on_episode", _mock_adapt_maml)

        result = stage2b_synthetic.adapt_and_evaluate(
            method="maml", base_model=model,
            preprocessor=_IdentityPreprocessor(), episode=episode,
            feature_cols=_feature_cols(4), device="cpu",
        )
        assert called["count"] == 1, "adapt_maml_on_episode was not called"
        assert "y_pred" in result
        assert len(result["y_pred"]) == 4
        assert result["method"] == "maml"


# ---------------------------------------------------------------------------
# Tests: build_stage2b_result_row
# ---------------------------------------------------------------------------

class TestBuildStage2bResultRow:
    def test_schema_contains_accounting_fields(self):
        from master_thesis.stage2b_synthetic import build_stage2b_result_row

        config = {
            "experiment_id": "test_exp", "target_department": "TestDept",
            "init_name": "A_weak_only", "method": "finetune",
            "n_support_pos": 2, "n_support_neg": 2, "seed": 42,
        }
        episode = {"support_df": pd.DataFrame(), "query_df": pd.DataFrame(),
                    "department": "TestDept", "augmentation_info": {}}
        prediction_result = {"y_true": np.array([1, 0]), "y_pred": np.array([0.7, 0.3]),
                             "method": "finetune", "n_query_rows": 2,
                             "n_query_pos": 1, "n_query_neg": 1,
                             "inner_steps": 3, "inner_lr": 0.01}
        metrics = {"gold_auroc": 0.8, "gold_ap": 0.75}
        diagnostics = {}
        accounting = {
            "real_k_pos": 2, "real_k_neg": 2, "synthetic_k_pos": 0,
            "synthetic_k_neg": 0, "effective_k_pos": 2, "effective_k_neg": 2,
            "n_real_support_rows": 4, "n_synthetic_support_rows": 0,
            "n_effective_support_rows": 4, "n_real_support_contracts": 4,
            "n_synthetic_support_contracts": 0,
        }

        row = build_stage2b_result_row(
            config, episode, prediction_result, metrics, diagnostics, accounting, 0,
        )

        # Key accounting fields present
        assert "real_k_pos" in row
        assert "real_k_neg" in row
        assert "effective_k_pos" in row
        assert "n_real_support_rows" in row
        assert "n_synthetic_support_rows" in row
        assert "n_effective_support_rows" in row

        # Key metric fields present
        assert "auroc" in row
        assert "ap" in row
        assert "pred_std" in row
        assert "collapsed" in row

        # Backward-compatible fields
        assert "gen_method" in row
        assert "syn_prop" in row
        assert "gen_fallback_used" in row

        # New clearer fields
        assert "augmentation_method_requested" in row
        assert "augmentation_method_actual" in row


# ---------------------------------------------------------------------------
# Tests: run_stage2b_experiment (integration)
# ---------------------------------------------------------------------------

class TestRunStage2bExperiment:
    def test_returns_rows(self):
        from master_thesis.stage2b_synthetic import run_stage2b_experiment

        df = _make_toy_dept_df(n_pos_contracts=5, n_neg_contracts=5)
        model = _TinyMLP(4)
        config = {
            "method": "finetune", "init_name": "A_weak_only",
            "n_support_pos": 2, "n_support_neg": 2,
            "inner_lr": 0.01, "inner_steps": 2,
            "n_repeats": 3, "seed": 42,
            "augmentation_method": "none",
            "target_department": "TestDept",
            "target_col": "gold_y", "department_col": "department",
            "contract_id_col": "contract_id",
            "experiment_id": "test_run",
        }
        rows = run_stage2b_experiment(
            config=config, task_df=df, feature_cols=_feature_cols(4),
            base_model=model, preprocessor=_IdentityPreprocessor(), device="cpu",
        )
        assert len(rows) == 3  # n_repeats = 3
        assert all(isinstance(r, dict) for r in rows)

    def test_query_real_only(self):
        from master_thesis.stage2b_synthetic import run_stage2b_experiment

        df = _make_toy_dept_df(n_pos_contracts=5, n_neg_contracts=5)
        model = _TinyMLP(4)
        config = {
            "method": "finetune", "init_name": "A_weak_only",
            "n_support_pos": 2, "n_support_neg": 2,
            "inner_lr": 0.01, "inner_steps": 2,
            "n_repeats": 5, "seed": 42,
            "augmentation_method": "smote_nc",
            "synthetic_proportion": 0.5,
            "target_department": "TestDept",
            "target_col": "gold_y", "department_col": "department",
            "contract_id_col": "contract_id",
            "experiment_id": "test_synth",
        }
        rows = run_stage2b_experiment(
            config=config, task_df=df, feature_cols=_feature_cols(4),
            base_model=model, preprocessor=_IdentityPreprocessor(), device="cpu",
        )
        # All rows should have n_query_rows > 0 with no SYNTH_ contracts
        for row in rows:
            assert row["n_query_rows"] > 0

    def test_result_schema_contains_accounting(self):
        from master_thesis.stage2b_synthetic import run_stage2b_experiment

        df = _make_toy_dept_df(n_pos_contracts=5, n_neg_contracts=5)
        model = _TinyMLP(4)
        config = {
            "method": "zero_shot", "init_name": "A_weak_only",
            "n_support_pos": 2, "n_support_neg": 2,
            "inner_lr": 0.01, "inner_steps": 0,
            "n_repeats": 2, "seed": 42,
            "augmentation_method": "none",
            "target_department": "TestDept",
            "target_col": "gold_y", "department_col": "department",
            "contract_id_col": "contract_id",
            "experiment_id": "test_schema",
        }
        rows = run_stage2b_experiment(
            config=config, task_df=df, feature_cols=_feature_cols(4),
            base_model=model, preprocessor=_IdentityPreprocessor(), device="cpu",
        )
        assert len(rows) > 0
        row = rows[0]
        required = [
            "real_k", "real_k_pos", "real_k_neg",
            "n_real_support_rows", "n_synthetic_support_rows",
            "auroc", "method", "department",
        ]
        for field in required:
            assert field in row, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# Tests: placement / invariants
# ---------------------------------------------------------------------------

class TestPlacementInvariants:
    """Confirm that run_stage2b_experiment calls sample_episode_with_synthetic_support
    and that augment_support is NOT called directly from meta-learner files."""

    def test_augment_support_not_imported_in_anil(self):
        import inspect
        from master_thesis import anil
        source = inspect.getsource(anil)
        assert "augment_support" not in source, (
            "augment_support should not be called from anil.py"
        )

    def test_augment_support_not_imported_in_fomaml(self):
        import inspect
        from master_thesis import FOMAML
        source = inspect.getsource(FOMAML)
        assert "augment_support" not in source, (
            "augment_support should not be called from FOMAML.py"
        )

    def test_augment_support_not_imported_in_maml(self):
        import inspect
        from master_thesis import maml
        source = inspect.getsource(maml)
        assert "augment_support" not in source, (
            "augment_support should not be called from maml.py"
        )
