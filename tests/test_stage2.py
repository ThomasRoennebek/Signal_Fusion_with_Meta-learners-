"""
Tests for Stage 2 backend and CLI runner.

Validates:
- resolve_experiment_grid output size and unique IDs
- build_experiment_preview_df structure
- run_experiments (preview-only / dry-run mode from run_stage2.py)
- save_stage2_result artifact schema consistency
- generate_experiment_id stability
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import pytest
import pandas as pd
import tempfile

# Ensure src/ is importable when running tests from repo root.
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from master_thesis.stage2 import (
    build_experiment_preview_df,
    generate_experiment_id,
    load_stage2_config,
    normalize_stage2_config,
    resolve_experiment_grid,
    save_stage2_result,
    summarize_stage2_result,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent / "experiments" / "stage2_config.yaml"


def _minimal_config(
    method: str = "finetune",
    n_support_pos: int = 2,
    n_support_neg: int = 2,
    inner_steps: int = 3,
    init_name: str = "A_weak_only",
    target_department: str = "Logistics",
) -> dict:
    return {
        "seed": 42,
        "verbose": False,
        "method": method,
        "support_config": {"n_support_pos": n_support_pos, "n_support_neg": n_support_neg},
        "data": {
            "data_filename": "contract_with_features_labeled_with_gold.csv",
            "target_col": "gold_y",
            "group_col": "contract_id",
            "department_col": "department",
            "observation_year_col": "observation_year",
        },
        "features": {"source": "stage1_preprocessor"},
        "stage1_init": {
            "init_name": init_name,
            "model_filename": "mlp_pretrained.pt",
            "preprocessor_filename": "mlp_pretrained_preprocessor.joblib",
        },
        "task_config": {
            "target_department": target_department,
            "require_both_query_classes": False,
            "min_query_contracts": 1,
            "exclude_departments": [],
        },
        "meta_config": {
            "meta_batch_size": 2,
            "meta_iterations": 5,
            "inner_lr": 0.01,
            "outer_lr": 0.001,
            "inner_steps": inner_steps,
            "target_inner_steps": inner_steps,
            "n_repeats": 2,
        },
        "mlp_config": {"hidden_dim_1": 256, "hidden_dim_2": 128, "dropout": 0.3},
        "evaluation": {"ranking_k": 10, "calibration_method": None},
        "output": {"save_predictions": True, "save_history": True, "skip_if_exists": True, "force_rerun": False},
    }


# ---------------------------------------------------------------------------
# generate_experiment_id
# ---------------------------------------------------------------------------

class TestGenerateExperimentId:
    def test_deterministic(self):
        exp_id_a = generate_experiment_id("A_weak_only", "anil", 2, 2, 3, "Logistics")
        exp_id_b = generate_experiment_id("A_weak_only", "anil", 2, 2, 3, "Logistics")
        assert exp_id_a == exp_id_b

    def test_different_methods_differ(self):
        id_anil = generate_experiment_id("A_weak_only", "anil", 2, 2, 3, "Logistics")
        id_ft = generate_experiment_id("A_weak_only", "finetune", 2, 2, 3, "Logistics")
        assert id_anil != id_ft

    def test_contains_key_components(self):
        exp_id = generate_experiment_id("A_weak_only", "anil", 2, 3, 5, "Logistics")
        assert "A_weak_only" in exp_id
        assert "anil" in exp_id
        assert "kpos-2" in exp_id
        assert "kneg-3" in exp_id
        assert "steps-5" in exp_id
        assert "Logistics" in exp_id

    def test_spaces_replaced_in_department(self):
        exp_id = generate_experiment_id("A_weak_only", "anil", 2, 2, 3, "Supply Chain")
        assert " " not in exp_id


# ---------------------------------------------------------------------------
# resolve_experiment_grid
# ---------------------------------------------------------------------------

class TestResolveExperimentGrid:
    @pytest.fixture
    def full_config(self):
        if not _CONFIG_PATH.exists():
            pytest.skip("stage2_config.yaml not found")
        return load_stage2_config(_CONFIG_PATH)

    def test_quick_debug_returns_expected_count(self, full_config):
        # quick_debug: 2 methods x 1 support x 1 inner_steps x 1 init = 2
        experiment_list = resolve_experiment_grid(
            full_config=full_config,
            preset_name="quick_debug",
        )
        assert len(experiment_list) == 2

    def test_all_experiment_ids_unique(self, full_config):
        experiment_list = resolve_experiment_grid(
            full_config=full_config,
            preset_name="thesis_main",
        )
        ids = [exp_id for exp_id, _ in experiment_list]
        assert len(ids) == len(set(ids)), "Duplicate experiment IDs detected"

    def test_overrides_are_applied(self, full_config):
        experiment_list = resolve_experiment_grid(
            full_config=full_config,
            preset_name="quick_debug",
            overrides={"methods": ["maml"]},
        )
        methods = [cfg["method"] for _, cfg in experiment_list]
        assert all(m == "maml" for m in methods)

    def test_returns_tuples(self, full_config):
        experiment_list = resolve_experiment_grid(
            full_config=full_config,
            preset_name="quick_debug",
        )
        for item in experiment_list:
            assert isinstance(item, tuple)
            assert len(item) == 2
            exp_id, cfg = item
            assert isinstance(exp_id, str)
            assert isinstance(cfg, dict)

    def test_invalid_preset_raises(self, full_config):
        with pytest.raises(ValueError, match="Unknown Stage 2 preset"):
            resolve_experiment_grid(
                full_config=full_config,
                preset_name="nonexistent_preset",
            )


# ---------------------------------------------------------------------------
# build_experiment_preview_df
# ---------------------------------------------------------------------------

class TestBuildExperimentPreviewDf:
    def test_has_required_columns(self):
        cfg = _minimal_config()
        exp_id = generate_experiment_id("A_weak_only", "anil", 2, 2, 3, "Logistics")
        preview = build_experiment_preview_df([(exp_id, cfg)])
        required_cols = {"experiment_id", "method", "n_support_pos", "n_support_neg",
                         "inner_steps", "init_name", "target_department"}
        assert required_cols.issubset(set(preview.columns))

    def test_row_count_matches_list(self):
        configs = [
            (_minimal_config(method=m), m)
            for m in ["finetune", "anil", "fomaml"]
        ]
        experiment_list = [
            (generate_experiment_id("A_weak_only", m, 2, 2, 3, "Logistics"), cfg)
            for cfg, m in configs
        ]
        preview = build_experiment_preview_df(experiment_list)
        assert len(preview) == 3

    def test_empty_list_returns_empty_df(self):
        preview = build_experiment_preview_df([])
        assert isinstance(preview, pd.DataFrame)
        assert len(preview) == 0


# ---------------------------------------------------------------------------
# normalize_stage2_config
# ---------------------------------------------------------------------------

class TestNormalizeStage2Config:
    def test_unwraps_base_config(self):
        wrapped = {"base_config": _minimal_config(), "presets": {}}
        normalized = normalize_stage2_config(wrapped)
        assert "method" in normalized
        assert "base_config" not in normalized

    def test_flat_config_passthrough(self):
        flat = _minimal_config()
        normalized = normalize_stage2_config(flat)
        assert normalized["method"] == flat["method"]


# ---------------------------------------------------------------------------
# save_stage2_result and summarize_stage2_result
# ---------------------------------------------------------------------------

class TestSaveStage2Result:
    def test_saves_all_expected_artifacts(self, tmp_path):
        config = _minimal_config()
        result = {
            "method": "finetune",
            "status": "success",
            "target_department": "Logistics",
            "raw_metrics": pd.DataFrame({"gold_auroc": [0.75, 0.80], "episode_idx": [0, 1]}),
            "predictions": pd.DataFrame({"y_true": [1, 0], "y_prob": [0.7, 0.3], "episode_idx": [0, 0]}),
            "history": None,
        }
        exp_id = "test_exp_001"
        save_stage2_result(result, config, tmp_path, experiment_id=exp_id)

        assert (tmp_path / "resolved_config.yaml").exists()
        assert (tmp_path / "metrics.csv").exists()
        assert (tmp_path / "predictions.csv").exists()
        assert (tmp_path / "stage2_result_summary.json").exists()
        assert (tmp_path / "stage2_result.json").exists()

    def test_summary_json_has_expected_keys(self, tmp_path):
        config = _minimal_config()
        result = {
            "method": "finetune",
            "status": "success",
            "target_department": "Logistics",
            "raw_metrics": pd.DataFrame({"gold_auroc": [0.75], "episode_idx": [0]}),
            "predictions": pd.DataFrame({"y_true": [1], "y_prob": [0.7], "episode_idx": [0]}),
            "history": None,
        }
        save_stage2_result(result, config, tmp_path, experiment_id="test_exp_002")

        with open(tmp_path / "stage2_result_summary.json") as f:
            summary = json.load(f)

        expected_keys = {"experiment_id", "method", "init_name", "target_department",
                         "n_support_pos", "n_support_neg", "inner_steps", "status",
                         "n_metric_rows", "n_predictions"}
        assert expected_keys.issubset(set(summary.keys()))

    def test_missing_dataframes_do_not_crash(self, tmp_path):
        config = _minimal_config()
        result = {
            "method": "finetune",
            "status": "no_target_episodes_available",
            "target_department": "Logistics",
            "raw_metrics": None,
            "predictions": None,
            "history": None,
        }
        save_stage2_result(result, config, tmp_path, experiment_id="test_exp_003")
        assert (tmp_path / "resolved_config.yaml").exists()
        assert (tmp_path / "stage2_result_summary.json").exists()
        # metrics.csv / predictions.csv should NOT be written when None
        assert not (tmp_path / "metrics.csv").exists()
        assert not (tmp_path / "predictions.csv").exists()


# ---------------------------------------------------------------------------
# CLI runner import (preview-only mode)
# ---------------------------------------------------------------------------

class TestRunStage2Runner:
    def test_runner_importable(self):
        runner_path = Path(__file__).parent.parent / "experiments" / "run_stage2.py"
        if not runner_path.exists():
            pytest.skip("run_stage2.py not found")
        import importlib.util
        spec = importlib.util.spec_from_file_location("run_stage2", runner_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "run_experiments")
        assert hasattr(mod, "main")

    def test_preview_only_returns_dataframe(self):
        if not _CONFIG_PATH.exists():
            pytest.skip("stage2_config.yaml not found")
        runner_path = Path(__file__).parent.parent / "experiments" / "run_stage2.py"
        if not runner_path.exists():
            pytest.skip("run_stage2.py not found")

        import importlib.util
        spec = importlib.util.spec_from_file_location("run_stage2", runner_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        result = mod.main(
            config_path=str(_CONFIG_PATH),
            preset="quick_debug",
            preview_only=True,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # quick_debug: 2 methods x 1 support x 1 inner_steps x 1 init
