"""
stage2.py — Orchestration utilities for Stage 2 department adaptation.

Responsibilities:
- load and normalize Stage 2 configuration
- resolve experiment grids and experiment identifiers
- load the Stage 2 dataset and Stage 1 artifacts
- build contract-aware department tasks
- dispatch Stage 2 methods (zero-shot / fine-tune / ANIL / FOMAML / MAML)
- save Stage 2 artifacts and summaries

This module is the shared Stage 2 backend for both CLI execution and
notebook-driven execution.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import pandas as pd
import torch
import yaml

from master_thesis.config import DATA_PROCESSED, MODELS_STAGE1, MODELS_STAGE2, SEED
from master_thesis.episode_sampler import (
    build_department_task_table,
    filter_valid_departments,
    make_logistics_meta_test_split,
    summarize_department_tasks,
)
from master_thesis.mlp import TabularMLP, set_seed

from master_thesis.anil import meta_train_anil, evaluate_anil_on_target_episodes
from master_thesis.FOMAML import meta_train_fomaml, evaluate_fomaml_on_target_episodes
from master_thesis.maml import meta_train_maml, evaluate_maml_on_target_episodes
from master_thesis.meta_common import evaluate_zero_shot_on_target_episodes


@dataclass
class Stage2Paths:
    task_table_path: Path
    stage1_model_path: Optional[Path]
    stage1_preprocessor_path: Optional[Path]
    output_dir: Path


def load_stage2_config(config_path: str | Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def normalize_stage2_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if "base_config" in config:
        return copy.deepcopy(config["base_config"])
    return copy.deepcopy(config)


def generate_experiment_id(
    init_name: str,
    method: str,
    n_support_pos: int,
    n_support_neg: int,
    inner_steps: int,
    target_department: str,
    inner_lr: Optional[float] = None,
) -> str:
    safe_target = str(target_department).replace(" ", "-").replace("/", "-")
    id_str = (
        f"stage2__init-{init_name}"
        f"__method-{method}"
        f"__kpos-{n_support_pos}"
        f"__kneg-{n_support_neg}"
        f"__steps-{inner_steps}"
        f"__target-{safe_target}"
    )
    if inner_lr is not None:
        # Format as e.g. __lr-0.001 — strip trailing zeros for readability.
        lr_str = f"{inner_lr:.6f}".rstrip("0").rstrip(".")
        id_str += f"__lr-{lr_str}"
    return id_str


def resolve_experiment_grid(
    full_config: Dict[str, Any],
    preset_name: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    overrides = overrides or {}

    if "base_config" not in full_config:
        resolved = normalize_stage2_config(full_config)
        exp_id = generate_experiment_id(
            init_name=resolved["stage1_init"]["init_name"],
            method=resolved["method"],
            n_support_pos=resolved["support_config"]["n_support_pos"],
            n_support_neg=resolved["support_config"]["n_support_neg"],
            inner_steps=resolved["meta_config"]["inner_steps"],
            target_department=resolved["task_config"]["target_department"],
        )
        return [(exp_id, resolved)]

    base_config = copy.deepcopy(full_config["base_config"])
    presets = full_config.get("presets", {})
    experiment_grid = copy.deepcopy(full_config.get("experiment_grid", {}))

    if preset_name:
        preset_cfg = presets.get(preset_name)
        if preset_cfg is None:
            raise ValueError(f"Unknown Stage 2 preset: {preset_name}")
        experiment_grid = _deep_update(experiment_grid, preset_cfg)

    experiment_grid = _deep_update(experiment_grid, overrides)

    methods = experiment_grid.get("methods", [base_config["method"]])
    support_grid = experiment_grid.get("support_grid", [copy.deepcopy(base_config["support_config"])])
    inner_steps_grid = experiment_grid.get("inner_steps_grid", [base_config["meta_config"]["inner_steps"]])
    init_names = experiment_grid.get("init_names", [base_config["stage1_init"]["init_name"]])
    target_departments = experiment_grid.get(
        "target_departments", [base_config["task_config"]["target_department"]]
    )

    # inner_lr_grid: when present, lr is swept AND included in the experiment ID
    # so different lr values produce distinct experiment directories.
    inner_lr_grid_raw = experiment_grid.get("inner_lr_grid", None)
    sweep_lr = inner_lr_grid_raw is not None
    inner_lr_grid = (
        [float(lr) for lr in inner_lr_grid_raw]
        if sweep_lr
        else [float(base_config["meta_config"]["inner_lr"])]
    )

    preset_meta_overrides = {}
    for key in ["meta_iterations", "n_repeats", "inner_lr", "outer_lr", "meta_batch_size", "target_inner_steps"]:
        if key in experiment_grid:
            preset_meta_overrides[key] = experiment_grid[key]

    experiment_list: List[Tuple[str, Dict[str, Any]]] = []
    for method, support_cfg, inner_steps, init_name, target_department, inner_lr in product(
        methods,
        support_grid,
        inner_steps_grid,
        init_names,
        target_departments,
        inner_lr_grid,
    ):
        resolved = copy.deepcopy(base_config)
        resolved["method"] = method
        resolved["support_config"]["n_support_pos"] = int(support_cfg["n_support_pos"])
        resolved["support_config"]["n_support_neg"] = int(support_cfg["n_support_neg"])
        resolved["meta_config"]["inner_steps"] = int(inner_steps)
        resolved["stage1_init"]["init_name"] = init_name
        resolved["task_config"]["target_department"] = target_department
        resolved["meta_config"].update(copy.deepcopy(preset_meta_overrides))
        # Always apply the per-experiment inner_lr (overrides any preset_meta_overrides value)
        resolved["meta_config"]["inner_lr"] = inner_lr

        exp_id = generate_experiment_id(
            init_name=init_name,
            method=method,
            n_support_pos=resolved["support_config"]["n_support_pos"],
            n_support_neg=resolved["support_config"]["n_support_neg"],
            inner_steps=resolved["meta_config"]["inner_steps"],
            target_department=target_department,
            inner_lr=inner_lr if sweep_lr else None,
        )
        experiment_list.append((exp_id, resolved))

    exp_ids = [exp_id for exp_id, _ in experiment_list]
    if len(exp_ids) != len(set(exp_ids)):
        raise ValueError("Duplicate Stage 2 experiment IDs were generated.")

    return experiment_list


def build_experiment_preview_df(
    experiment_list: Sequence[Tuple[str, Dict[str, Any]]]
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for experiment_id, config in experiment_list:
        rows.append(
            {
                "experiment_id": experiment_id,
                "method": config["method"],
                "n_support_pos": config["support_config"]["n_support_pos"],
                "n_support_neg": config["support_config"]["n_support_neg"],
                "inner_steps": config["meta_config"]["inner_steps"],
                "init_name": config["stage1_init"]["init_name"],
                "target_department": config["task_config"]["target_department"],
                "data_filename": config["data"]["data_filename"],
            }
        )
    return pd.DataFrame(rows)


def resolve_stage2_paths(config: Dict[str, Any], experiment_id: Optional[str] = None) -> Stage2Paths:
    config = normalize_stage2_config(config)

    task_table_path = DATA_PROCESSED / config["data"]["data_filename"]
    init_type = config["stage1_init"]["init_name"]
    method = config["method"]

    if init_type == "random":
        # Random init: no pretrained weights are loaded, but a Stage 1
        # preprocessor is still required so that feature encoding and input
        # dimensionality remain identical to pretrained baselines, making the
        # random-init condition directly comparable.  The preprocessor source
        # is controlled by stage1_init.random_preprocessor_source
        # (default: A_weak_only).
        random_preprocessor_source = config["stage1_init"].get(
            "random_preprocessor_source", "A_weak_only"
        )
        stage1_model_path = None
        stage1_preprocessor_path = (
            MODELS_STAGE1
            / random_preprocessor_source
            / config["stage1_init"]["preprocessor_filename"]
        )
    else:
        stage1_model_path = MODELS_STAGE1 / init_type / config["stage1_init"]["model_filename"]
        stage1_preprocessor_path = MODELS_STAGE1 / init_type / config["stage1_init"]["preprocessor_filename"]

    output_dir = MODELS_STAGE2 / "experiments" / experiment_id if experiment_id else MODELS_STAGE2 / init_type / method
    output_dir.mkdir(parents=True, exist_ok=True)

    return Stage2Paths(
        task_table_path=task_table_path,
        stage1_model_path=stage1_model_path,
        stage1_preprocessor_path=stage1_preprocessor_path,
        output_dir=output_dir,
    )


def get_feature_cols(config: Dict[str, Any], preprocessor=None) -> List[str]:
    features_cfg = config["features"]
    if "columns" in features_cfg:
        feature_cols = features_cfg["columns"]
        if not isinstance(feature_cols, list) or len(feature_cols) == 0:
            raise ValueError("config['features']['columns'] must be a non-empty list.")
        return feature_cols

    if features_cfg.get("source") == "stage1_preprocessor":
        if preprocessor is None:
            raise ValueError("features.source='stage1_preprocessor' requires a loaded preprocessor.")
        if not hasattr(preprocessor, "feature_names_in_"):
            raise ValueError("Loaded preprocessor does not expose feature_names_in_.")
        return list(preprocessor.feature_names_in_)

    raise ValueError(
        "Could not resolve feature columns. Provide either features.columns or "
        "features.source='stage1_preprocessor'."
    )


def load_stage2_dataframe(task_table_path: Path) -> pd.DataFrame:
    if not task_table_path.exists():
        raise FileNotFoundError(f"Stage 2 dataset not found: {task_table_path}")
    return pd.read_csv(task_table_path, low_memory=False)


def load_stage1_preprocessor(preprocessor_path: Optional[Path]):
    if preprocessor_path is None:
        return None
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Stage 1 preprocessor not found: {preprocessor_path}")
    return joblib.load(preprocessor_path)


def load_stage2_raw_data(config: Dict[str, Any], experiment_id: Optional[str] = None) -> Dict[str, Any]:
    config = normalize_stage2_config(config)
    paths = resolve_stage2_paths(config, experiment_id=experiment_id)
    df = load_stage2_dataframe(paths.task_table_path)
    preprocessor = load_stage1_preprocessor(paths.stage1_preprocessor_path)
    feature_cols = get_feature_cols(config, preprocessor=preprocessor)
    return {"paths": paths, "df": df, "preprocessor": preprocessor, "feature_cols": feature_cols}


def build_stage2_model_from_config(
    input_dim: int,
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
) -> TabularMLP:
    mlp_cfg = config["mlp_config"]
    model = TabularMLP(
        input_dim=input_dim,
        hidden_dim_1=mlp_cfg["hidden_dim_1"],
        hidden_dim_2=mlp_cfg["hidden_dim_2"],
        dropout=mlp_cfg["dropout"],
    )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)


def load_stage1_model_weights(
    model: torch.nn.Module,
    model_path: Optional[Path],
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    if model_path is None:
        return model
    if not model_path.exists():
        raise FileNotFoundError(f"Stage 1 model not found: {model_path}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model


def prepare_stage2_tasks(
    config: Dict[str, Any],
    raw_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = normalize_stage2_config(config)
    if raw_data is None:
        raw_data = load_stage2_raw_data(config)

    paths = raw_data["paths"]
    df = raw_data["df"]
    feature_cols = raw_data["feature_cols"]

    task_df = build_department_task_table(
        features_df=df,
        feature_cols=feature_cols,
        contract_id_col=config["data"]["group_col"],
        department_col=config["data"]["department_col"],
        target_col=config["data"]["target_col"],
        observation_year_col=config["data"]["observation_year_col"],
        drop_missing_features=False,
    )

    summary_df = summarize_department_tasks(
        task_df=task_df,
        department_col=config["data"]["department_col"],
        target_col=config["data"]["target_col"],
        contract_id_col=config["data"]["group_col"],
    )

    filtered_task_df, valid_departments, validity_df = filter_valid_departments(
        task_df=task_df,
        department_col=config["data"]["department_col"],
        target_col=config["data"]["target_col"],
        contract_id_col=config["data"]["group_col"],
        n_support_pos=config["support_config"]["n_support_pos"],
        n_support_neg=config["support_config"]["n_support_neg"],
        min_query_contracts=config["task_config"]["min_query_contracts"],
        require_both_query_classes=config["task_config"]["require_both_query_classes"],
    )

    target_department = config["task_config"]["target_department"]
    exclude_departments = set(config["task_config"].get("exclude_departments", []))
    meta_train_departments = [
        dept for dept in valid_departments if dept != target_department and dept not in exclude_departments
    ]

    try:
        target_episodes = make_logistics_meta_test_split(
            task_df=task_df,
            feature_cols=feature_cols,
            target_department=target_department,
            department_col=config["data"]["department_col"],
            target_col=config["data"]["target_col"],
            contract_id_col=config["data"]["group_col"],
            n_support_pos=config["support_config"]["n_support_pos"],
            n_support_neg=config["support_config"]["n_support_neg"],
            n_repeats=config["meta_config"]["n_repeats"],
        )
    except ValueError:
        target_episodes = []

    return {
        "paths": paths,
        "raw_df": df,
        "task_df": task_df,
        "summary_df": summary_df,
        "filtered_task_df": filtered_task_df,
        "valid_departments": valid_departments,
        "validity_df": validity_df,
        "meta_train_departments": meta_train_departments,
        "target_department": target_department,
        "target_episodes": target_episodes,
        "feature_cols": feature_cols,
    }


def run_finetune_baseline(config: Dict[str, Any], prepared: Dict[str, Any]) -> Dict[str, Any]:
    from master_thesis.metrics import evaluate_on_gold_binary

    paths = prepared["paths"]
    target_episodes = prepared["target_episodes"]
    feature_cols = prepared["feature_cols"]
    target_col = config["data"]["target_col"]

    preprocessor = load_stage1_preprocessor(paths.stage1_preprocessor_path)
    if preprocessor is None:
        raise ValueError("Fine-tuning baseline requires a Stage 1 preprocessor.")

    if len(target_episodes) == 0:
        return {
            "method": "finetune",
            "status": "no_target_episodes_available",
            "target_department": prepared["target_department"],
            "raw_metrics": None,
            "predictions": None,
            "history": None,
        }

    inner_lr = config["meta_config"]["inner_lr"]
    target_inner_steps = config["meta_config"].get("target_inner_steps", config["meta_config"]["inner_steps"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dummy_processed = preprocessor.transform(target_episodes[0]["support_df"][feature_cols].iloc[:2])
    if hasattr(dummy_processed, "toarray"):
        dummy_processed = dummy_processed.toarray()
    base_model = build_stage2_model_from_config(dummy_processed.shape[1], config, device)
    base_model = load_stage1_model_weights(base_model, paths.stage1_model_path, device)

    all_metrics, all_predictions = [], []
    criterion = torch.nn.BCEWithLogitsLoss()

    for ep_idx, episode in enumerate(target_episodes):
        X_supp = preprocessor.transform(episode["support_df"][feature_cols])
        X_query = preprocessor.transform(episode["query_df"][feature_cols])
        if hasattr(X_supp, "toarray"):
            X_supp = X_supp.toarray()
        if hasattr(X_query, "toarray"):
            X_query = X_query.toarray()

        y_supp = torch.tensor(episode["support_df"][target_col].to_numpy(), dtype=torch.float32, device=device).view(-1, 1)
        y_query_np = episode["query_df"][target_col].to_numpy()

        X_supp_t = torch.tensor(X_supp, dtype=torch.float32, device=device)
        X_query_t = torch.tensor(X_query, dtype=torch.float32, device=device)

        adapted_model = copy.deepcopy(base_model)
        adapted_model.train()
        optimizer = torch.optim.Adam(adapted_model.parameters(), lr=inner_lr)

        for _ in range(target_inner_steps):
            optimizer.zero_grad()
            loss = criterion(adapted_model(X_supp_t), y_supp)
            loss.backward()
            optimizer.step()

        adapted_model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(adapted_model(X_query_t)).cpu().numpy().ravel()

        metrics = evaluate_on_gold_binary(y_query_np, probs, model_name="FINETUNE")
        metrics["episode_idx"] = ep_idx
        all_metrics.append(metrics)
        all_predictions.append(pd.DataFrame({"episode_idx": ep_idx, "y_true": y_query_np, "y_prob": probs}))

    return {
        "method": "finetune",
        "status": "success",
        "target_department": prepared["target_department"],
        "n_target_episodes": len(target_episodes),
        "raw_metrics": pd.concat(all_metrics, ignore_index=True),
        "predictions": pd.concat(all_predictions, ignore_index=True),
        "history": None,
    }


def _run_meta_method(
    *,
    method_name: str,
    meta_train_fn,
    eval_fn,
    weight_filename: str,
    config: Dict[str, Any],
    prepared: Dict[str, Any],
) -> Dict[str, Any]:
    feature_cols = prepared["feature_cols"]
    task_df = prepared["filtered_task_df"]
    meta_train_departments = prepared["meta_train_departments"]
    target_episodes = prepared["target_episodes"]
    paths = prepared["paths"]

    target_col = config["data"]["target_col"]
    department_col = config["data"]["department_col"]
    contract_id_col = config["data"]["group_col"]
    meta_cfg = config["meta_config"]

    preprocessor = load_stage1_preprocessor(paths.stage1_preprocessor_path)
    if preprocessor is None:
        raise ValueError(f"{method_name.upper()} requires a Stage 1 preprocessor.")

    dummy_processed = preprocessor.transform(task_df[feature_cols].iloc[:2])
    if hasattr(dummy_processed, "toarray"):
        dummy_processed = dummy_processed.toarray()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_stage2_model_from_config(dummy_processed.shape[1], config, device)
    model = load_stage1_model_weights(model, paths.stage1_model_path, device)

    meta_train_result = meta_train_fn(
        model=model,
        preprocessor=preprocessor,
        task_df=task_df,
        feature_cols=feature_cols,
        meta_train_departments=meta_train_departments,
        target_col=target_col,
        department_col=department_col,
        contract_id_col=contract_id_col,
        meta_batch_size=meta_cfg["meta_batch_size"],
        meta_iterations=meta_cfg["meta_iterations"],
        inner_lr=meta_cfg["inner_lr"],
        outer_lr=meta_cfg["outer_lr"],
        inner_steps=meta_cfg["inner_steps"],
        n_support_pos=config["support_config"].get("n_support_pos", 1),
        n_support_neg=config["support_config"].get("n_support_neg", 1),
        device=device,
        random_state=config["seed"],
    )

    trained_model = meta_train_result["model"]
    save_path = paths.output_dir / weight_filename
    torch.save(trained_model.state_dict(), save_path)

    eval_result = eval_fn(
        model=trained_model,
        preprocessor=preprocessor,
        target_episodes=target_episodes,
        feature_cols=feature_cols,
        target_col=target_col,
        inner_lr=meta_cfg["inner_lr"],
        target_inner_steps=meta_cfg.get("target_inner_steps", meta_cfg["inner_steps"]),
        device=device,
    )

    return {
        "method": method_name,
        "status": meta_train_result.get("status", "success"),
        "target_department": prepared["target_department"],
        "weight_path": str(save_path),
        "raw_metrics": eval_result.get("raw_metrics"),
        "predictions": eval_result.get("predictions"),
        "history": meta_train_result.get("history"),
        "diagnostics": meta_train_result.get("diagnostics"),
    }


def run_anil_method(config: Dict[str, Any], prepared: Dict[str, Any]) -> Dict[str, Any]:
    return _run_meta_method(
        method_name="anil",
        meta_train_fn=meta_train_anil,
        eval_fn=evaluate_anil_on_target_episodes,
        weight_filename="anil_initialized.pt",
        config=config,
        prepared=prepared,
    )


def run_fomaml_method(config: Dict[str, Any], prepared: Dict[str, Any]) -> Dict[str, Any]:
    return _run_meta_method(
        method_name="fomaml",
        meta_train_fn=meta_train_fomaml,
        eval_fn=evaluate_fomaml_on_target_episodes,
        weight_filename="fomaml_initialized.pt",
        config=config,
        prepared=prepared,
    )


def run_maml_method(config: Dict[str, Any], prepared: Dict[str, Any]) -> Dict[str, Any]:
    return _run_meta_method(
        method_name="maml",
        meta_train_fn=meta_train_maml,
        eval_fn=evaluate_maml_on_target_episodes,
        weight_filename="maml_initialized.pt",
        config=config,
        prepared=prepared,
    )


def run_zero_shot_method(config: Dict[str, Any], prepared: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the Stage 1 model directly on target episodes (no adaptation).

    This is the pure no-adapt floor: the Stage 1 MLP is loaded with its
    pretrained weights and queried on each target episode's query set. It is
    the correct reference point for every adaptive method (finetune, ANIL,
    FOMAML, MAML) because it isolates the contribution of Stage 2 adaptation
    from the underlying pretrained representation.
    """
    paths = prepared["paths"]
    target_episodes = prepared["target_episodes"]
    feature_cols = prepared["feature_cols"]
    target_col = config["data"]["target_col"]

    preprocessor = load_stage1_preprocessor(paths.stage1_preprocessor_path)
    if preprocessor is None:
        raise ValueError("Zero-shot evaluation requires a Stage 1 preprocessor.")

    if len(target_episodes) == 0:
        return {
            "method": "zero_shot",
            "status": "no_target_episodes_available",
            "target_department": prepared["target_department"],
            "raw_metrics": None,
            "predictions": None,
            "history": None,
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dummy_processed = preprocessor.transform(target_episodes[0]["support_df"][feature_cols].iloc[:2])
    if hasattr(dummy_processed, "toarray"):
        dummy_processed = dummy_processed.toarray()
    base_model = build_stage2_model_from_config(dummy_processed.shape[1], config, device)
    base_model = load_stage1_model_weights(base_model, paths.stage1_model_path, device)

    eval_result = evaluate_zero_shot_on_target_episodes(
        model=base_model,
        preprocessor=preprocessor,
        target_episodes=target_episodes,
        feature_cols=feature_cols,
        target_col=target_col,
        device=device,
        method_name="zero_shot",
        contract_id_col=config["data"].get("group_col", "contract_id"),
        department_col=config["data"].get("department_col", "department"),
    )

    return {
        "method": "zero_shot",
        "status": eval_result.get("status", "success"),
        "target_department": prepared["target_department"],
        "n_target_episodes": eval_result.get("n_target_episodes", 0),
        "raw_metrics": eval_result.get("raw_metrics"),
        "predictions": eval_result.get("predictions"),
        "prediction_stats": eval_result.get("prediction_stats"),
        "history": None,
    }


def run_stage2_method(config: Dict[str, Any], prepared: Dict[str, Any]) -> Dict[str, Any]:
    method = config["method"].lower()
    if method == "zero_shot":
        return run_zero_shot_method(config, prepared)
    if method == "finetune":
        return run_finetune_baseline(config, prepared)
    if method == "anil":
        return run_anil_method(config, prepared)
    if method == "fomaml":
        return run_fomaml_method(config, prepared)
    if method == "maml":
        return run_maml_method(config, prepared)
    raise ValueError(f"Unsupported Stage 2 method: {method}")


def summarize_stage2_result(
    result: Dict[str, Any],
    config: Dict[str, Any],
    experiment_id: Optional[str] = None,
) -> Dict[str, Any]:
    summary = {
        "experiment_id": experiment_id,
        "method": config.get("method"),
        "init_name": config.get("stage1_init", {}).get("init_name"),
        "target_department": config.get("task_config", {}).get("target_department"),
        "n_support_pos": config.get("support_config", {}).get("n_support_pos"),
        "n_support_neg": config.get("support_config", {}).get("n_support_neg"),
        "inner_steps": config.get("meta_config", {}).get("inner_steps"),
        "inner_lr": config.get("meta_config", {}).get("inner_lr"),
        "meta_batch_size": config.get("meta_config", {}).get("meta_batch_size"),
        "status": result.get("status"),
    }

    raw_metrics = result.get("raw_metrics")
    if isinstance(raw_metrics, pd.DataFrame) and not raw_metrics.empty:
        numeric_cols = [
            col for col in raw_metrics.columns
            if pd.api.types.is_numeric_dtype(raw_metrics[col]) and col != "episode_idx"
        ]
        for col in numeric_cols:
            summary[f"{col}_mean"] = float(raw_metrics[col].mean())
            summary[f"{col}_std"] = float(raw_metrics[col].std(ddof=0))
        summary["n_metric_rows"] = int(len(raw_metrics))
    else:
        summary["n_metric_rows"] = 0

    predictions = result.get("predictions")
    summary["n_predictions"] = int(len(predictions)) if isinstance(predictions, pd.DataFrame) else 0

    # Post-hoc collapse diagnostics (method-agnostic; label-free).
    pred_stats = result.get("prediction_stats")
    if not isinstance(pred_stats, pd.DataFrame) and isinstance(predictions, pd.DataFrame) and not predictions.empty:
        try:
            from master_thesis.meta_common import prediction_stats_frame_from_predictions
            pred_stats = prediction_stats_frame_from_predictions(predictions)
        except Exception:
            pred_stats = None

    if isinstance(pred_stats, pd.DataFrame) and not pred_stats.empty:
        summary["n_episodes_diagnosed"] = int(len(pred_stats))
        if "collapsed" in pred_stats.columns:
            summary["collapse_rate"] = float(pred_stats["collapsed"].mean())
        for col in ("pred_mean", "pred_std", "frac_near_zero", "frac_near_one"):
            if col in pred_stats.columns:
                summary[f"{col}_mean"] = float(pred_stats[col].mean())

    return summary


def save_stage2_tables(prepared: Dict[str, Any], output_dir: Path) -> None:
    prepared["summary_df"].to_csv(output_dir / "department_summary.csv", index=False)
    prepared["validity_df"].to_csv(output_dir / "department_validity.csv", index=False)


def _make_serializable(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.nn.Module):
        return f"<torch.nn.Module: {obj.__class__.__name__}>"
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def save_stage2_result(
    result: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: Path,
    experiment_id: Optional[str] = None,
) -> None:
    with open(output_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    if isinstance(result.get("raw_metrics"), pd.DataFrame):
        result["raw_metrics"].to_csv(output_dir / "metrics.csv", index=False)
    if isinstance(result.get("predictions"), pd.DataFrame):
        result["predictions"].to_csv(output_dir / "predictions.csv", index=False)
    if isinstance(result.get("history"), pd.DataFrame):
        result["history"].to_csv(output_dir / "history.csv", index=False)

    # Post-hoc prediction-stats / collapse diagnostic (method-agnostic).
    # Computed from `predictions` if not already supplied by the method.
    predictions_df = result.get("predictions")
    pred_stats = result.get("prediction_stats")
    if not isinstance(pred_stats, pd.DataFrame):
        if isinstance(predictions_df, pd.DataFrame) and not predictions_df.empty:
            try:
                from master_thesis.meta_common import prediction_stats_frame_from_predictions
                pred_stats = prediction_stats_frame_from_predictions(predictions_df)
            except Exception:
                pred_stats = None
    if isinstance(pred_stats, pd.DataFrame) and not pred_stats.empty:
        pred_stats.to_csv(output_dir / "prediction_stats.csv", index=False)

    summary = summarize_stage2_result(result, config, experiment_id)
    with open(output_dir / "stage2_result_summary.json", "w", encoding="utf-8") as f:
        json.dump(_make_serializable(summary), f, indent=2)
    with open(output_dir / "stage2_result.json", "w", encoding="utf-8") as f:
        json.dump(_make_serializable(result), f, indent=2)


def backfill_legacy_stage2_results(
    models_stage2_dir: Optional[Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Scan legacy Stage 2 result directories and backfill them into the
    grid-based experiment registry.

    Legacy format: ``models/stage_2/{init_name}/{method}/``
    Current format: ``models/stage_2/experiments/{experiment_id}/``

    The function is idempotent: experiments already present in the
    registry (matched by experiment_id) are skipped.

    Parameters
    ----------
    models_stage2_dir
        Root of the Stage 2 models directory.  Defaults to MODELS_STAGE2.
    verbose
        Print progress messages.

    Returns
    -------
    pd.DataFrame
        Updated master experiment summary (all experiments, new + existing).
    """
    if models_stage2_dir is None:
        models_stage2_dir = MODELS_STAGE2

    experiment_root = models_stage2_dir / "experiments"
    experiment_root.mkdir(parents=True, exist_ok=True)
    summary_path = experiment_root / "experiment_summary.csv"

    # Load current registry if it exists.
    if summary_path.exists():
        existing_summary = pd.read_csv(summary_path)
        existing_ids: set = set(existing_summary["experiment_id"].dropna().tolist())
    else:
        existing_summary = pd.DataFrame()
        existing_ids = set()

    new_rows: List[Dict[str, Any]] = []

    # Scan directories that look like legacy runs: models/stage_2/{init_name}/{method}/
    # These are NOT inside the "experiments" subfolder.
    for init_dir in models_stage2_dir.iterdir():
        if not init_dir.is_dir() or init_dir.name == "experiments":
            continue
        init_name = init_dir.name
        for method_dir in init_dir.iterdir():
            if not method_dir.is_dir():
                continue

            # Must contain either metrics.csv or stage2_config_snapshot.json to be a valid legacy run.
            metrics_path = method_dir / "metrics.csv"
            snapshot_path = method_dir / "stage2_config_snapshot.json"
            if not metrics_path.exists() and not snapshot_path.exists():
                continue

            # Resolve config from snapshot if available, otherwise use defaults.
            if snapshot_path.exists():
                try:
                    with open(snapshot_path, "r", encoding="utf-8") as f:
                        snap_cfg = json.load(f)
                except Exception:
                    snap_cfg = {}
            else:
                snap_cfg = {}

            method = snap_cfg.get("method") or method_dir.name
            support_cfg = snap_cfg.get("support_config", {})
            n_support_pos = int(support_cfg.get("n_support_pos", 2))
            n_support_neg = int(support_cfg.get("n_support_neg", 2))
            inner_steps = int(
                snap_cfg.get("meta_config", {}).get("inner_steps", 3)
            )
            target_department = (
                snap_cfg.get("task_config", {}).get("target_department", "Logistics")
            )
            resolved_init = snap_cfg.get("stage1_init", {}).get("init_name") or init_name

            exp_id = generate_experiment_id(
                init_name=resolved_init,
                method=method,
                n_support_pos=n_support_pos,
                n_support_neg=n_support_neg,
                inner_steps=inner_steps,
                target_department=target_department,
            )

            if exp_id in existing_ids:
                if verbose:
                    print(f"Already registered, skipping: {exp_id}")
                continue

            # Create destination directory.
            dest_dir = experiment_root / exp_id
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Copy artifacts (do not overwrite if already present).
            artifacts_to_copy = [
                "metrics.csv",
                "predictions.csv",
                "history.csv",
                "department_summary.csv",
                "department_validity.csv",
            ]
            for fname in artifacts_to_copy:
                src = method_dir / fname
                dst = dest_dir / fname
                if src.exists() and not dst.exists():
                    import shutil
                    shutil.copy2(src, dst)

            # Copy weight file (any .pt file).
            for pt_file in method_dir.glob("*.pt"):
                dst_pt = dest_dir / pt_file.name
                if not dst_pt.exists():
                    import shutil
                    shutil.copy2(pt_file, dst_pt)

            # Write resolved_config.yaml from snapshot.
            resolved_config_path = dest_dir / "resolved_config.yaml"
            if not resolved_config_path.exists() and snap_cfg:
                with open(resolved_config_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(snap_cfg, f, sort_keys=False)

            # Reconstruct full summary from metrics.csv.
            summary_row: Dict[str, Any] = {
                "experiment_id": exp_id,
                "method": method,
                "init_name": resolved_init,
                "target_department": target_department,
                "n_support_pos": n_support_pos,
                "n_support_neg": n_support_neg,
                "inner_steps": inner_steps,
                "status": "success",
                "was_skipped": False,
                "backfilled": True,
            }

            if metrics_path.exists():
                try:
                    metrics_df = pd.read_csv(metrics_path)
                    numeric_cols = [
                        c for c in metrics_df.columns
                        if pd.api.types.is_numeric_dtype(metrics_df[c])
                        and c not in ("episode_idx", "repeat_idx")
                    ]
                    for col in numeric_cols:
                        summary_row[f"{col}_mean"] = float(metrics_df[col].mean())
                        summary_row[f"{col}_std"] = float(metrics_df[col].std(ddof=0))
                    summary_row["n_metric_rows"] = int(len(metrics_df))
                except Exception as exc:
                    summary_row["n_metric_rows"] = 0
                    summary_row["metrics_read_error"] = str(exc)
            else:
                summary_row["n_metric_rows"] = 0

            pred_path = method_dir / "predictions.csv"
            if pred_path.exists():
                try:
                    summary_row["n_predictions"] = int(len(pd.read_csv(pred_path)))
                except Exception:
                    summary_row["n_predictions"] = 0
            else:
                summary_row["n_predictions"] = 0

            # Write a canonical stage2_result_summary.json to the new location.
            with open(dest_dir / "stage2_result_summary.json", "w", encoding="utf-8") as f:
                json.dump({k: v for k, v in summary_row.items()}, f, indent=2)

            new_rows.append(summary_row)

            if verbose:
                print(f"Backfilled: {exp_id}")

    if not new_rows:
        if verbose:
            print("No new legacy experiments found to backfill.")
        return existing_summary

    new_df = pd.DataFrame(new_rows)
    if existing_summary.empty:
        updated_summary = new_df
    else:
        updated_summary = pd.concat([existing_summary, new_df], ignore_index=True, sort=False)

    sort_cols = [
        c for c in ["method", "init_name", "n_support_pos", "n_support_neg", "inner_steps", "target_department"]
        if c in updated_summary.columns
    ]
    if sort_cols:
        updated_summary = updated_summary.sort_values(sort_cols).reset_index(drop=True)

    updated_summary.to_csv(summary_path, index=False)

    if verbose:
        print(f"Added {len(new_rows)} legacy experiment(s) to registry.")
        print(f"Saved updated experiment_summary.csv to {summary_path}")

    return updated_summary


def run_stage2_pipeline(
    config: Dict[str, Any],
    experiment_id: Optional[str] = None,
    raw_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = normalize_stage2_config(config)
    set_seed(config.get("seed", SEED))

    if raw_data is None:
        raw_data = load_stage2_raw_data(config)

    prepared = prepare_stage2_tasks(config=config, raw_data=raw_data)
    if experiment_id:
        prepared["paths"] = resolve_stage2_paths(config, experiment_id=experiment_id)

    output_dir = prepared["paths"].output_dir
    save_stage2_tables(prepared, output_dir)
    result = run_stage2_method(config, prepared)
    save_stage2_result(result, config, output_dir, experiment_id=experiment_id)

    return {
        "prepared": prepared,
        "result": result,
        "summary": summarize_stage2_result(result, config, experiment_id=experiment_id),
    }
