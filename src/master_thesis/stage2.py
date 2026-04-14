"""
stage2.py — Orchestration utilities for Stage 2 department adaptation.

Responsibilities:
- load Stage 2 config
- load Stage 2 dataset
- load Stage 1 pretrained MLP + preprocessor
- build contract-aware department tasks
- select meta-train vs meta-test departments
- dispatch Stage 2 methods (fine-tune / ANIL / FOMAML / optional MAML)
- save Stage 2 artifacts and summaries

This file is the pipeline coordinator.

"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import copy
import json

import joblib
import pandas as pd
import torch
import yaml

from master_thesis.config import DATA_PROCESSED, MODELS_STAGE1, MODELS_STAGE2, SEED
from master_thesis.episode_sampler import (
    build_department_task_table,
    summarize_department_tasks,
    filter_valid_departments,
    make_logistics_meta_test_split,
)
from master_thesis.mlp import TabularMLP, set_seed

# anil
from master_thesis.anil import (
    meta_train_anil,
    evaluate_anil_on_target_episodes,
)
# fomaml
from master_thesis.FOMAML import (
    meta_train_fomaml,
    evaluate_fomaml_on_target_episodes,
)
# maml
from master_thesis.maml import (
    meta_train_maml,
    evaluate_maml_on_target_episodes,
)



# ============================================================================
# Config loading
# ============================================================================

@dataclass
class Stage2Paths:
    task_table_path: Path
    stage1_model_path: Optional[Path]
    stage1_preprocessor_path: Optional[Path]
    output_dir: Path


def load_stage2_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load YAML config for Stage 2.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def resolve_stage2_paths(config: Dict[str, Any]) -> Stage2Paths:
    """
    Resolve all important file paths from config.
    """
    data_filename = config["data"]["data_filename"]
    task_table_path = DATA_PROCESSED / data_filename

    init_type = config["stage1_init"]["init_name"]
    method = config["method"]

    if init_type == "random":
        stage1_model_path = None
        stage1_preprocessor_path = None
    else:
        stage1_model_path = (
            MODELS_STAGE1
            / init_type
            / config["stage1_init"]["model_filename"]
        )
        stage1_preprocessor_path = (
            MODELS_STAGE1
            / init_type
            / config["stage1_init"]["preprocessor_filename"]
        )

    output_dir = MODELS_STAGE2 / init_type / method
    output_dir.mkdir(parents=True, exist_ok=True)

    return Stage2Paths(
        task_table_path=task_table_path,
        stage1_model_path=stage1_model_path,
        stage1_preprocessor_path=stage1_preprocessor_path,
        output_dir=output_dir,
    )


# ============================================================================
# Feature handling
# ============================================================================
def get_feature_cols(
    config: Dict[str, Any],
    preprocessor=None,
) -> List[str]:
    """
    Resolve Stage 2 feature columns.

    Supported modes:
    - explicit list in config["features"]["columns"]
    - automatic inference from Stage 1 preprocessor via
      config["features"]["source"] == "stage1_preprocessor"
    """
    features_cfg = config["features"]

    # Option 1: explicit feature list in YAML
    if "columns" in features_cfg:
        feature_cols = features_cfg["columns"]

        if not isinstance(feature_cols, list) or len(feature_cols) == 0:
            raise ValueError(
                "config['features']['columns'] must be a non-empty list."
            )

        return feature_cols

    # Option 2: infer from fitted Stage 1 preprocessor
    if features_cfg.get("source") == "stage1_preprocessor":
        if preprocessor is None:
            raise ValueError(
                "features.source='stage1_preprocessor' requires a loaded preprocessor."
            )

        if not hasattr(preprocessor, "feature_names_in_"):
            raise ValueError(
                "Loaded preprocessor does not expose feature_names_in_."
            )

        return list(preprocessor.feature_names_in_)

    raise ValueError(
        "Could not resolve feature columns. "
        "Provide either features.columns or features.source='stage1_preprocessor'."
    )


# ============================================================================
# Data + artifact loading
# ============================================================================

def load_stage2_dataframe(task_table_path: Path) -> pd.DataFrame:
    """
    Load the merged gold-labeled Stage 2 dataframe.
    """
    if not task_table_path.exists():
        raise FileNotFoundError(f"Stage 2 dataset not found: {task_table_path}")

    df = pd.read_csv(task_table_path, low_memory=False)
    return df

# mlp_pretrained_preprocessor.joblib
def load_stage1_preprocessor(preprocessor_path: Optional[Path]):
    """
    Load the Stage 1 fitted preprocessor.
    """
    if preprocessor_path is None:
        return None

    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Stage 1 preprocessor not found: {preprocessor_path}")

    return joblib.load(preprocessor_path)


def build_stage2_model_from_config(
    input_dim: int,
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
) -> TabularMLP:
    """
    Build Stage 2 MLP architecture consistent with mlp.py.
    """
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
    """
    Load pretrained Stage 1 weights into a Stage 2 model.
    """
    if model_path is None:
        return model

    if not model_path.exists():
        raise FileNotFoundError(f"Stage 1 model not found: {model_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


# ============================================================================
# Task preparation
# ============================================================================

def prepare_stage2_tasks(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load Stage 2 data and construct contract-aware task objects.
    """
    paths = resolve_stage2_paths(config)
    df = load_stage2_dataframe(paths.task_table_path)

    # Backward compatibility for older Stage 1 preprocessors.
    # The feature was renamed during feature engineering from
    # lpi_relative_risk to lpi_below_supplier_median.
    if "lpi_relative_risk" not in df.columns and "lpi_below_supplier_median" in df.columns:
        df["lpi_relative_risk"] = df["lpi_below_supplier_median"]

    preprocessor = load_stage1_preprocessor(paths.stage1_preprocessor_path)
    feature_cols = get_feature_cols(config, preprocessor=preprocessor)


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

    meta_train_departments = [
        dept for dept in valid_departments
        if dept != target_department and dept not in config["task_config"]["exclude_departments"]
    ]

    try:
        logistics_test_episodes = make_logistics_meta_test_split(
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
        logistics_test_episodes = []

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
        "target_episodes": logistics_test_episodes,
        "feature_cols": feature_cols,
    }


# ============================================================================
# Method dispatch
# ============================================================================


def run_finetune_baseline(
    config: Dict[str, Any],
    prepared: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Smoke-test version of the Stage 2 fine-tuning baseline.

    This version does not train yet.
    It only:
    - loads the Stage 1 preprocessor
    - grabs one target episode
    - preprocesses support/query features
    - builds a Stage 2 model
    - loads Stage 1 pretrained weights
    """
    paths = prepared["paths"]
    target_episodes = prepared["target_episodes"]
    feature_cols = prepared["feature_cols"]
    target_col = config["data"]["target_col"]

    # ---------------------------------------------------------
    # 1. Load Stage 1 preprocessor
    # ---------------------------------------------------------
    preprocessor = load_stage1_preprocessor(paths.stage1_preprocessor_path)

    if preprocessor is None:
        raise ValueError(
            "Fine-tuning baseline requires a Stage 1 preprocessor."
        )

    # ---------------------------------------------------------
    # 2. Check that at least one target episode exists
    # ---------------------------------------------------------
    if len(target_episodes) == 0:
        return {
            "method": "finetune",
            "status": "no_target_episodes_available",
            "target_department": prepared["target_department"],
        }

    import copy
    import pandas as pd
    from master_thesis.metrics import evaluate_on_gold_binary

    inner_lr = config["meta_config"]["inner_lr"]
    target_inner_steps = config["meta_config"]["target_inner_steps"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = build_stage2_model_from_config(
        input_dim=preprocessor.transform(target_episodes[0]["support_df"][feature_cols].iloc[:2]).shape[1],
        config=config,
        device=device,
    )
    base_model = load_stage1_model_weights(
        model=base_model,
        model_path=paths.stage1_model_path,
        device=device,
    )
    
    all_metrics = []
    criterion = torch.nn.BCEWithLogitsLoss()

    for ep_idx, episode in enumerate(target_episodes):
        X_supp_raw = episode["support_df"][feature_cols]
        X_query_raw = episode["query_df"][feature_cols]
        
        X_supp = preprocessor.transform(X_supp_raw)
        X_query = preprocessor.transform(X_query_raw)
        if hasattr(X_supp, "toarray"): X_supp = X_supp.toarray()
        if hasattr(X_query, "toarray"): X_query = X_query.toarray()
        
        y_supp = torch.tensor(episode["support_df"][target_col].to_numpy(), dtype=torch.float32, device=device).view(-1, 1)
        y_query_np = episode["query_df"][target_col].to_numpy()
        
        X_supp_t = torch.tensor(X_supp, dtype=torch.float32, device=device)
        X_query_t = torch.tensor(X_query, dtype=torch.float32, device=device)
        
        adapted_model = copy.deepcopy(base_model)
        adapted_model.train()
        optimizer = torch.optim.Adam(adapted_model.parameters(), lr=inner_lr)
        
        for _ in range(target_inner_steps):
            optimizer.zero_grad()
            logits = adapted_model(X_supp_t)
            loss = criterion(logits, y_supp)
            loss.backward()
            optimizer.step()
            
        adapted_model.eval()
        with torch.no_grad():
            logits_query = adapted_model(X_query_t)
            probs = torch.sigmoid(logits_query).cpu().numpy().ravel()
            
        metrics = evaluate_on_gold_binary(y_query_np, probs, model_name="FINETUNE") # All Caps to match visually
        metrics["episode_idx"] = ep_idx
        all_metrics.append(metrics)

    eval_result = {
        "method": "finetune",
        "status": "success",
        "n_target_episodes": len(target_episodes),
        "raw_metrics": pd.concat(all_metrics, ignore_index=True),
    }

    return {
        "method": "finetune",
        "status": "fully_integrated",
        "target_department": prepared["target_department"],
        "weight_path": str(paths.stage1_model_path),
        "target_eval_result": eval_result,
    }


def run_anil_method(
    config: Dict[str, Any],
    prepared: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run the fully integrated ANIL Stage 2 method.
    """
    from master_thesis.anil import meta_train_anil, evaluate_anil_on_target_episodes
    
    feature_cols = prepared["feature_cols"]
    task_df = prepared["filtered_task_df"]
    meta_train_departments = prepared["meta_train_departments"]
    target_episodes = prepared["target_episodes"]
    paths = prepared["paths"]

    target_col = config["data"]["target_col"]
    department_col = config["data"]["department_col"]
    contract_id_col = config["data"]["group_col"]

    meta_cfg = config["meta_config"]
    
    # ---------------------------------------------------------
    # 1. Pipeline Initialization
    # ---------------------------------------------------------
    preprocessor = load_stage1_preprocessor(paths.stage1_preprocessor_path)
    if preprocessor is None:
        raise ValueError("ANIL requires a Stage 1 preprocessor.")
        
    dummy_processed = preprocessor.transform(task_df[feature_cols].iloc[:2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_stage2_model_from_config(
        input_dim=dummy_processed.shape[1],
        config=config,
        device=device,
    )
    model = load_stage1_model_weights(
        model=model,
        model_path=paths.stage1_model_path,
        device=device,
    )

    # ---------------------------------------------------------
    # 2. Meta-train ANIL across source departments
    # ---------------------------------------------------------
    meta_train_result = meta_train_anil(
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
    
    # ---------------------------------------------------------
    # 3. Save Final Meta-Network Weights to Disk!
    # ---------------------------------------------------------
    save_path = paths.output_dir / "anil_initialized.pt"
    torch.save(trained_model.state_dict(), save_path)

    # ---------------------------------------------------------
    # 4. Evaluate on held-out target episodes
    # ---------------------------------------------------------
    eval_result = evaluate_anil_on_target_episodes(
        model=trained_model,
        preprocessor=preprocessor,
        target_episodes=target_episodes,
        feature_cols=feature_cols,
        target_col=target_col,
        inner_lr=meta_cfg["inner_lr"],
        target_inner_steps=meta_cfg["target_inner_steps"],
        device=device,
    )

    return {
        "method": "anil",
        "status": "fully_integrated",
        "target_department": prepared["target_department"],
        "weight_path": str(save_path),
        "target_eval_result": eval_result,
        "history": meta_train_result.get("history"),
    }


def run_fomaml_method(
    config: Dict[str, Any],
    prepared: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run the fully integrated FOMAML Stage 2 method.
    """
    from master_thesis.FOMAML import meta_train_fomaml, evaluate_fomaml_on_target_episodes
    
    feature_cols = prepared["feature_cols"]
    task_df = prepared["filtered_task_df"]
    meta_train_departments = prepared["meta_train_departments"]
    target_episodes = prepared["target_episodes"]
    paths = prepared["paths"]

    target_col = config["data"]["target_col"]
    department_col = config["data"]["department_col"]
    contract_id_col = config["data"]["group_col"]

    meta_cfg = config["meta_config"]
    
    # ---------------------------------------------------------
    # 1. Pipeline Initialization
    # ---------------------------------------------------------
    preprocessor = load_stage1_preprocessor(paths.stage1_preprocessor_path)
    if preprocessor is None:
        raise ValueError("FOMAML requires a Stage 1 preprocessor.")
        
    dummy_processed = preprocessor.transform(task_df[feature_cols].iloc[:2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_stage2_model_from_config(
        input_dim=dummy_processed.shape[1],
        config=config,
        device=device,
    )
    model = load_stage1_model_weights(
        model=model,
        model_path=paths.stage1_model_path,
        device=device,
    )

    # ---------------------------------------------------------
    # 2. Meta-train FOMAML
    # ---------------------------------------------------------
    meta_train_result = meta_train_fomaml(
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
    
    # ---------------------------------------------------------
    # 3. Save Final Meta-Network Weights to Disk!
    # ---------------------------------------------------------
    save_path = paths.output_dir / "fomaml_initialized.pt"
    torch.save(trained_model.state_dict(), save_path)

    # ---------------------------------------------------------
    # 4. Evaluate on target episodes
    # ---------------------------------------------------------
    eval_result = evaluate_fomaml_on_target_episodes(
        model=trained_model,  
        preprocessor=preprocessor,
        target_episodes=target_episodes,
        feature_cols=feature_cols,
        target_col=target_col,
        inner_lr=meta_cfg["inner_lr"],
        target_inner_steps=meta_cfg["target_inner_steps"],
        device=device,
    )

    return {
        "method": "fomaml",
        "status": "fully_integrated",
        "target_department": prepared["target_department"],
        "weight_path": str(save_path),
        "target_eval_result": eval_result,
        "history": meta_train_result.get("history"),
    }

def run_maml_method(
    config: Dict[str, Any],
    prepared: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run the fully integrated MAML Stage 2 method.
    """
    from master_thesis.maml import meta_train_maml, evaluate_maml_on_target_episodes
    
    feature_cols = prepared["feature_cols"]
    task_df = prepared["filtered_task_df"]
    meta_train_departments = prepared["meta_train_departments"]
    target_episodes = prepared["target_episodes"]
    paths = prepared["paths"]

    target_col = config["data"]["target_col"]
    department_col = config["data"]["department_col"]
    contract_id_col = config["data"]["group_col"]

    meta_cfg = config["meta_config"]
    
    # ---------------------------------------------------------
    # 1. Pipeline Initialization
    # ---------------------------------------------------------
    preprocessor = load_stage1_preprocessor(paths.stage1_preprocessor_path)
    if preprocessor is None:
        raise ValueError("MAML requires a Stage 1 preprocessor.")
        
    dummy_processed = preprocessor.transform(task_df[feature_cols].iloc[:2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_stage2_model_from_config(
        input_dim=dummy_processed.shape[1],
        config=config,
        device=device,
    )
    model = load_stage1_model_weights(
        model=model,
        model_path=paths.stage1_model_path,
        device=device,
    )

    # ---------------------------------------------------------
    # 2. Meta-train Full MAML
    # ---------------------------------------------------------
    meta_train_result = meta_train_maml(
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
    
    # ---------------------------------------------------------
    # 3. Save Final Meta-Network Weights to Disk!
    # ---------------------------------------------------------
    save_path = paths.output_dir / "maml_initialized.pt"
    torch.save(trained_model.state_dict(), save_path)

    # ---------------------------------------------------------
    # 4. Evaluate on target episodes
    # ---------------------------------------------------------
    eval_result = evaluate_maml_on_target_episodes(
        model=trained_model,  
        preprocessor=preprocessor,
        target_episodes=target_episodes,
        feature_cols=feature_cols,
        target_col=target_col,
        inner_lr=meta_cfg["inner_lr"],
        target_inner_steps=meta_cfg["target_inner_steps"],
        device=device,
    )

    return {
        "method": "maml",
        "status": "fully_integrated",
        "target_department": prepared["target_department"],
        "weight_path": str(save_path),
        "target_eval_result": eval_result,
        "history": meta_train_result.get("history"),
    }



def run_stage2_method(
    config: Dict[str, Any],
    prepared: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Dispatch the selected Stage 2 method.
    """
    method = config["method"].lower()

    if method == "finetune":
        return run_finetune_baseline(config, prepared)

    if method == "anil":
        return run_anil_method(config, prepared)

    if method == "fomaml":
        return run_fomaml_method(config, prepared)

    if method == "maml":
        return run_maml_method(config, prepared)

    raise ValueError(f"Unsupported Stage 2 method: {method}")


# ============================================================================
# Saving
# ============================================================================

def save_stage2_tables(
    prepared: Dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Save useful Stage 2 summary tables for reproducibility.
    """
    prepared["summary_df"].to_csv(output_dir / "department_summary.csv", index=False)
    prepared["validity_df"].to_csv(output_dir / "department_validity.csv", index=False)


def save_stage2_result(
    result: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Save config and result summary.
    """
    with open(output_dir / "stage2_config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    def _make_serializable(obj):
        import pandas as pd
        import torch
        from pathlib import Path

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

    safe_result = _make_serializable(result)

    with open(output_dir / "stage2_result.json", "w", encoding="utf-8") as f:
        json.dump(safe_result, f, indent=2)


# ============================================================================
# Top-level pipeline
# ============================================================================

def run_stage2_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the Stage 2 pipeline end-to-end.

    High-level flow:
    1. set seed
    2. resolve paths
    3. prepare task data
    4. run selected Stage 2 method
    5. save artifacts
    """
    seed = config.get("seed", SEED)
    set_seed(seed)

    prepared = prepare_stage2_tasks(config)
    output_dir = prepared["paths"].output_dir

    save_stage2_tables(prepared, output_dir)

    result = run_stage2_method(config, prepared)
    save_stage2_result(result, config, output_dir)

    return {
        "prepared": prepared,
        "result": result,
    }