"""
stage2b_synthetic.py — Reusable Stage 2b synthetic-support experiment logic.

This module implements the label-budget curve experiment for the thesis:
"How little real department-specific labelled data is needed, and how well
does the model perform at each level of real labelled input?"

It builds on top of:
- episode_sampler.sample_episode_with_synthetic_support()
- synthetic_augment.augment_support()
- synthetic_diagnostics.compute_realism_diagnostics()
- metrics.evaluate_on_gold_binary()

Design rules
------------
- SMOTENC is applied ONLY to meta-test support sets (never meta-training).
- Query sets are always real-only.
- Synthetic rows never enter evaluation metrics.
- ANIL / FOMAML / MAML are wired through existing project functions.
  If an adapter cannot be wired safely, a clear NotImplementedError is raised.
- No files are written inside experiment runner functions.
- Input DataFrames are never mutated.

Public API
----------
resolve_categorical_cols(...)          → list[str]
build_department_eligibility_table(...) → pd.DataFrame
compute_support_accounting(...)        → dict
adapt_and_evaluate(...)                → dict
build_stage2b_result_row(...)          → dict
run_stage2b_experiment(...)            → list[dict]
run_stage2b_grid(...)                  → pd.DataFrame
"""

from __future__ import annotations

import copy
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from master_thesis.config import SEED
from master_thesis.episode_sampler import (
    sample_episode_with_synthetic_support,
    filter_valid_departments,
)
from master_thesis.metrics import evaluate_on_gold_binary
from master_thesis.synthetic_diagnostics import compute_realism_diagnostics
from master_thesis.stage2 import (
    EPISODIC_RESULT_ROW_SCHEMA,
    generate_experiment_id,
)

logger = logging.getLogger(__name__)


# ============================================================================
# 1. resolve_categorical_cols
# ============================================================================

def resolve_categorical_cols(
    df: pd.DataFrame,
    feature_cols: list[str],
    explicit_categorical_cols: list[str] | None = None,
) -> list[str]:
    """Resolve which feature columns are categorical.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame containing the features.
    feature_cols : list[str]
        All feature column names.
    explicit_categorical_cols : list[str] | None
        If provided, use these (intersected with *feature_cols*).
        Otherwise infer: any column whose dtype is not numeric
        (including ``pd.StringDtype``) is considered categorical.

    Returns
    -------
    list[str]
        Sorted list of categorical column names that exist in *feature_cols*.
    """
    if explicit_categorical_cols is not None:
        return sorted(c for c in explicit_categorical_cols if c in feature_cols)

    cat_cols: list[str] = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            cat_cols.append(col)
    return sorted(cat_cols)


# ============================================================================
# 2. build_department_eligibility_table
# ============================================================================

def build_department_eligibility_table(
    df: pd.DataFrame,
    department_col: str = "department",
    contract_id_col: str = "contract_id",
    target_col: str = "gold_y",
    label_source_col: str = "label_source",
    allowed_label_sources: tuple[str, ...] = ("manual_hardcoded",),
    max_k: int = 5,
) -> pd.DataFrame:
    """Create an analysis-ready overview of department Stage 2/2b eligibility.

    Counts are at the *contract* level (not row level) because support
    feasibility is determined by number of distinct contracts with gold labels.

    Parameters
    ----------
    df : pd.DataFrame
        The full task table (rows × contracts × years).
    department_col, contract_id_col, target_col, label_source_col : str
        Column names.
    allowed_label_sources : tuple[str, ...]
        Only rows with ``label_source`` in this set are counted as "gold".
        Pass ``None`` to skip source filtering (use all rows with non-NaN target).
    max_k : int
        Maximum balanced k to check (1..max_k).

    Returns
    -------
    pd.DataFrame
        One row per department with eligibility columns.
    """
    work = df.copy()

    # Filter to gold-labelled rows only
    work = work.dropna(subset=[target_col])
    if allowed_label_sources is not None and label_source_col in work.columns:
        work = work[work[label_source_col].isin(allowed_label_sources)]

    departments = sorted(df[department_col].dropna().unique())
    rows: list[dict] = []

    for dept in departments:
        dept_df = work[work[department_col] == dept]

        # Count distinct contracts per class
        contracts_by_class = (
            dept_df.groupby(target_col)[contract_id_col]
            .nunique()
        )
        n_pos = int(contracts_by_class.get(1, 0))
        n_neg = int(contracts_by_class.get(0, 0))
        n_gold_contracts = n_pos + n_neg
        n_gold_rows = len(dept_df)

        max_balanced_real_k = min(n_pos, n_neg)

        # Query-safe k: leaves at least 1 pos + 1 neg for query
        max_query_safe_real_k = min(max(n_pos - 1, 0), max(n_neg - 1, 0))

        row: dict[str, Any] = {
            "department": dept,
            "n_gold_rows": n_gold_rows,
            "n_gold_contracts": n_gold_contracts,
            "n_pos_contracts": n_pos,
            "n_neg_contracts": n_neg,
            "max_balanced_real_k": max_balanced_real_k,
            "max_query_safe_real_k": max_query_safe_real_k,
        }

        for k in range(1, max_k + 1):
            row[f"can_run_k{k}"] = max_balanced_real_k >= k
            # SMOTENC needs ≥2 real examples per class for proper interpolation
            # SMOTENC needs ≥2 real examples per class AND enough contracts
            # to actually draw k balanced support rows.
            row[f"smotenc_k{k}_possible"] = (
                max_balanced_real_k >= k and n_pos >= 2 and n_neg >= 2
            )
            # AUROC on query requires both classes in the query set
            query_pos = max(n_pos - k, 0)
            query_neg = max(n_neg - k, 0)
            row[f"auroc_query_possible_at_k{k}"] = (query_pos >= 1 and query_neg >= 1)

        row["smotenc_possible"] = (n_pos >= 2 and n_neg >= 2)

        # Stage 2b evaluable: can run at least k=2 with query feasibility
        row["stage2b_evaluable"] = (
            max_query_safe_real_k >= 2
            and n_pos >= 2
            and n_neg >= 2
        )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# 3. compute_support_accounting
# ============================================================================

def compute_support_accounting(
    support_df: pd.DataFrame,
    target_col: str = "gold_y",
    synthetic_col: str = "is_synthetic",
    contract_id_col: str = "contract_id",
) -> dict:
    """Separate real vs synthetic support counts.

    Parameters
    ----------
    support_df : pd.DataFrame
        Support set (may contain both real and synthetic rows).
    target_col : str
        Binary target column.
    synthetic_col : str
        Boolean column indicating synthetic rows.
    contract_id_col : str
        Contract identifier column for contract-level counts.

    Returns
    -------
    dict
        Accounting dictionary with real/synthetic/effective counts at both
        row and contract level.
    """
    if synthetic_col not in support_df.columns:
        is_syn = pd.Series(False, index=support_df.index)
    else:
        is_syn = support_df[synthetic_col].fillna(False).astype(bool)

    real_mask = ~is_syn
    syn_mask = is_syn

    real_df = support_df[real_mask]
    syn_df = support_df[syn_mask]

    y_real = real_df[target_col] if len(real_df) > 0 else pd.Series(dtype=float)
    y_syn = syn_df[target_col] if len(syn_df) > 0 else pd.Series(dtype=float)

    real_k_pos = int((y_real == 1).sum())
    real_k_neg = int((y_real == 0).sum())
    syn_k_pos = int((y_syn == 1).sum())
    syn_k_neg = int((y_syn == 0).sum())

    accounting = {
        "real_k_pos": real_k_pos,
        "real_k_neg": real_k_neg,
        "synthetic_k_pos": syn_k_pos,
        "synthetic_k_neg": syn_k_neg,
        "effective_k_pos": real_k_pos + syn_k_pos,
        "effective_k_neg": real_k_neg + syn_k_neg,
        "n_real_support_rows": int(real_mask.sum()),
        "n_synthetic_support_rows": int(syn_mask.sum()),
        "n_effective_support_rows": len(support_df),
    }

    # Contract-level counts (if contract_id_col available)
    if contract_id_col in support_df.columns:
        accounting["n_real_support_contracts"] = int(
            real_df[contract_id_col].nunique() if len(real_df) > 0 else 0
        )
        accounting["n_synthetic_support_contracts"] = int(
            syn_df[contract_id_col].nunique() if len(syn_df) > 0 else 0
        )
    else:
        accounting["n_real_support_contracts"] = accounting["n_real_support_rows"]
        accounting["n_synthetic_support_contracts"] = accounting["n_synthetic_support_rows"]

    return accounting


# ============================================================================
# 4. adapt_and_evaluate
# ============================================================================

def _prepare_tensors(
    support_df: pd.DataFrame,
    query_df: pd.DataFrame,
    feature_cols: list[str],
    preprocessor: Any,
    target_col: str,
    device: torch.device,
) -> tuple:
    """Transform DataFrames → tensors via preprocessor, matching ANIL/FOMAML/MAML conventions."""
    X_supp = preprocessor.transform(support_df[feature_cols])
    X_query = preprocessor.transform(query_df[feature_cols])

    if hasattr(X_supp, "toarray"):
        X_supp = X_supp.toarray()
    if hasattr(X_query, "toarray"):
        X_query = X_query.toarray()

    y_supp_np = support_df[target_col].to_numpy(dtype=np.float32)
    y_query_np = query_df[target_col].to_numpy(dtype=np.float32)

    X_supp_t = torch.tensor(X_supp, dtype=torch.float32, device=device)
    y_supp_t = torch.tensor(y_supp_np, dtype=torch.float32, device=device)
    X_query_t = torch.tensor(X_query, dtype=torch.float32, device=device)
    y_query_t = torch.tensor(y_query_np, dtype=torch.float32, device=device)

    return X_supp_t, y_supp_t, X_query_t, y_query_t, y_query_np


def _finetune_and_predict(
    model: nn.Module,
    X_support: torch.Tensor,
    y_support: torch.Tensor,
    X_query: torch.Tensor,
    inner_lr: float,
    inner_steps: int,
) -> np.ndarray:
    """Fine-tune a *copy* of the model on support, predict on query."""
    ft_model = copy.deepcopy(model)
    ft_model.train()
    optimizer = torch.optim.Adam(ft_model.parameters(), lr=inner_lr)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(inner_steps):
        optimizer.zero_grad()
        logits = ft_model(X_support).squeeze(-1)
        loss = criterion(logits, y_support)
        loss.backward()
        optimizer.step()

    ft_model.eval()
    with torch.no_grad():
        logits = ft_model(X_query).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def adapt_and_evaluate(
    method: str,
    base_model: nn.Module,
    preprocessor: Any,
    episode: dict,
    feature_cols: list[str],
    target_col: str = "gold_y",
    inner_lr: float = 0.01,
    inner_steps: int = 5,
    device: str | None = None,
    **kwargs,
) -> dict:
    """Adapt on an episode's support set, evaluate on its query set.

    Parameters
    ----------
    method : str
        One of ``"zero_shot"``, ``"finetune"``, ``"anil"``, ``"fomaml"``, ``"maml"``.
    base_model : nn.Module
        The TabularMLP (or compatible) model with loaded weights.
    preprocessor
        Fitted sklearn preprocessor (ColumnTransformer / Pipeline).
    episode : dict
        Must contain ``"support_df"`` and ``"query_df"`` DataFrames.
    feature_cols : list[str]
        Feature column names.
    target_col : str
        Target column in support/query DataFrames.
    inner_lr, inner_steps : float, int
        Adaptation hyperparameters.
    device : str | None
        PyTorch device string.

    Returns
    -------
    dict
        Contains at least: ``y_true``, ``y_pred``, ``method``,
        ``n_query_rows``, ``n_query_pos``, ``n_query_neg``,
        ``inner_steps``, ``inner_lr``.

    Raises
    ------
    NotImplementedError
        If an ANIL/FOMAML/MAML adapter function cannot be wired.
    ValueError
        If *method* is unknown.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    support_df = episode["support_df"]
    query_df = episode["query_df"]

    X_supp_t, y_supp_t, X_query_t, y_query_t, y_query_np = _prepare_tensors(
        support_df, query_df, feature_cols, preprocessor, target_col, dev,
    )

    model = copy.deepcopy(base_model).to(dev)
    model.eval()

    method_lower = method.lower().replace("-", "_")

    if method_lower == "zero_shot":
        # No adaptation: predict directly on query
        with torch.no_grad():
            logits = model(X_query_t).squeeze(-1)
            y_pred = torch.sigmoid(logits).cpu().numpy()

    elif method_lower in ("finetune", "fine_tune"):
        y_pred = _finetune_and_predict(
            model, X_supp_t, y_supp_t, X_query_t, inner_lr, inner_steps,
        )

    elif method_lower == "anil":
        try:
            from master_thesis.anil import adapt_anil_on_episode
        except ImportError as exc:
            raise NotImplementedError(
                f"ANIL adapter could not be imported: {exc}. "
                "Check that master_thesis.anil.adapt_anil_on_episode exists."
            ) from exc

        head_weight, head_bias = adapt_anil_on_episode(
            model, X_supp_t, y_supp_t,
            inner_lr=inner_lr, inner_steps=inner_steps,
        )
        # Predict on query using adapted head
        with torch.no_grad():
            features = X_query_t
            for layer in model.net[:-1]:
                features = layer(features)
            logits = F.linear(features, head_weight, head_bias).squeeze(-1)
            y_pred = torch.sigmoid(logits).cpu().numpy()

    elif method_lower == "fomaml":
        try:
            from master_thesis.FOMAML import adapt_fomaml_on_episode
        except ImportError as exc:
            raise NotImplementedError(
                f"FOMAML adapter could not be imported: {exc}. "
                "Check that master_thesis.FOMAML.adapt_fomaml_on_episode exists."
            ) from exc

        adapted_params = adapt_fomaml_on_episode(
            model, X_supp_t, y_supp_t,
            inner_lr=inner_lr, inner_steps=inner_steps,
        )
        # Predict using functional call with adapted parameters
        buffers = dict(model.named_buffers())
        with torch.no_grad():
            logits = torch.func.functional_call(
                model, (adapted_params, buffers), X_query_t,
            ).squeeze(-1)
            y_pred = torch.sigmoid(logits).cpu().numpy()

    elif method_lower == "maml":
        try:
            from master_thesis.maml import adapt_maml_on_episode
        except ImportError as exc:
            raise NotImplementedError(
                f"MAML adapter could not be imported: {exc}. "
                "Check that master_thesis.maml.adapt_maml_on_episode exists."
            ) from exc

        adapted_params = adapt_maml_on_episode(
            model, X_supp_t, y_supp_t,
            inner_lr=inner_lr, inner_steps=inner_steps,
            create_graph=False,  # No outer loop needed at eval time
        )
        buffers = dict(model.named_buffers())
        with torch.no_grad():
            logits = torch.func.functional_call(
                model, (adapted_params, buffers), X_query_t,
            ).squeeze(-1)
            y_pred = torch.sigmoid(logits).cpu().numpy()

    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Expected one of: zero_shot, finetune, anil, fomaml, maml."
        )

    y_true = y_query_np.astype(int)
    n_query = len(y_true)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "method": method,
        "n_query_rows": n_query,
        "n_query_pos": int((y_true == 1).sum()),
        "n_query_neg": int((y_true == 0).sum()),
        "inner_steps": inner_steps,
        "inner_lr": inner_lr,
    }


# ============================================================================
# 5. build_stage2b_result_row
# ============================================================================

def build_stage2b_result_row(
    config: dict,
    episode: dict,
    prediction_result: dict,
    metrics: dict,
    diagnostics: dict,
    support_accounting: dict,
    episode_idx: int,
) -> dict:
    """Build one long-form episode-level result row.

    Produces a flat dictionary compatible with the existing
    ``EPISODIC_RESULT_ROW_SCHEMA`` plus additional accounting fields for
    Stage 2b analysis.

    Parameters
    ----------
    config : dict
        Resolved experiment config with keys like ``experiment_id``,
        ``init_name``, ``method``, ``n_support_pos``, ``n_support_neg``, etc.
    episode : dict
        Output of ``sample_episode_with_synthetic_support``.
    prediction_result : dict
        Output of ``adapt_and_evaluate``.
    metrics : dict
        Metric dict (e.g. from ``evaluate_on_gold_binary`` or raw).
    diagnostics : dict
        Output of ``compute_realism_diagnostics`` (or empty dict for real-only).
    support_accounting : dict
        Output of ``compute_support_accounting``.
    episode_idx : int
        Episode index within the experiment.

    Returns
    -------
    dict
        One result row ready for pd.DataFrame concatenation.
    """
    aug_info = episode.get("augmentation_info", {})
    pred_probs = prediction_result.get("y_pred", np.array([]))

    # Compute prediction statistics
    pred_mean = float(np.mean(pred_probs)) if len(pred_probs) > 0 else np.nan
    pred_std = float(np.std(pred_probs)) if len(pred_probs) > 0 else np.nan
    collapsed = bool(pred_std < 0.01) if not np.isnan(pred_std) else False

    # Extract metrics — support both pd.DataFrame format and dict format
    if isinstance(metrics, pd.DataFrame):
        m = metrics.iloc[0].to_dict() if len(metrics) > 0 else {}
    else:
        m = metrics

    # Resolve augmentation method fields
    aug_method_requested = config.get("augmentation_method", "none")
    aug_method_actual = aug_info.get("method_used", aug_method_requested)
    fallback_used = aug_info.get("fallback_used", False)
    fallback_reason = aug_info.get("fallback_reason", "")

    # Build the synthetic proportion from config
    syn_prop = config.get("synthetic_proportion", 0.0)
    target_per_class = config.get("target_per_class", None)
    target_effective_k = config.get("target_effective_k", None)

    # Real k
    real_k = min(
        config.get("n_support_pos", 0),
        config.get("n_support_neg", 0),
    )

    row: dict[str, Any] = {
        # Identity
        "experiment_id": config.get("experiment_id", ""),
        "target_department": config.get("target_department", ""),
        "episode_department": episode.get("department", ""),
        "department": episode.get("department", config.get("target_department", "")),
        "init_name": config.get("init_name", ""),
        "method": config.get("method", ""),
        "seed": config.get("seed", SEED),
        "episode_idx": episode_idx,

        # Label budget
        "real_k": real_k,
        "n_support_pos": config.get("n_support_pos", 0),
        "n_support_neg": config.get("n_support_neg", 0),
        "real_k_pos": support_accounting.get("real_k_pos", 0),
        "real_k_neg": support_accounting.get("real_k_neg", 0),
        "target_effective_k": target_effective_k,
        "effective_k_pos": support_accounting.get("effective_k_pos", 0),
        "effective_k_neg": support_accounting.get("effective_k_neg", 0),

        # Augmentation
        "syn_prop": syn_prop,
        "gen_method": aug_method_actual if aug_method_actual != "none" else "none",
        "augmentation_method_requested": aug_method_requested,
        "augmentation_method_actual": aug_method_actual,
        "target_per_class": target_per_class,
        "fallback_used": fallback_used,
        "gen_fallback_used": fallback_used,
        "fallback_reason": fallback_reason if fallback_used else "",

        # Counts
        "n_real_support_rows": support_accounting.get("n_real_support_rows", 0),
        "n_real_support_contracts": support_accounting.get("n_real_support_contracts", 0),
        "n_synthetic_support_rows": support_accounting.get("n_synthetic_support_rows", 0),
        "n_effective_support_rows": support_accounting.get("n_effective_support_rows", 0),
        "n_query_rows": prediction_result.get("n_query_rows", 0),
        "n_query_pos": prediction_result.get("n_query_pos", 0),
        "n_query_neg": prediction_result.get("n_query_neg", 0),

        # Metrics
        "auroc": m.get("gold_auroc", m.get("auroc", np.nan)),
        "ap": m.get("gold_ap", m.get("ap", np.nan)),
        "average_precision": m.get("gold_ap", m.get("average_precision", np.nan)),
        "log_loss": m.get("gold_logloss", m.get("log_loss", np.nan)),
        "brier": m.get("gold_brier", m.get("brier", np.nan)),
        "ece": m.get("gold_ece", m.get("ece", np.nan)),
        "precision_at_5": m.get("gold_precision_at_5", m.get("precision_at_5", np.nan)),
        "precision_at_10": m.get("gold_precision_at_10", m.get("precision_at_10", np.nan)),
        "recall_at_5": m.get("gold_recall_at_5", m.get("recall_at_5", np.nan)),
        "recall_at_10": m.get("gold_recall_at_10", m.get("recall_at_10", np.nan)),
        "ndcg_at_5": m.get("gold_ndcg_at_5", m.get("ndcg_at_5", np.nan)),
        "ndcg_at_10": m.get("gold_ndcg_at_10", m.get("ndcg_at_10", np.nan)),
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "collapsed": collapsed,

        # Diagnostics
        "dist_to_nearest_real_mean": diagnostics.get("dist_to_nearest_real_mean", np.nan),
        "dist_to_nearest_real_p95": diagnostics.get("dist_to_nearest_real_p95", np.nan),
        "ks_max": diagnostics.get("ks_max", np.nan),
        "ks_median": diagnostics.get("ks_median", np.nan),
        "cat_mode_preservation_mean": diagnostics.get("cat_mode_preservation_mean", np.nan),
        "cat_mode_preservation_min": diagnostics.get("cat_mode_preservation_min", np.nan),
    }

    return row


# ============================================================================
# 6. run_stage2b_experiment
# ============================================================================

def _resolve_target_per_class_from_effective_k(
    real_k: int,
    target_effective_k: int | None,
    augmentation_method: str,
) -> int | None:
    """Convert ``target_effective_k`` into a ``target_per_class`` value.

    If ``target_effective_k`` is set and augmentation is active, we want each
    class to have exactly ``target_effective_k`` support rows (real + synthetic).
    This means we request ``target_per_class = target_effective_k``.

    If augmentation is "none", return None (no synthetic rows).
    """
    if target_effective_k is None:
        return None
    if augmentation_method == "none":
        return None
    return target_effective_k


def run_stage2b_experiment(
    config: dict,
    task_df: pd.DataFrame,
    feature_cols: list[str],
    base_model: nn.Module,
    preprocessor: Any,
    device: str | None = None,
    output_dir: Optional[Path] = None,
) -> list[dict]:
    """Run one Stage 2b experiment configuration.

    Expected flow per episode::

        episode = sample_episode_with_synthetic_support(...)
        prediction_result = adapt_and_evaluate(...)
        metrics = evaluate_on_gold_binary(...)
        diagnostics = compute_realism_diagnostics(...)
        support_accounting = compute_support_accounting(...)
        row = build_stage2b_result_row(...)

    Parameters
    ----------
    config : dict
        Resolved experiment config with at least:
        ``method``, ``init_name``, ``n_support_pos``, ``n_support_neg``,
        ``inner_lr``, ``inner_steps``, ``n_repeats``, ``seed``,
        ``augmentation_method``, ``target_department``.
        Optionally: ``synthetic_proportion``, ``target_per_class``,
        ``target_effective_k``, ``categorical_cols``, ``k_neighbors``.
    task_df : pd.DataFrame
        Department-level task table (filtered to the target department).
    feature_cols : list[str]
        Feature column names.
    base_model : nn.Module
        Stage 1 pretrained model (or random init).
    preprocessor
        Fitted sklearn preprocessor.
    device : str | None
        PyTorch device.
    output_dir : Path | None
        When provided, write ``support_accounting.csv`` to this directory
        after all episodes complete.  The directory is created if needed.
        This makes the synthetic-support accounting auditable per experiment.

    Returns
    -------
    list[dict]
        One result row per episode.
    """
    n_repeats = config.get("n_repeats", 20)
    seed = config.get("seed", SEED)
    method = config.get("method", "finetune")
    inner_lr = config.get("inner_lr", 0.01)
    inner_steps = config.get("inner_steps", 5)
    n_support_pos = config.get("n_support_pos", 2)
    n_support_neg = config.get("n_support_neg", 2)
    augmentation_method = config.get("augmentation_method", "none")
    synthetic_proportion = config.get("synthetic_proportion", 0.0)
    target_effective_k = config.get("target_effective_k", None)
    categorical_cols = config.get("categorical_cols", None)
    k_neighbors = config.get("k_neighbors", 5)
    target_col = config.get("target_col", "gold_y")
    department_col = config.get("department_col", "department")
    contract_id_col = config.get("contract_id_col", "contract_id")

    # Resolve target_per_class from target_effective_k
    target_per_class = config.get("target_per_class", None)
    if target_per_class is None:
        target_per_class = _resolve_target_per_class_from_effective_k(
            min(n_support_pos, n_support_neg),
            target_effective_k,
            augmentation_method,
        )

    # Resolve categorical columns
    cat_cols = resolve_categorical_cols(task_df, feature_cols, categorical_cols)

    # Detect numeric columns for diagnostics
    num_cols = [c for c in feature_cols if c not in cat_cols and c in task_df.columns]

    rows: list[dict] = []

    for ep_idx in range(n_repeats):
        ep_seed = seed + ep_idx

        # 1. Sample episode with optional synthetic support
        try:
            episode = sample_episode_with_synthetic_support(
                dept_df=task_df,
                feature_cols=feature_cols,
                department_name=config.get("target_department"),
                target_col=target_col,
                department_col=department_col,
                contract_id_col=contract_id_col,
                n_support_pos=n_support_pos,
                n_support_neg=n_support_neg,
                query_strategy="remainder",
                augmentation_method=augmentation_method,
                synthetic_proportion=synthetic_proportion if target_per_class is None else None,
                target_per_class=target_per_class,
                categorical_cols=cat_cols if cat_cols else None,
                k_neighbors=k_neighbors,
                random_state=ep_seed,
            )
        except Exception as exc:
            logger.warning(
                "Episode %d failed during sampling: %s", ep_idx, exc,
            )
            continue

        support_df = episode["support_df"]
        query_df = episode["query_df"]

        # 2. Adapt and evaluate
        try:
            prediction_result = adapt_and_evaluate(
                method=method,
                base_model=base_model,
                preprocessor=preprocessor,
                episode=episode,
                feature_cols=feature_cols,
                target_col=target_col,
                inner_lr=inner_lr,
                inner_steps=inner_steps,
                device=device,
            )
        except NotImplementedError:
            raise  # Propagate — do not silently skip
        except Exception as exc:
            logger.warning(
                "Episode %d failed during adapt_and_evaluate: %s", ep_idx, exc,
            )
            continue

        # 3. Compute metrics on real query only
        y_true = prediction_result["y_true"]
        y_pred = prediction_result["y_pred"]

        try:
            metrics_df = evaluate_on_gold_binary(
                y_true, y_pred,
                model_name=method,
                k_values=(5, 10),
            )
            metrics_dict = metrics_df.iloc[0].to_dict() if len(metrics_df) > 0 else {}
        except Exception:
            metrics_dict = {}

        # 4. Compute diagnostics (only when there are synthetic rows)
        # Use the authoritative is_synthetic column on support_df directly.
        is_syn_col = support_df["is_synthetic"].astype(bool) if "is_synthetic" in support_df.columns \
            else pd.Series(False, index=support_df.index)
        if is_syn_col.any():
            real_support = support_df[~is_syn_col]
            synth_support = support_df[is_syn_col]
            try:
                diagnostics_result = compute_realism_diagnostics(
                    real_support_df=real_support,
                    synth_support_df=synth_support,
                    num_cols=num_cols,
                    cat_cols=cat_cols,
                    target_col=target_col,
                )
            except Exception:
                diagnostics_result = {}
        else:
            diagnostics_result = {}

        # 5. Support accounting
        accounting = compute_support_accounting(
            support_df, target_col=target_col,
            synthetic_col="is_synthetic",
            contract_id_col=contract_id_col,
        )

        # 6. Build result row
        row = build_stage2b_result_row(
            config=config,
            episode=episode,
            prediction_result=prediction_result,
            metrics=metrics_dict,
            diagnostics=diagnostics_result,
            support_accounting=accounting,
            episode_idx=ep_idx,
        )
        rows.append(row)

    # ── Write support_accounting.csv ───────────────────────────────────────
    # When output_dir is provided, save a per-episode accounting summary so
    # the real/synthetic support split is auditable on disk.
    if output_dir is not None and rows:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _acct_keys = [
            "episode_idx",
            "real_k_pos", "real_k_neg",
            "synthetic_k_pos", "synthetic_k_neg",
            "effective_k_pos", "effective_k_neg",
            "n_real_support_rows", "n_synthetic_support_rows",
            "n_effective_support_rows", "n_real_support_contracts",
            "n_synthetic_support_contracts",
            "gen_method", "syn_prop",
        ]
        _acct_rows = [{k: r.get(k) for k in _acct_keys} for r in rows]
        pd.DataFrame(_acct_rows).to_csv(output_dir / "support_accounting.csv", index=False)

    return rows


# ============================================================================
# 7. run_stage2b_grid
# ============================================================================

def run_stage2b_grid(
    experiment_configs: list[dict],
    task_df: pd.DataFrame,
    feature_cols: list[str],
    base_model: nn.Module,
    preprocessor: Any,
    device: str | None = None,
) -> pd.DataFrame:
    """Run a list of resolved Stage 2b configs and return concatenated results.

    Parameters
    ----------
    experiment_configs : list[dict]
        Each dict is a resolved Stage 2b experiment config.
    task_df : pd.DataFrame
        Department-level task table.
    feature_cols : list[str]
        Feature column names.
    base_model : nn.Module
        Stage 1 pretrained model.
    preprocessor
        Fitted preprocessor.
    device : str | None
        PyTorch device.

    Returns
    -------
    pd.DataFrame
        Long-form results with one row per episode.
    """
    all_rows: list[dict] = []

    for i, cfg in enumerate(experiment_configs):
        exp_id = cfg.get("experiment_id", f"stage2b_{i:03d}")
        method = cfg.get("method", "?")
        real_k = min(cfg.get("n_support_pos", 0), cfg.get("n_support_neg", 0))
        aug = cfg.get("augmentation_method", "none")
        logger.info(
            "Running config %d/%d: %s | method=%s | real_k=%d | aug=%s",
            i + 1, len(experiment_configs), exp_id, method, real_k, aug,
        )

        rows = run_stage2b_experiment(
            config=cfg,
            task_df=task_df,
            feature_cols=feature_cols,
            base_model=base_model,
            preprocessor=preprocessor,
            device=device,
        )
        all_rows.extend(rows)

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "resolve_categorical_cols",
    "build_department_eligibility_table",
    "compute_support_accounting",
    "adapt_and_evaluate",
    "build_stage2b_result_row",
    "run_stage2b_experiment",
    "run_stage2b_grid",
]
