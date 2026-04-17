"""
meta_common.py — Shared helpers for Stage 2 meta-learners.

MAML, FOMAML, and ANIL share a large amount of boilerplate (tensor preparation,
collapse detection, per-iteration deterministic seeding, prediction-frame
construction, history-row formatting). This module consolidates that logic so
the individual method modules can stay small and focused on their adaptation
update rule.

None of the helpers here depend on the gold labels themselves — they only
process whatever `target_col` the caller specifies. This makes them safe to
reuse for label-free diagnostics (e.g. zero-shot baselines evaluated on
Stage-1 weak probabilities, collapse rate on unlabeled query sets, etc.).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Tensor preparation
# ---------------------------------------------------------------------------

def prepare_episode_tensors(
    support_df: pd.DataFrame,
    query_df: pd.DataFrame,
    feature_cols: List[str],
    preprocessor: Any,
    target_col: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """Transform (support_df, query_df) through `preprocessor` and wrap as tensors.

    Mirrors the private `_prepare_tensors` in each learner module, but lives in
    a single place so downstream diagnostics (zero-shot, collapse checks) can
    reuse it without pulling in a specific learner dependency.
    """
    X_supp = preprocessor.transform(support_df[feature_cols])
    X_query = preprocessor.transform(query_df[feature_cols])

    if hasattr(X_supp, "toarray"):
        X_supp = X_supp.toarray()
    if hasattr(X_query, "toarray"):
        X_query = X_query.toarray()

    y_supp_np = support_df[target_col].to_numpy(dtype=np.float32)
    y_query_np = query_df[target_col].to_numpy(dtype=np.float32)

    X_supp_t = torch.tensor(X_supp, dtype=torch.float32, device=device)
    X_query_t = torch.tensor(X_query, dtype=torch.float32, device=device)
    y_supp_t = torch.tensor(y_supp_np, dtype=torch.float32, device=device).view(-1, 1)
    y_query_t = torch.tensor(y_query_np, dtype=torch.float32, device=device).view(-1, 1)

    return X_supp_t, y_supp_t, X_query_t, y_query_t, y_query_np


# ---------------------------------------------------------------------------
# Collapse detection / prediction statistics
# ---------------------------------------------------------------------------

@dataclass
class PredictionStats:
    """Summary of a vector of predicted probabilities on a single query set."""

    n: int
    mean: float
    std: float
    min: float
    max: float
    frac_near_zero: float
    frac_near_one: float
    collapsed: bool
    collapse_reason: Optional[str] = None

    def as_row(self) -> Dict[str, Any]:
        return {
            "n_query": self.n,
            "pred_mean": self.mean,
            "pred_std": self.std,
            "pred_min": self.min,
            "pred_max": self.max,
            "frac_near_zero": self.frac_near_zero,
            "frac_near_one": self.frac_near_one,
            "collapsed": bool(self.collapsed),
            "collapse_reason": self.collapse_reason,
        }


def compute_prediction_stats(
    probs: np.ndarray,
    std_threshold: float = 0.01,
    extreme_mean_low: float = 0.02,
    extreme_mean_high: float = 0.98,
    near_zero_threshold: float = 0.05,
    near_one_threshold: float = 0.95,
) -> PredictionStats:
    """Compute summary statistics and a collapse flag for a prediction vector.

    Collapse is defined as either:
      * predictive standard deviation < ``std_threshold`` (all predictions
        basically the same), or
      * predictive mean outside ``[extreme_mean_low, extreme_mean_high]``
        (the model is stuck predicting one class).
    Both conditions commonly occur when meta-training destabilises the
    initialisation.
    """
    probs = np.asarray(probs).ravel()
    if probs.size == 0:
        return PredictionStats(
            n=0, mean=float("nan"), std=float("nan"),
            min=float("nan"), max=float("nan"),
            frac_near_zero=float("nan"), frac_near_one=float("nan"),
            collapsed=True, collapse_reason="empty_prediction_vector",
        )

    mean = float(np.mean(probs))
    std = float(np.std(probs))
    reason: Optional[str] = None
    collapsed = False

    if std < std_threshold:
        collapsed = True
        reason = f"std<{std_threshold:g}"
    elif mean < extreme_mean_low:
        collapsed = True
        reason = f"mean<{extreme_mean_low:g}"
    elif mean > extreme_mean_high:
        collapsed = True
        reason = f"mean>{extreme_mean_high:g}"

    return PredictionStats(
        n=int(probs.size),
        mean=mean,
        std=std,
        min=float(np.min(probs)),
        max=float(np.max(probs)),
        frac_near_zero=float(np.mean(probs < near_zero_threshold)),
        frac_near_one=float(np.mean(probs > near_one_threshold)),
        collapsed=collapsed,
        collapse_reason=reason,
    )


# ---------------------------------------------------------------------------
# Deterministic per-iteration seeding
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Seed Python, numpy, and torch for deterministic runs.

    This does NOT enable ``torch.backends.cudnn.deterministic = True`` because
    that trades significant speed for bit-exact reproducibility; most of the
    variance in Stage 2 comes from episode sampling which is already
    seed-controlled.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def iteration_seed(base_seed: int, iteration: int) -> int:
    """Return a deterministic per-iteration seed.

    Keeps outer-loop iteration reproducible while varying episode samples.
    """
    return int(base_seed) + int(iteration) * 101


# ---------------------------------------------------------------------------
# Gradient tracking
# ---------------------------------------------------------------------------

def total_grad_norm(parameters, norm_type: float = 2.0) -> float:
    """Compute the total gradient norm of an iterable of parameters.

    Returns 0.0 when no parameter has a gradient yet (e.g. before the first
    backward pass, or on a meta-iteration that skipped all tasks).
    """
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    device = grads[0].device
    norms = torch.stack([torch.norm(g, p=norm_type).to(device) for g in grads])
    return float(torch.norm(norms, p=norm_type).item())


# ---------------------------------------------------------------------------
# Per-iteration history row / logging
# ---------------------------------------------------------------------------

def build_iter_history_row(
    *,
    iteration: int,
    meta_loss: Optional[float],
    n_valid_tasks: int,
    n_skipped_cumulative: int,
    grad_norm: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "iteration": int(iteration),
        "meta_loss": float(meta_loss) if meta_loss is not None else float("nan"),
        "n_valid_tasks": int(n_valid_tasks),
        "n_skipped": int(n_skipped_cumulative),
    }
    if grad_norm is not None:
        row["grad_norm"] = float(grad_norm)
    if extra:
        for k, v in extra.items():
            row[k] = v
    return row


# ---------------------------------------------------------------------------
# Standard prediction-row builder
# ---------------------------------------------------------------------------

def build_episode_prediction_frame(
    *,
    episode: Dict[str, Any],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method_name: str,
    ep_idx: int,
    contract_id_col: str = "contract_id",
    department_col: str = "department",
) -> pd.DataFrame:
    """Construct a per-episode prediction DataFrame with a consistent schema.

    Columns: episode_idx, repeat_idx, contract_id, department, y_true, y_prob, method.
    """
    query_df = episode["query_df"]
    return pd.DataFrame(
        {
            "episode_idx": ep_idx,
            "repeat_idx": episode.get("repeat_idx", ep_idx),
            "contract_id": (
                query_df[contract_id_col].astype(str).to_numpy()
                if contract_id_col in query_df.columns else None
            ),
            "department": (
                query_df[department_col].astype(str).to_numpy()
                if department_col in query_df.columns else None
            ),
            "y_true": np.asarray(y_true).ravel(),
            "y_prob": np.asarray(y_prob).ravel(),
            "method": method_name,
        }
    )


# ---------------------------------------------------------------------------
# Zero-shot (no-adapt) evaluation of a Stage 1 model on target episodes
# ---------------------------------------------------------------------------

def evaluate_zero_shot_on_target_episodes(
    model: nn.Module,
    preprocessor: Any,
    target_episodes: List[Dict[str, Any]],
    feature_cols: List[str],
    target_col: str = "gold_y",
    device: Optional[torch.device] = None,
    method_name: str = "zero_shot",
    contract_id_col: str = "contract_id",
    department_col: str = "department",
    collect_prediction_stats: bool = True,
) -> Dict[str, Any]:
    """Run ``model`` on each target episode's query set *without* any adaptation.

    This is the pure-Stage-1 baseline: feed the preprocessed query features
    through the model as-is, record sigmoid probabilities, and score against
    ``target_col``. Useful as a floor reference for every adaptive method,
    independent of which gold-label analysis is done later.
    """
    from master_thesis.metrics import evaluate_on_gold_binary

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    metric_rows: List[pd.DataFrame] = []
    prediction_rows: List[pd.DataFrame] = []
    pred_stat_rows: List[Dict[str, Any]] = []

    for ep_idx, episode in enumerate(target_episodes):
        _, _, X_query_t, _, y_query_np = prepare_episode_tensors(
            episode["support_df"],
            episode["query_df"],
            feature_cols,
            preprocessor,
            target_col,
            device,
        )

        with torch.no_grad():
            logits = model(X_query_t)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()

        metrics = evaluate_on_gold_binary(y_query_np, probs, model_name=method_name.upper())
        metrics["episode_idx"] = ep_idx
        metric_rows.append(metrics)

        prediction_rows.append(
            build_episode_prediction_frame(
                episode=episode,
                y_true=y_query_np,
                y_prob=probs,
                method_name=method_name,
                ep_idx=ep_idx,
                contract_id_col=contract_id_col,
                department_col=department_col,
            )
        )

        if collect_prediction_stats:
            stats = compute_prediction_stats(probs)
            row = stats.as_row()
            row["episode_idx"] = ep_idx
            row["repeat_idx"] = episode.get("repeat_idx", ep_idx)
            pred_stat_rows.append(row)

    if not metric_rows:
        return {
            "method": method_name,
            "status": "failed",
            "n_target_episodes": 0,
            "raw_metrics": pd.DataFrame(),
            "predictions": pd.DataFrame(),
            "prediction_stats": pd.DataFrame(),
        }

    return {
        "method": method_name,
        "status": "success",
        "n_target_episodes": len(target_episodes),
        "raw_metrics": pd.concat(metric_rows, ignore_index=True),
        "predictions": pd.concat(prediction_rows, ignore_index=True),
        "prediction_stats": pd.DataFrame(pred_stat_rows),
    }


# ---------------------------------------------------------------------------
# Convenience: prediction-stats DataFrame from an existing predictions frame
# ---------------------------------------------------------------------------

def prediction_stats_frame_from_predictions(
    predictions: pd.DataFrame,
    episode_col: str = "episode_idx",
    repeat_col: str = "repeat_idx",
    method_col: str = "method",
    prob_col: str = "y_prob",
) -> pd.DataFrame:
    """Group a predictions frame by episode and return one collapse-diagnostic row each."""
    if predictions is None or predictions.empty:
        return pd.DataFrame()

    group_cols = [c for c in (method_col, episode_col, repeat_col) if c in predictions.columns]
    rows: List[Dict[str, Any]] = []
    for keys, group in predictions.groupby(group_cols, dropna=False):
        key_dict = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        stats = compute_prediction_stats(group[prob_col].to_numpy())
        row = {**key_dict, **stats.as_row()}
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "PredictionStats",
    "build_episode_prediction_frame",
    "build_iter_history_row",
    "compute_prediction_stats",
    "evaluate_zero_shot_on_target_episodes",
    "iteration_seed",
    "prediction_stats_frame_from_predictions",
    "prepare_episode_tensors",
    "seed_everything",
    "total_grad_norm",
]
