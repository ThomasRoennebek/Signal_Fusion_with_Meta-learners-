"""
anil.py — Stage 2 ANIL implementation.

ANIL (Almost No Inner Loop) adapts only the final classification head during
the inner loop while keeping the feature extractor fixed. This module provides
two public functions used by the Stage 2 backend:

- meta_train_anil(...)
- evaluate_anil_on_target_episodes(...)

Both functions return standardized dictionaries so the shared Stage 2 runner can
save metrics, predictions, and training history consistently across methods.

Configurable clipping (added for MAML stabilization):
- gradient_clip_norm: outer-loop gradient clip max_norm.
    1.0 = legacy default; None = no outer clipping.
- inner_grad_clip: accepted for API compatibility but ignored (ANIL adapts
    only the final head; inner gradients are inherently stable).
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from master_thesis.episode_sampler import sample_support_query_split
from master_thesis.metrics import evaluate_on_gold_binary


def _prepare_tensors(
    support_df: pd.DataFrame,
    query_df: pd.DataFrame,
    feature_cols: List[str],
    preprocessor: Any,
    target_col: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
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


def adapt_anil_on_episode(
    model: nn.Module,
    X_support: torch.Tensor,
    y_support: torch.Tensor,
    inner_lr: float = 0.01,
    inner_steps: int = 3,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Adapt only the final head parameters functionally."""
    model.eval()

    with torch.no_grad():
        support_features = X_support
        for layer in model.net[:-1]:
            support_features = layer(support_features)

    head_weight = model.net[-1].weight.clone()
    head_bias = model.net[-1].bias.clone() if getattr(model.net[-1], "bias", None) is not None else None

    criterion = nn.BCEWithLogitsLoss()

    for _ in range(inner_steps):
        logits = F.linear(support_features, head_weight, head_bias)
        loss = criterion(logits, y_support)
        # Skip gradient update if the support loss is non-finite to prevent
        # Inf/NaN from propagating through the head parameters.
        if not torch.isfinite(loss):
            break

        variables = [head_weight]
        if head_bias is not None:
            variables.append(head_bias)

        grads = torch.autograd.grad(loss, variables, create_graph=True)

        head_weight = head_weight - inner_lr * grads[0]
        if head_bias is not None:
            head_bias = head_bias - inner_lr * grads[1]

    return head_weight, head_bias


def meta_train_anil(
    model: nn.Module,
    preprocessor: Any,
    task_df: pd.DataFrame,
    feature_cols: List[str],
    meta_train_departments: List[str],
    target_col: str = "gold_y",
    department_col: str = "department",
    contract_id_col: str = "contract_id",
    meta_batch_size: int = 4,
    meta_iterations: int = 200,
    inner_lr: float = 0.01,
    outer_lr: float = 0.001,
    inner_steps: int = 3,
    n_support_pos: int = 5,
    n_support_neg: int = 5,
    device: Optional[torch.device] = None,
    random_state: int = 42,
    gradient_clip_norm: Optional[float] = 1.0,
    inner_grad_clip: Optional[float] = 1.0,  # accepted for API compat; unused
) -> Dict[str, Any]:
    """Meta-train ANIL with configurable outer gradient clipping.

    Parameters
    ----------
    gradient_clip_norm : float or None
        Max norm for outer-loop gradient clipping.
        1.0 = legacy default.  None = no outer clipping.
    inner_grad_clip : float or None
        Accepted for API compatibility with _run_meta_method but unused.
        ANIL adapts only the final linear head; inner gradients are stable.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    outer_optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)
    criterion = nn.BCEWithLogitsLoss()

    # Defensive filter: keep only rows with a valid label even though
    # build_department_task_table already removes missing gold_y values.
    train_df = task_df[
        task_df[department_col].isin(meta_train_departments)
        & task_df[target_col].notna()
    ].copy()
    history_rows: List[Dict[str, Any]] = []

    total_skipped = 0
    rng = random.Random(random_state)

    for iteration in range(meta_iterations):
        outer_optimizer.zero_grad()
        valid_query_losses = []
        skipped_before_iteration = total_skipped

        batch_depts = rng.choices(meta_train_departments, k=meta_batch_size)

        for dept in batch_depts:
            dept_df = train_df[train_df[department_col] == dept]
            if len(dept_df) < (n_support_pos + n_support_neg + 2):
                total_skipped += 1
                continue

            try:
                episode = sample_support_query_split(
                    dept_df=dept_df,
                    feature_cols=feature_cols,
                    department_name=dept,
                    target_col=target_col,
                    contract_id_col=contract_id_col,
                    n_support_pos=n_support_pos,
                    n_support_neg=n_support_neg,
                    random_state=random_state + iteration,
                )
            except ValueError:
                total_skipped += 1
                continue

            X_supp, y_supp, X_query, y_query, _ = _prepare_tensors(
                episode["support_df"],
                episode["query_df"],
                feature_cols,
                preprocessor,
                target_col,
                device,
            )

            head_weight, head_bias = adapt_anil_on_episode(
                model=model,
                X_support=X_supp,
                y_support=y_supp,
                inner_lr=inner_lr,
                inner_steps=inner_steps,
            )

            with torch.no_grad():
                query_features = X_query
                for layer in model.net[:-1]:
                    query_features = layer(query_features)

            logits_query = F.linear(query_features, head_weight, head_bias)
            query_loss = criterion(logits_query, y_query)

            if not torch.isfinite(query_loss):
                total_skipped += 1
                continue

            valid_query_losses.append(query_loss)

        if valid_query_losses:
            outer_loss = torch.stack(valid_query_losses).mean()
            outer_loss.backward()

            # ── Diagnostic: outer gradient norm (before clipping) ──
            outer_grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    outer_grad_norm += float(param.grad.norm(2).item() ** 2)
            outer_grad_norm = outer_grad_norm ** 0.5

            # Apply outer gradient clipping (configurable).
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            outer_optimizer.step()

            skipped_this_iteration = total_skipped - skipped_before_iteration
            history_rows.append(
                {
                    "iteration": iteration + 1,
                    "meta_loss": float(outer_loss.item()),
                    "n_valid_tasks": len(valid_query_losses),
                    "n_skipped": total_skipped,              # kept for backward compatibility
                    "n_skipped_this_iteration": skipped_this_iteration,
                    "n_skipped_cumulative": total_skipped,
                    # Diagnostics:
                    "outer_grad_norm": outer_grad_norm,
                }
            )

    total_possible_task_episodes = meta_iterations * meta_batch_size
    diagnostics = {
        "meta_iterations_requested": meta_iterations,
        "iterations_with_valid_tasks": len(history_rows),
        "meta_batch_size": meta_batch_size,
        "total_possible_task_episodes": total_possible_task_episodes,
        "total_skipped_task_episodes": total_skipped,
        "skip_rate": total_skipped / max(1, total_possible_task_episodes),
        "meta_train_departments_used": list(meta_train_departments),
        "gradient_clip_norm": gradient_clip_norm,
    }

    return {
        "method": "anil",
        "status": "success",
        "history": pd.DataFrame(history_rows),
        "diagnostics": diagnostics,
        "model": model,
    }


def evaluate_anil_on_target_episodes(
    model: nn.Module,
    preprocessor: Any,
    target_episodes: List[Dict[str, Any]],
    feature_cols: List[str],
    target_col: str = "gold_y",
    inner_lr: float = 0.01,
    target_inner_steps: int = 5,
    device: Optional[torch.device] = None,
    inner_grad_clip: Optional[float] = 1.0,  # accepted for API compat; unused
) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    metric_rows: List[pd.DataFrame] = []
    prediction_rows: List[pd.DataFrame] = []

    for ep_idx, episode in enumerate(target_episodes):
        X_supp, y_supp, X_query, _, y_query_np = _prepare_tensors(
            episode["support_df"],
            episode["query_df"],
            feature_cols,
            preprocessor,
            target_col,
            device,
        )

        head_weight, head_bias = adapt_anil_on_episode(
            model=model,
            X_support=X_supp,
            y_support=y_supp,
            inner_lr=inner_lr,
            inner_steps=target_inner_steps,
        )

        with torch.no_grad():
            query_features = X_query
            for layer in model.net[:-1]:
                query_features = layer(query_features)
            logits = F.linear(query_features, head_weight, head_bias)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()

        # Guard 1: replace any NaN/Inf probs with 0.5 (uninformative).
        if not np.all(np.isfinite(probs)):
            probs = np.where(np.isfinite(probs), probs, 0.5)

        # Guard 2: filter rows where the gold label itself is NaN/missing.
        # The query_df may contain unlabelled contracts (gold_y = NaN) that
        # are not valid evaluation targets — passing NaN y_true to sklearn
        # raises "Input contains NaN".
        valid_mask = np.isfinite(y_query_np)
        y_query_eval = y_query_np[valid_mask].astype(int)
        probs_eval = probs[valid_mask]

        if len(y_query_eval) == 0:
            continue  # no labelled contracts in this episode's query set

        metrics = evaluate_on_gold_binary(y_query_eval, probs_eval, model_name="ANIL")
        metrics["episode_idx"] = ep_idx
        metric_rows.append(metrics)

        query_df = episode["query_df"].copy()
        pred_df = pd.DataFrame(
            {
                "episode_idx": ep_idx,
                "repeat_idx": episode.get("repeat_idx", ep_idx),
                "contract_id": query_df["contract_id"].astype(str).to_numpy() if "contract_id" in query_df.columns else None,
                "department": query_df["department"].astype(str).to_numpy() if "department" in query_df.columns else None,
                "y_true": y_query_np,
                "y_prob": probs,
                "method": "anil",
            }
        )
        prediction_rows.append(pred_df)

    if not metric_rows:
        return {
            "method": "anil",
            "status": "failed",
            "n_target_episodes": len(target_episodes),
            "n_evaluated_episodes": 0,
            "n_skipped_episodes": len(target_episodes),
            "raw_metrics": pd.DataFrame(),
            "predictions": pd.DataFrame(),
        }

    return {
        "method": "anil",
        "status": "success",
        "n_target_episodes": len(target_episodes),
        "n_evaluated_episodes": len(metric_rows),
        "n_skipped_episodes": len(target_episodes) - len(metric_rows),
        "raw_metrics": pd.concat(metric_rows, ignore_index=True),
        "predictions": pd.concat(prediction_rows, ignore_index=True),
    }
