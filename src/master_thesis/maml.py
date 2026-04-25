"""
maml.py — Stage 2 full MAML implementation.

This module provides:
- meta_train_maml(...)
- evaluate_maml_on_target_episodes(...)

The implementation keeps the current functional adaptation design while
standardizing outputs so the shared Stage 2 runner can save method artifacts
consistently.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.func

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


def _safe_buffers(model: nn.Module, min_var: float = 1e-3) -> Dict[str, torch.Tensor]:
    """Return model buffers with BatchNorm running_var clamped to a safe minimum.

    BatchNorm computes (x - mean) / sqrt(var + eps).  When running_var is
    near-zero (common for sparse or near-constant features after Stage 1
    pre-training), the denominator collapses to ~0.003, amplifying inputs by
    300x and producing Inf logits / exploding gradients.  Clamping var >= 1e-3
    keeps the amplification factor below ~32x, which is within the range that
    gradient clipping can handle cleanly.
    """
    return {
        name: buf.clamp(min=min_var) if "running_var" in name else buf
        for name, buf in model.named_buffers()
    }


def adapt_maml_on_episode(
    model: nn.Module,
    X_support: torch.Tensor,
    y_support: torch.Tensor,
    inner_lr: float = 0.01,
    inner_steps: int = 3,
    create_graph: bool = True,
) -> Dict[str, torch.Tensor]:
    model.eval()
    params = dict(model.named_parameters())
    # Clamp near-zero running_var to prevent BatchNorm from amplifying sparse
    # features into Inf territory during few-shot adaptation.
    buffers = _safe_buffers(model)
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(inner_steps):
        logits = torch.func.functional_call(model, (params, buffers), X_support)
        loss = criterion(logits, y_support)
        # Skip gradient update if the support loss is non-finite to prevent
        # Inf/NaN from propagating through the computation graph.
        if not torch.isfinite(loss):
            break
        grads = torch.autograd.grad(loss, params.values(), create_graph=create_graph, allow_unused=True)

        params = {
            name: (param - inner_lr * torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)) if grad is not None else param
            for (name, param), grad in zip(params.items(), grads)
        }

    return params


def meta_train_maml(
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
    n_support_pos: int = 1,
    n_support_neg: int = 1,
    device: Optional[torch.device] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
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

        # Sample departments without replacement when the pool is large enough,
        # so the same department cannot appear twice in one meta-batch.
        if meta_batch_size <= len(meta_train_departments):
            batch_depts = rng.sample(meta_train_departments, k=meta_batch_size)
        else:
            batch_depts = rng.choices(meta_train_departments, k=meta_batch_size)

        for dept_idx, dept in enumerate(batch_depts):
            dept_df = train_df[train_df[department_col] == dept]
            if len(dept_df) < (n_support_pos + n_support_neg + 2):
                total_skipped += 1
                continue

            try:
                # Use a unique seed per (iteration, department slot) so that
                # different departments in the same meta-batch draw independent
                # support/query splits rather than correlated ones.
                episode = sample_support_query_split(
                    dept_df=dept_df,
                    feature_cols=feature_cols,
                    department_name=dept,
                    target_col=target_col,
                    contract_id_col=contract_id_col,
                    n_support_pos=n_support_pos,
                    n_support_neg=n_support_neg,
                    random_state=random_state + iteration * 997 + dept_idx,
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

            adapted_params = adapt_maml_on_episode(
                model=model,
                X_support=X_supp,
                y_support=y_supp,
                inner_lr=inner_lr,
                inner_steps=inner_steps,
            )

            # Use clamped buffers here too so the outer query forward pass
            # sees the same stable normalization as the inner loop did.
            buffers = _safe_buffers(model)
            logits_query = torch.func.functional_call(model, (adapted_params, buffers), X_query)
            query_loss = criterion(logits_query, y_query)

            if not torch.isfinite(query_loss):
                total_skipped += 1
                continue

            valid_query_losses.append(query_loss)

        if valid_query_losses:
            outer_loss = torch.stack(valid_query_losses).mean()
            outer_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    }

    return {
        "method": "maml",
        "status": "success",
        "history": pd.DataFrame(history_rows),
        "diagnostics": diagnostics,
        "model": model,
    }


def evaluate_maml_on_target_episodes(
    model: nn.Module,
    preprocessor: Any,
    target_episodes: List[Dict[str, Any]],
    feature_cols: List[str],
    target_col: str = "gold_y",
    inner_lr: float = 0.01,
    target_inner_steps: int = 5,
    device: Optional[torch.device] = None,
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

        adapted_params = adapt_maml_on_episode(
            model=model,
            X_support=X_supp,
            y_support=y_supp,
            inner_lr=inner_lr,
            inner_steps=target_inner_steps,
            create_graph=False,
        )

        with torch.no_grad():
            buffers = _safe_buffers(model)
            logits = torch.func.functional_call(model, (adapted_params, buffers), X_query)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()

        # Guard 1: replace any NaN/Inf probs with 0.5 (uninformative).
        # nan_to_num in the adapt loop prevents this in practice, but kept
        # as a belt-and-suspenders safety net.
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

        metrics = evaluate_on_gold_binary(y_query_eval, probs_eval, model_name="MAML")
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
                "method": "maml",
            }
        )
        prediction_rows.append(pred_df)

    if not metric_rows:
        return {
            "method": "maml",
            "status": "failed",
            "n_target_episodes": len(target_episodes),
            "n_evaluated_episodes": 0,
            "n_skipped_episodes": len(target_episodes),
            "raw_metrics": pd.DataFrame(),
            "predictions": pd.DataFrame(),
        }

    return {
        "method": "maml",
        "status": "success",
        "n_target_episodes": len(target_episodes),
        "n_evaluated_episodes": len(metric_rows),
        "n_skipped_episodes": len(target_episodes) - len(metric_rows),
        "raw_metrics": pd.concat(metric_rows, ignore_index=True),
        "predictions": pd.concat(prediction_rows, ignore_index=True),
    }
