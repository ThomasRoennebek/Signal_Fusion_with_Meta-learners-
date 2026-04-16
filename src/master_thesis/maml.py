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


def adapt_maml_on_episode(
    model: nn.Module,
    X_support: torch.Tensor,
    y_support: torch.Tensor,
    inner_lr: float = 0.01,
    inner_steps: int = 3,
) -> Dict[str, torch.Tensor]:
    model.eval()
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    criterion = nn.BCEWithLogitsLoss()

    for _ in range(inner_steps):
        logits = torch.func.functional_call(model, (params, buffers), X_support)
        loss = criterion(logits, y_support)
        grads = torch.autograd.grad(loss, params.values(), create_graph=True, allow_unused=True)

        params = {
            name: (param - inner_lr * grad) if grad is not None else param
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

    train_df = task_df[task_df[department_col].isin(meta_train_departments)].copy()
    history_rows: List[Dict[str, Any]] = []
    total_skipped = 0
    rng = random.Random(random_state)

    for iteration in range(meta_iterations):
        outer_optimizer.zero_grad()
        valid_query_losses = []

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

            adapted_params = adapt_maml_on_episode(
                model=model,
                X_support=X_supp,
                y_support=y_supp,
                inner_lr=inner_lr,
                inner_steps=inner_steps,
            )

            buffers = dict(model.named_buffers())
            logits_query = torch.func.functional_call(model, (adapted_params, buffers), X_query)
            query_loss = criterion(logits_query, y_query)

            if torch.isnan(query_loss):
                total_skipped += 1
                continue

            valid_query_losses.append(query_loss)

        if valid_query_losses:
            outer_loss = torch.stack(valid_query_losses).mean()
            outer_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            outer_optimizer.step()

            history_rows.append(
                {
                    "iteration": iteration + 1,
                    "meta_loss": float(outer_loss.item()),
                    "n_skipped": total_skipped,
                    "n_valid_tasks": len(valid_query_losses),
                }
            )

    return {
        "method": "maml",
        "status": "success",
        "history": pd.DataFrame(history_rows),
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
        )

        with torch.no_grad():
            buffers = dict(model.named_buffers())
            logits = torch.func.functional_call(model, (adapted_params, buffers), X_query)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()

        metrics = evaluate_on_gold_binary(y_query_np, probs, model_name="MAML")
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
            "n_target_episodes": 0,
            "raw_metrics": pd.DataFrame(),
            "predictions": pd.DataFrame(),
        }

    return {
        "method": "maml",
        "status": "success",
        "n_target_episodes": len(target_episodes),
        "raw_metrics": pd.concat(metric_rows, ignore_index=True),
        "predictions": pd.concat(prediction_rows, ignore_index=True),
    }
