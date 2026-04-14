"""
maml.py — Skeleton implementation for full MAML in Stage 2.

This file currently provides the Stage 2 method interface and placeholder
logic so it can be integrated cleanly into stage2.py.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
import torch.nn.functional as F
import torch.func

from master_thesis.episode_sampler import sample_support_query_split
from master_thesis.metrics import evaluate_on_gold_binary

def _prepare_tensors(support_df, query_df, feature_cols, preprocessor, target_col, device):
    X_supp_raw = support_df[feature_cols]
    X_query_raw = query_df[feature_cols]
    
    X_supp = preprocessor.transform(X_supp_raw)
    X_query = preprocessor.transform(X_query_raw)
    
    if hasattr(X_supp, "toarray"): X_supp = X_supp.toarray()
    if hasattr(X_query, "toarray"): X_query = X_query.toarray()
    
    y_supp = support_df[target_col].to_numpy(dtype=np.float32)
    y_query = query_df[target_col].to_numpy(dtype=np.float32)
    
    X_supp_t = torch.tensor(X_supp, dtype=torch.float32, device=device)
    X_query_t = torch.tensor(X_query, dtype=torch.float32, device=device)
    y_supp_t = torch.tensor(y_supp, dtype=torch.float32, device=device).view(-1, 1)
    y_query_t = torch.tensor(y_query, dtype=torch.float32, device=device).view(-1, 1)
    
    return X_supp_t, y_supp_t, X_query_t, y_query_t, y_query

def adapt_maml_on_episode(
    model: nn.Module,
    X_support: torch.Tensor,
    y_support: torch.Tensor,
    inner_lr: float = 0.01,
    inner_steps: int = 3,
) -> Dict[str, torch.Tensor]:
    """
    Full MAML Functional Adaptation using Second-Order Gradients.
    Unfreezes and updates ALL parameters inside the PyTorch model while
    tracking the massive graph of gradients for the Hessian calculation.
    """
    model.eval() # Keep batch norm stats frozen in inner loop
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    criterion = nn.BCEWithLogitsLoss()
    
    for _ in range(inner_steps):
        # 1. Functional forward pass
        logits = torch.func.functional_call(model, (params, buffers), X_support)
        loss = criterion(logits, y_support)
        
        # 2. Backpropagate with create_graph=True to enable Full Hessian 2nd order derivatives!
        grads = torch.autograd.grad(loss, params.values(), create_graph=True, allow_unused=True)
        
        # 3. Functional SGD Update with differentiable SGD update
        safe_params = {}
        for (name, param), grad in zip(params.items(), grads):
            if grad is not None:
                safe_params[name] = param - inner_lr * grad
            else:
                safe_params[name] = param
                
        params = safe_params
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
    n_support_pos: int = 1, # Kept low to support small background departments!
    n_support_neg: int = 1,
    device: Optional[torch.device] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    import random
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model.to(device)
    outer_optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)
    criterion = nn.BCEWithLogitsLoss()
    
    history = []
    train_df = task_df[task_df[department_col].isin(meta_train_departments)].copy()

    total_skipped = 0
    for iteration in range(meta_iterations):
        outer_optimizer.zero_grad()
        valid_query_losses = []
        
        batch_depts = random.choices(meta_train_departments, k=meta_batch_size)
        
        for dept in batch_depts:
            dept_df = train_df[train_df[department_col] == dept]
            if len(dept_df) < (n_support_pos + n_support_neg + 2):
                continue
                
            try:
                episode = sample_support_query_split(
                    dept_df=dept_df, feature_cols=feature_cols, department_name=dept,
                    target_col=target_col, contract_id_col=contract_id_col,
                    n_support_pos=n_support_pos, n_support_neg=n_support_neg,
                    random_state=random_state + iteration,
                )
            except ValueError:
                continue
                
            X_supp, y_supp, X_query, y_query, _ = _prepare_tensors(
                episode["support_df"], episode["query_df"], feature_cols, preprocessor, target_col, device
            )

            # --- Full MAML Inner Loop ---
            adapted_params = adapt_maml_on_episode(model, X_supp, y_supp, inner_lr, inner_steps)
            
            # --- Full MAML Outer Loop ---
            buffers = dict(model.named_buffers())
            logits = torch.func.functional_call(model, (adapted_params, buffers), X_query)
            
            # Check for NaN in current episode loss
            loss_val = criterion(logits, y_query)
            if torch.isfinite(loss_val):
                valid_query_losses.append(loss_val)
            else:
                total_skipped += 1

        if valid_query_losses:
            outer_loss = torch.stack(valid_query_losses).mean()
            outer_loss.backward()
            
            # Numerical stability: clip outer-loop meta-gradients after backpropagation
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            outer_optimizer.step()
            
            if (iteration + 1) % 10 == 0:
                print(f"MAML Meta-Iteration {iteration+1:03d} | Global Loss: {outer_loss.item():.4f} | Skipped: {total_skipped}")
            history.append({
                "iteration": iteration + 1, 
                "meta_loss": float(outer_loss.item()),
                "n_skipped": total_skipped
            })
        else:
            if (iteration + 1) % 10 == 0:
                print(f"MAML Meta-Iteration {iteration+1:03d} | [WARNING] All episodes skipped due to NaN.")
            
    return {"method": "maml", "status": "success", "history": pd.DataFrame(history), "model": model}

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
    all_metrics = []
    
    for ep_idx, episode in enumerate(target_episodes):
        X_supp, y_supp, X_query, y_query, y_query_np = _prepare_tensors(
            episode["support_df"], episode["query_df"], feature_cols, preprocessor, target_col, device
        )
        
        # Adapt ALL parameters functionally
        adapted_params = adapt_maml_on_episode(model, X_supp, y_supp, inner_lr, target_inner_steps)
        
        with torch.no_grad():
            buffers = dict(model.named_buffers())
            logits = torch.func.functional_call(model, (adapted_params, buffers), X_query)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            
        metrics = evaluate_on_gold_binary(y_query_np, probs, model_name="MAML")
        metrics["episode_idx"] = ep_idx
        all_metrics.append(metrics)
        
    if len(all_metrics) == 0:
        return {"method": "maml", "status": "failed", "metrics": pd.DataFrame()}
        
    return {
        "method": "maml",
        "status": "success",
        "n_target_episodes": len(target_episodes),
        "raw_metrics": pd.concat(all_metrics, ignore_index=True),
    }

