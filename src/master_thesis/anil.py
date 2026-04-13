
"""
anil.py — Implementation of ANIL in Stage 2.

ANIL (Almost No Inner Loop) is a meta-learning variant where:
- the feature extractor / encoder is shared across tasks
- the inner-loop adaptation updates only the final classification head
- the outer loop learns an initialization that adapts quickly to new tasks

"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import copy

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from master_thesis.stage2 import build_stage2_model_from_config
from master_thesis.episode_sampler import sample_support_query_split
from master_thesis.metrics import evaluate_on_gold_binary

# Ensure you have a helper globally accessible or redefine a minimal tensor creator here
# For stage2 integration, it's simpler to ask the user to provide a tensor conversion helper
# or we can define a small one inside `anil.py` based on `preprocessor`.

def _prepare_tensors(support_df, query_df, feature_cols, preprocessor, target_col, device):
    import torch
    import numpy as np
    
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


import torch.nn.functional as F

def adapt_anil_on_episode(
    model: nn.Module,
    X_support: torch.Tensor,
    y_support: torch.Tensor,
    inner_lr: float = 0.01,
    inner_steps: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Functional ANIL Adaptation.
    Updates only the Head dynamically without breaking computation graphs!
    """
    model.eval() # Keep batchnorm components frozen for consistency
    
    # Pass features through the frozen body once
    with torch.no_grad():
        supp_features = X_support
        for layer in model.net[:-1]:
            supp_features = layer(supp_features)
            
    # Extract the initial global weights of the Head functionally
    head_weight = model.net[-1].weight.clone()
    head_bias = model.net[-1].bias.clone() if getattr(model.net[-1], 'bias', None) is not None else None

    criterion = nn.BCEWithLogitsLoss()
    
    for _ in range(inner_steps):
        # 1. Forward Pass exactly through the head
        logits = F.linear(supp_features, head_weight, head_bias)
        loss = criterion(logits, y_support)
        
        # 2. Backpropagate manually preserving the graph!
        # create_graph=True explicitly keeps the math connected
        var_list = [head_weight]
        if head_bias is not None: var_list.append(head_bias)
            
        grads = torch.autograd.grad(loss, var_list, create_graph=True)
        
        # 3. Manual SGD Update (New Weight = Old Weight - lr * Gradient)
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
) -> Dict[str, Any]:
    import random
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model.to(device)
    outer_optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)
    criterion = nn.BCEWithLogitsLoss()
    
    history = []
    train_df = task_df[task_df[department_col].isin(meta_train_departments)].copy()

    for iteration in range(meta_iterations):
        outer_optimizer.zero_grad()
        outer_loss = 0.0
        
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

            # Functional Inner Loop
            head_weight, head_bias = adapt_anil_on_episode(model, X_supp, y_supp, inner_lr, inner_steps)
            
            # --- Outer Loop Evaluation ---
            # 1. Pass Queries through the Global Body (this tracks gradients backward directly into the body)
            query_features = X_query
            for layer in model.net[:-1]:
                query_features = layer(query_features)
                
            # 2. Pass those features into the Adapted Head
            logits = F.linear(query_features, head_weight, head_bias)
            
            # Accumulate Total Loss
            outer_loss += criterion(logits, y_query)

        if isinstance(outer_loss, torch.Tensor):
            # Magical PyTorch backpropagation that flawlessly updates both the Global Body AND Global Head!
            outer_loss.backward()
            outer_optimizer.step()
            
            if (iteration + 1) % 10 == 0:
                print(f"ANIL Meta-Iteration {iteration+1:03d} | Global Loss: {outer_loss.item():.4f}")
            history.append({"iteration": iteration + 1, "meta_loss": float(outer_loss.item())})
            
    return {"method": "anil", "status": "success", "history": pd.DataFrame(history), "model": model}


def evaluate_anil_on_target_episodes(
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
        
        # Adapt ANIL on the Target Support set functionally
        head_weight, head_bias = adapt_anil_on_episode(model, X_supp, y_supp, inner_lr, target_inner_steps)
        
        # Evaluate on the Target Query set
        with torch.no_grad():
            query_features = X_query
            for layer in model.net[:-1]:
                query_features = layer(query_features)
                
            logits = F.linear(query_features, head_weight, head_bias)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            
        metrics = evaluate_on_gold_binary(y_query_np, probs, model_name="ANIL")
        metrics["episode_idx"] = ep_idx
        all_metrics.append(metrics)
        
    if len(all_metrics) == 0:
        return {"method": "anil", "status": "failed", "metrics": pd.DataFrame()}
        
    return {
        "method": "anil",
        "status": "success",
        "n_target_episodes": len(target_episodes),
        "raw_metrics": pd.concat(all_metrics, ignore_index=True),
    }