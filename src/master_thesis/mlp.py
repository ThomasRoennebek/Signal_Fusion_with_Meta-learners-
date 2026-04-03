"""
mlp.py — Reusable PyTorch utilities for tabular MLP training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducible experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TabularMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int = 128,
        hidden_dim_2: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class MLPTrainingConfig:
    n_epochs: int = 50
    patience: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-5
    train_batch_size: int = 128
    val_batch_size: int = 256
    hidden_dim_1: int = 128
    hidden_dim_2: int = 64
    dropout: float = 0.2


def _to_dense_if_needed(X):
    """
    Convert sparse matrix to dense if needed.
    """
    if hasattr(X, "toarray"):
        return X.toarray()
    return X


def build_tabular_tensors(
    preprocessor,
    X_train,
    X_val,
    y_train,
    y_val,
):
    """
    Fit/transform tabular data and convert to PyTorch tensors.
    """
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)

    X_train_proc = _to_dense_if_needed(X_train_proc)
    X_val_proc = _to_dense_if_needed(X_val_proc)

    X_train_tensor = torch.tensor(X_train_proc, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_proc, dtype=torch.float32)

    y_train_tensor = torch.tensor(np.asarray(y_train), dtype=torch.float32).view(-1, 1)
    y_val_tensor = torch.tensor(np.asarray(y_val), dtype=torch.float32).view(-1, 1)

    return X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor


def make_dataloaders(
    X_train_tensor: torch.Tensor,
    X_val_tensor: torch.Tensor,
    y_train_tensor: torch.Tensor,
    y_val_tensor: torch.Tensor,
    train_batch_size: int = 128,
    val_batch_size: int = 256,
    shuffle_train: bool = True,
):
    """
    Create standard train/validation dataloaders.
    """
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=shuffle_train,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


def make_weighted_dataloaders(
    X_train_tensor: torch.Tensor,
    X_val_tensor: torch.Tensor,
    y_train_tensor: torch.Tensor,
    y_val_tensor: torch.Tensor,
    sample_weight,
    train_batch_size: int = 128,
    val_batch_size: int = 256,
):
    """
    Create weighted train dataloader and standard validation dataloader.
    """
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)

    train_weights_tensor = torch.tensor(np.asarray(sample_weight), dtype=torch.double)

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=train_weights_tensor,
        num_samples=len(train_weights_tensor),
        replacement=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=sampler,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


def build_mlp(
    input_dim: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    hidden_dim_1: int = 128,
    hidden_dim_2: int = 64,
    dropout: float = 0.2,
    device: Optional[torch.device] = None,
):
    """
    Build model, loss, optimizer, and device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TabularMLP(
        input_dim=input_dim,
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
        dropout=dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss() # Binary Cross Entropy with Logits Loss
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    return model, criterion, optimizer, device


def evaluate_mlp(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """
    Evaluate MLP on a dataloader.
    """
    model.eval()
    total_loss = 0.0
    preds = []
    targets = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)

            probs = torch.sigmoid(logits)
            preds.append(probs.cpu().numpy())
            targets.append(yb.cpu().numpy())

    preds = np.vstack(preds).ravel()
    targets = np.vstack(targets).ravel()

    avg_loss = total_loss / len(loader.dataset)
    rmse = mean_squared_error(targets, preds) ** 0.5
    mae = mean_absolute_error(targets, preds)

    return avg_loss, rmse, mae


def train_mlp(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_epochs: int = 50,
    patience: int = 5,
    verbose: bool = True,
) -> tuple[nn.Module, pd.DataFrame]:
    """
    Train an MLP with early stopping on validation loss.
    """
    best_val_loss = float("inf")
    best_state_dict: Optional[dict] = None
    patience_counter = 0

    train_losses = []
    val_losses = []
    val_rmses = []
    val_maes = []

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_rmse, val_mae = evaluate_mlp(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_rmses.append(val_rmse)
        val_maes.append(val_mae)

        if verbose:
            print(
                f"Epoch {epoch + 1:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val RMSE: {val_rmse:.4f} | "
                f"Val MAE: {val_mae:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    training_history = pd.DataFrame(
        {
            "epoch": range(1, len(train_losses) + 1),
            "train_loss": train_losses,
            "val_loss": val_losses,
            "val_rmse": val_rmses,
            "val_mae": val_maes,
        }
    )

    return model, training_history


def predict_mlp(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> np.ndarray:
    """
    Predict probabilities from an MLP using sigmoid(logits).
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            preds.append(probs.cpu().numpy())

    return np.vstack(preds).ravel()


def fit_mlp_model(
    X_train,
    X_val,
    y_train,
    y_val,
    preprocessor,
    sample_weight=None,
    config: Optional[MLPTrainingConfig] = None,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    High-level wrapper to fit one MLP experiment with configurable parameters.

    Returns
    -------
    dict
        Bundle containing:
        - model
        - preprocessor
        - device
        - history
        - config
    """
    if config is None:
        config = MLPTrainingConfig()

    set_seed(seed)

    X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor = build_tabular_tensors(
        preprocessor=preprocessor,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
    )

    if sample_weight is None:
        train_loader, val_loader = make_dataloaders(
            X_train_tensor=X_train_tensor,
            X_val_tensor=X_val_tensor,
            y_train_tensor=y_train_tensor,
            y_val_tensor=y_val_tensor,
            train_batch_size=config.train_batch_size,
            val_batch_size=config.val_batch_size,
            shuffle_train=True,
        )
    else:
        train_loader, val_loader = make_weighted_dataloaders(
            X_train_tensor=X_train_tensor,
            X_val_tensor=X_val_tensor,
            y_train_tensor=y_train_tensor,
            y_val_tensor=y_val_tensor,
            sample_weight=sample_weight,
            train_batch_size=config.train_batch_size,
            val_batch_size=config.val_batch_size,
        )

    model, criterion, optimizer, device = build_mlp(
        input_dim=X_train_tensor.shape[1],
        lr=config.lr,
        weight_decay=config.weight_decay,
        hidden_dim_1=config.hidden_dim_1,
        hidden_dim_2=config.hidden_dim_2,
        dropout=config.dropout,
    )

    model, history = train_mlp(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        n_epochs=config.n_epochs,
        patience=config.patience,
        verbose=verbose,
    )

    return {
        "model": model,
        "preprocessor": preprocessor,
        "device": device,
        "history": history,
        "config": asdict(config),
    }


def predict_mlp_bundle(
    mlp_bundle: dict,
    X_input,
) -> np.ndarray:
    """
    Predict from a fitted MLP bundle returned by fit_mlp_model().
    """
    preprocessor = mlp_bundle["preprocessor"]
    model = mlp_bundle["model"]
    device = mlp_bundle["device"]

    X_proc = preprocessor.transform(X_input)
    X_proc = _to_dense_if_needed(X_proc)

    X_tensor = torch.tensor(X_proc, dtype=torch.float32)
    y_dummy = torch.zeros((X_tensor.shape[0], 1), dtype=torch.float32)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tensor, y_dummy),
        batch_size=256,
        shuffle=False,
    )

    return predict_mlp(model=model, loader=loader, device=device)