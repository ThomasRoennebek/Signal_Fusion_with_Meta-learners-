from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error


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


def evaluate_mlp(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """
    Evaluate an MLP on a dataloader.

    Returns
    -------
    avg_loss : float
        Mean loss over the dataset.
    rmse : float
        RMSE on sigmoid-transformed predictions.
    mae : float
        MAE on sigmoid-transformed predictions.
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

    Returns
    -------
    model : nn.Module
        Best model after restoring the best validation checkpoint.
    training_history : pd.DataFrame
        Epoch-wise training and validation history.
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
    Predict probabilities from an MLP using sigmoid - logits.
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
