"""
explainability.py — SHAP explainability and representation diagnostics for Stage 2.

This module is the Stage 3 backend for the thesis. It operates on a trained
TabularMLP (Stage 1 weak-pretrained or a Stage 2 adapted checkpoint) and the
Stage 1 preprocessor. All heavy logic lives here so the dashboard notebook
remains a thin controller.

Public responsibilities
-----------------------
1. Compute SHAP values for the MLP (DeepExplainer preferred, KernelExplainer fallback).
2. Aggregate SHAP values to the data-view level using a prefix-based feature map.
3. Compute pre/post-adaptation SHAP deltas (Stage 1 MLP vs Stage 2 adapted model).
4. Extract hidden-layer embeddings from the MLP for representation diagnostics.
5. Project embeddings to 2D with PCA or t-SNE (UMAP used if available).
6. Measure pre/post-adaptation embedding shift magnitudes.

Design notes
------------
- All functions are stateless and deterministic given a seed.
- No Stage 2 gold labels are required to compute SHAP values or embeddings.
  Gold labels are only used as OPTIONAL colouring when plotting; the heavy
  compute pipeline here is gold-label free by construction.
- TabularMLP uses BatchNorm1d, which makes DeepExplainer sensitive to batch
  statistics. We always call ``model.eval()`` before explaining.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# View mapping
# -----------------------------------------------------------------------------

# Prefix-based view map. Matches the column naming used across Stage 0/1.
# Unknown features fall back to "other" so view-level aggregation is total.
VIEW_PREFIX_MAP: List[Tuple[str, str]] = [
    ("fin_", "financial"),
    ("esg_", "esg"),
    ("news_", "news"),
    ("market_", "market"),
    ("stock_", "market"),
    ("lpi_", "macro_logistics"),
    ("macro_", "macro_logistics"),
    ("logistics_", "macro_logistics"),
    ("supplier_", "contract_core"),
    ("company_", "contract_core"),
    ("contract_", "contract_core"),
    ("department_", "contract_core"),
    ("observation_", "contract_core"),
    ("missing_", "availability"),
    ("available_", "availability"),
    ("is_", "availability"),
    ("has_", "availability"),
]


def build_feature_view_map(
    feature_names: Sequence[str],
    overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Map each feature name to a view using prefix heuristics, with optional overrides.

    Parameters
    ----------
    feature_names:
        Ordered list of input feature columns (post one-hot or post-preprocessor).
    overrides:
        Optional per-feature override mapping.

    Returns
    -------
    Dict[str, str]
        Mapping feature_name -> view_name. Unknown features map to 'other'.
    """
    overrides = overrides or {}
    mapping: Dict[str, str] = {}
    for feat in feature_names:
        if feat in overrides:
            mapping[feat] = overrides[feat]
            continue

        lower = str(feat).lower()
        view = "other"
        # availability indicators are often suffix-based too
        if (
            lower.endswith("_available")
            or lower.endswith("_missing")
            or lower.endswith("_is_available")
        ):
            view = "availability"
        else:
            for prefix, tag in VIEW_PREFIX_MAP:
                if lower.startswith(prefix):
                    view = tag
                    break
        mapping[feat] = view
    return mapping


# -----------------------------------------------------------------------------
# Preprocessor helpers
# -----------------------------------------------------------------------------

def transform_to_dense(preprocessor: Any, df: pd.DataFrame, feature_cols: Sequence[str]) -> np.ndarray:
    """Run a fitted Stage 1 preprocessor on df[feature_cols] and return a dense float32 matrix."""
    X = preprocessor.transform(df[list(feature_cols)])
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def get_post_transform_feature_names(
    preprocessor: Any,
    input_feature_cols: Sequence[str],
) -> List[str]:
    """
    Best-effort resolution of post-preprocessor column names.

    Falls back to ``feat_0, feat_1, ...`` if the preprocessor does not expose
    ``get_feature_names_out``.
    """
    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            return [str(c) for c in preprocessor.get_feature_names_out(list(input_feature_cols))]
        except Exception:
            pass
    try:
        n_out = preprocessor.transform(
            pd.DataFrame(
                np.zeros((1, len(input_feature_cols))),
                columns=list(input_feature_cols),
            )
        ).shape[1]
    except Exception:
        n_out = len(input_feature_cols)
    return [f"feat_{i}" for i in range(n_out)]


# -----------------------------------------------------------------------------
# Embedding extraction
# -----------------------------------------------------------------------------

def extract_mlp_embeddings(
    model: nn.Module,
    X: np.ndarray,
    layer: str = "penultimate",
    device: Optional[torch.device] = None,
    batch_size: int = 1024,
) -> np.ndarray:
    """
    Forward a numpy feature matrix through a TabularMLP and return hidden activations.

    The model is expected to expose ``model.net`` as a ``nn.Sequential``. The
    ``penultimate`` layer is the last ReLU activation before the final linear
    head (``model.net[:-1]``).

    Parameters
    ----------
    model:
        Trained TabularMLP.
    X:
        Input matrix (n_samples, n_features_post_transform).
    layer:
        'penultimate' returns the output of ``model.net[:-1]``.
        'input' returns the raw input.
        'logit' returns the final logit.
    device:
        Optional torch device.

    Returns
    -------
    np.ndarray
        (n_samples, embedding_dim) float32 array.
    """
    if layer not in {"penultimate", "input", "logit"}:
        raise ValueError(f"Unknown layer: {layer}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if layer == "input":
        return np.asarray(X, dtype=np.float32)

    if not hasattr(model, "net"):
        raise AttributeError("Model does not expose a .net sequential block.")

    outputs: List[np.ndarray] = []
    X = np.asarray(X, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start : start + batch_size]).to(device)
            if layer == "logit":
                out = model(batch).cpu().numpy()
            else:
                # penultimate = all layers except the last linear head
                feats = batch
                for m in list(model.net)[:-1]:
                    feats = m(feats)
                out = feats.cpu().numpy()
            outputs.append(out)

    return np.concatenate(outputs, axis=0)


# -----------------------------------------------------------------------------
# 2D projection
# -----------------------------------------------------------------------------

@dataclass
class ProjectionResult:
    coords: np.ndarray  # (n, 2)
    method: str
    n_components: int
    extra: Dict[str, Any]


def compute_2d_projection(
    X: np.ndarray,
    method: str = "pca",
    random_state: int = 42,
    perplexity: float = 30.0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> ProjectionResult:
    """
    Project an (n, d) matrix to 2D using PCA, t-SNE, or UMAP (if installed).

    t-SNE perplexity auto-caps at ``(n_samples - 1) / 3`` to avoid sklearn errors.
    UMAP is attempted only if the ``umap-learn`` package is importable.
    """
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    method = method.lower()

    if n < 3:
        raise ValueError(f"compute_2d_projection requires at least 3 samples, got {n}.")

    if method == "pca":
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=random_state)
        coords = pca.fit_transform(X)
        return ProjectionResult(
            coords=coords,
            method="pca",
            n_components=2,
            extra={"explained_variance_ratio": pca.explained_variance_ratio_.tolist()},
        )

    if method == "tsne":
        from sklearn.manifold import TSNE

        # sklearn requires perplexity < n_samples
        eff_perp = float(min(perplexity, max(5.0, (n - 1) / 3.0)))
        tsne = TSNE(
            n_components=2,
            perplexity=eff_perp,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        )
        coords = tsne.fit_transform(X)
        return ProjectionResult(
            coords=coords,
            method="tsne",
            n_components=2,
            extra={"perplexity": eff_perp},
        )

    if method == "umap":
        try:
            import umap  # noqa: WPS433 (intentional optional import)
        except ImportError as e:
            warnings.warn(
                f"umap-learn not available ({e}); falling back to PCA.",
                RuntimeWarning,
                stacklevel=2,
            )
            return compute_2d_projection(X, method="pca", random_state=random_state)

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(n_neighbors, max(2, n - 1)),
            min_dist=min_dist,
            random_state=random_state,
        )
        coords = reducer.fit_transform(X)
        return ProjectionResult(
            coords=coords,
            method="umap",
            n_components=2,
            extra={"n_neighbors": n_neighbors, "min_dist": min_dist},
        )

    raise ValueError(f"Unknown projection method: {method}")


def compute_embedding_shift(
    embeddings_pre: np.ndarray,
    embeddings_post: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Row-wise distance between pre/post-adaptation embeddings.

    Parameters
    ----------
    embeddings_pre:
        (n, d) pre-adaptation embeddings.
    embeddings_post:
        (n, d) post-adaptation embeddings.
    metric:
        'euclidean' or 'cosine'.

    Returns
    -------
    np.ndarray
        (n,) array of row-wise shifts.
    """
    pre = np.asarray(embeddings_pre, dtype=np.float32)
    post = np.asarray(embeddings_post, dtype=np.float32)
    if pre.shape != post.shape:
        raise ValueError(f"Shape mismatch: pre={pre.shape}, post={post.shape}")

    if metric == "euclidean":
        return np.linalg.norm(post - pre, axis=1)
    if metric == "cosine":
        dot = (pre * post).sum(axis=1)
        norms = np.linalg.norm(pre, axis=1) * np.linalg.norm(post, axis=1)
        norms = np.where(norms == 0, 1e-12, norms)
        return 1.0 - dot / norms
    raise ValueError(f"Unknown metric: {metric}")


# -----------------------------------------------------------------------------
# SHAP value computation
# -----------------------------------------------------------------------------

@dataclass
class ShapResult:
    shap_values: np.ndarray        # (n_explain, n_features) in logit space by default
    base_value: float              # expected logit on background
    feature_names: List[str]       # post-transform feature names
    explainer: str                 # 'deep' | 'kernel' | 'gradient'
    prob_shap: Optional[np.ndarray] = None  # optional: SHAP values in probability space

    def as_frame(self) -> pd.DataFrame:
        """Return a long-form DataFrame: (feature, sample_idx, shap_value)."""
        rows = []
        for i in range(self.shap_values.shape[0]):
            for j, feat in enumerate(self.feature_names):
                rows.append({"sample_idx": i, "feature": feat, "shap_value": float(self.shap_values[i, j])})
        return pd.DataFrame(rows)

    def mean_abs(self) -> pd.DataFrame:
        """Global importance: mean(|SHAP|) per feature."""
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        return (
            pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )


def _clone_mlp_for_shap(model: nn.Module) -> nn.Module:
    """
    Return a detached eval-mode copy of the model with a sigmoid head appended.

    SHAP's DeepExplainer works more cleanly on a model that returns probabilities.
    BatchNorm layers are put into eval mode so running statistics are used.
    """
    import copy as _copy

    wrapped = _copy.deepcopy(model)
    wrapped.eval()
    # Ensure BatchNorm uses running stats, not batch stats
    for m in wrapped.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()
    return wrapped


class _SigmoidHead(nn.Module):
    """Wrap an MLP so its output is sigmoid(logit). Used for prob-space SHAP."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.base(x))


def compute_shap_values(
    model: nn.Module,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: Sequence[str],
    method: str = "deep",
    device: Optional[torch.device] = None,
    background_size: int = 100,
    kernel_nsamples: str | int = "auto",
    random_state: int = 42,
    output_space: str = "logit",
) -> ShapResult:
    """
    Compute SHAP values for a TabularMLP.

    Parameters
    ----------
    model:
        Trained TabularMLP (eval mode, no gradient required).
    X_background:
        Background distribution for SHAP (usually a sample of Stage 1 training data).
    X_explain:
        Rows to explain (e.g., target-department contracts or a repeated-episode query set).
    feature_names:
        Post-transform feature names, length == X.shape[1].
    method:
        'deep' tries DeepExplainer, falls back to 'gradient', then 'kernel'.
        'gradient' uses GradientExplainer.
        'kernel' uses model-agnostic KernelExplainer (slower, robust to BatchNorm quirks).
    background_size:
        Maximum number of background samples used to summarise the distribution.
    output_space:
        'logit' (default) or 'prob'. 'prob' wraps the model in a sigmoid first.
    random_state:
        Used when subsampling the background.

    Returns
    -------
    ShapResult
    """
    import shap  # local import so importing this module is cheap

    if output_space not in {"logit", "prob"}:
        raise ValueError(f"output_space must be 'logit' or 'prob', got {output_space!r}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_background = np.asarray(X_background, dtype=np.float32)
    X_explain = np.asarray(X_explain, dtype=np.float32)
    if X_background.shape[1] != X_explain.shape[1]:
        raise ValueError(
            f"Feature dim mismatch: background={X_background.shape[1]}, explain={X_explain.shape[1]}"
        )
    if len(feature_names) != X_explain.shape[1]:
        raise ValueError(
            f"feature_names length {len(feature_names)} != X_explain.shape[1] {X_explain.shape[1]}"
        )

    rng = np.random.default_rng(random_state)
    if X_background.shape[0] > background_size:
        idx = rng.choice(X_background.shape[0], size=background_size, replace=False)
        X_background = X_background[idx]

    wrapped = _clone_mlp_for_shap(model).to(device)
    if output_space == "prob":
        wrapped = _SigmoidHead(wrapped).to(device)
        wrapped.eval()

    method = method.lower()
    attempted: List[str] = []
    last_err: Optional[Exception] = None

    # Preferred order: deep -> gradient -> kernel
    order = [method]
    if method == "deep":
        order = ["deep", "gradient", "kernel"]
    elif method == "gradient":
        order = ["gradient", "kernel"]

    for attempt in order:
        attempted.append(attempt)
        try:
            if attempt == "deep":
                bg_t = torch.from_numpy(X_background).to(device)
                explainer = shap.DeepExplainer(wrapped, bg_t)
                x_t = torch.from_numpy(X_explain).to(device)
                sv = explainer.shap_values(x_t, check_additivity=False)
                base_val = float(np.asarray(explainer.expected_value).ravel()[0])
            elif attempt == "gradient":
                bg_t = torch.from_numpy(X_background).to(device)
                explainer = shap.GradientExplainer(wrapped, bg_t)
                x_t = torch.from_numpy(X_explain).to(device)
                sv = explainer.shap_values(x_t)
                with torch.no_grad():
                    base_val = float(wrapped(bg_t).mean().cpu().numpy())
            else:  # kernel
                def _predict(x_np: np.ndarray) -> np.ndarray:
                    with torch.no_grad():
                        t = torch.from_numpy(np.asarray(x_np, dtype=np.float32)).to(device)
                        out = wrapped(t).cpu().numpy().ravel()
                    return out

                explainer = shap.KernelExplainer(_predict, X_background)
                nsamples = kernel_nsamples
                sv = explainer.shap_values(X_explain, nsamples=nsamples)
                base_val = float(np.asarray(explainer.expected_value).ravel()[0])

            # Normalise SHAP outputs to (n, f)
            if isinstance(sv, list):
                sv = sv[0]
            sv = np.asarray(sv)
            if sv.ndim == 3:  # (n, f, 1) for some versions
                sv = sv.squeeze(-1)
            if sv.shape != X_explain.shape:
                raise ValueError(
                    f"Unexpected SHAP output shape {sv.shape}, expected {X_explain.shape}"
                )

            return ShapResult(
                shap_values=sv.astype(np.float32),
                base_value=base_val,
                feature_names=list(feature_names),
                explainer=attempt,
            )

        except Exception as exc:  # noqa: BLE001 — fall-through by design
            last_err = exc
            warnings.warn(
                f"SHAP explainer '{attempt}' failed ({exc.__class__.__name__}: {exc}); trying next.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

    raise RuntimeError(
        f"All SHAP explainers failed. Attempted: {attempted}. Last error: {last_err}"
    )


# -----------------------------------------------------------------------------
# Aggregation and deltas
# -----------------------------------------------------------------------------

def aggregate_shap_by_view(
    shap_result: ShapResult,
    view_map: Dict[str, str],
) -> pd.DataFrame:
    """
    Aggregate SHAP importance to the data-view level.

    Returns a DataFrame with columns ``['view', 'mean_abs_shap', 'n_features']``.
    Views are sorted by mean |SHAP|, descending.
    """
    mean_abs = np.abs(shap_result.shap_values).mean(axis=0)
    rows = []
    for feat, imp in zip(shap_result.feature_names, mean_abs):
        rows.append({"feature": feat, "view": view_map.get(feat, "other"), "mean_abs_shap": float(imp)})
    feat_df = pd.DataFrame(rows)

    view_df = (
        feat_df.groupby("view", as_index=False)
        .agg(mean_abs_shap=("mean_abs_shap", "sum"), n_features=("feature", "count"))
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return view_df


def compute_shap_delta(
    shap_pre: ShapResult,
    shap_post: ShapResult,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """
    Per-feature delta of mean-absolute SHAP value: post minus pre.

    A positive delta means the feature became MORE influential after adaptation.
    """
    if list(shap_pre.feature_names) != list(shap_post.feature_names):
        raise ValueError("shap_pre and shap_post must have identical feature names.")

    pre_imp = np.abs(shap_pre.shap_values).mean(axis=0)
    post_imp = np.abs(shap_post.shap_values).mean(axis=0)
    df = pd.DataFrame(
        {
            "feature": shap_pre.feature_names,
            "pre_mean_abs_shap": pre_imp,
            "post_mean_abs_shap": post_imp,
            "delta_importance": post_imp - pre_imp,
        }
    )
    df["abs_delta"] = df["delta_importance"].abs()
    df = df.sort_values("abs_delta", ascending=False).reset_index(drop=True)
    if top_n is not None:
        df = df.head(top_n)
    return df


def shap_feature_ranking(
    shap_result: ShapResult,
    top_n: int = 20,
) -> pd.DataFrame:
    """Return the top-N features by mean-absolute SHAP value."""
    return shap_result.mean_abs().head(top_n).reset_index(drop=True)


# -----------------------------------------------------------------------------
# End-to-end convenience helpers
# -----------------------------------------------------------------------------

def run_full_shap_pipeline(
    model: nn.Module,
    preprocessor: Any,
    background_df: pd.DataFrame,
    explain_df: pd.DataFrame,
    feature_cols: Sequence[str],
    method: str = "deep",
    background_size: int = 100,
    device: Optional[torch.device] = None,
    overrides: Optional[Dict[str, str]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    One-call helper that returns SHAP values, view-level aggregation, and rankings.

    Returns
    -------
    dict with keys:
        - 'shap_result' : ShapResult
        - 'feature_ranking' : DataFrame (top features)
        - 'view_importance' : DataFrame (view-level aggregation)
        - 'view_map' : Dict[str, str]
    """
    X_background = transform_to_dense(preprocessor, background_df, feature_cols)
    X_explain = transform_to_dense(preprocessor, explain_df, feature_cols)
    feature_names = get_post_transform_feature_names(preprocessor, feature_cols)

    shap_res = compute_shap_values(
        model=model,
        X_background=X_background,
        X_explain=X_explain,
        feature_names=feature_names,
        method=method,
        device=device,
        background_size=background_size,
        random_state=random_state,
    )

    view_map = build_feature_view_map(feature_names, overrides=overrides)
    view_imp = aggregate_shap_by_view(shap_res, view_map)
    ranking = shap_feature_ranking(shap_res, top_n=25)

    return {
        "shap_result": shap_res,
        "feature_ranking": ranking,
        "view_importance": view_imp,
        "view_map": view_map,
    }


def run_pre_post_shap_pipeline(
    model_pre: nn.Module,
    model_post: nn.Module,
    preprocessor: Any,
    background_df: pd.DataFrame,
    explain_df: pd.DataFrame,
    feature_cols: Sequence[str],
    method: str = "deep",
    background_size: int = 100,
    device: Optional[torch.device] = None,
    overrides: Optional[Dict[str, str]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Compute SHAP for the pre-adaptation (Stage 1) and post-adaptation (Stage 2)
    models on the same background/explain rows, and return side-by-side artifacts.
    """
    X_background = transform_to_dense(preprocessor, background_df, feature_cols)
    X_explain = transform_to_dense(preprocessor, explain_df, feature_cols)
    feature_names = get_post_transform_feature_names(preprocessor, feature_cols)
    view_map = build_feature_view_map(feature_names, overrides=overrides)

    shap_pre = compute_shap_values(
        model=model_pre,
        X_background=X_background,
        X_explain=X_explain,
        feature_names=feature_names,
        method=method,
        device=device,
        background_size=background_size,
        random_state=random_state,
    )
    shap_post = compute_shap_values(
        model=model_post,
        X_background=X_background,
        X_explain=X_explain,
        feature_names=feature_names,
        method=method,
        device=device,
        background_size=background_size,
        random_state=random_state,
    )

    delta = compute_shap_delta(shap_pre, shap_post)
    view_pre = aggregate_shap_by_view(shap_pre, view_map).rename(
        columns={"mean_abs_shap": "pre_mean_abs_shap"}
    )
    view_post = aggregate_shap_by_view(shap_post, view_map).rename(
        columns={"mean_abs_shap": "post_mean_abs_shap"}
    )
    view_delta = view_pre.merge(view_post, on=["view", "n_features"], how="outer").fillna(0.0)
    view_delta["delta_importance"] = view_delta["post_mean_abs_shap"] - view_delta["pre_mean_abs_shap"]
    view_delta = view_delta.sort_values("delta_importance", ascending=False).reset_index(drop=True)

    return {
        "shap_pre": shap_pre,
        "shap_post": shap_post,
        "feature_delta": delta,
        "view_delta": view_delta,
        "view_map": view_map,
    }
