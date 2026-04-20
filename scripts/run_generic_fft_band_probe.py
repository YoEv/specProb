"""
Generic FFT-based spectral probing for a single target class on (B, F, T) embeddings.

This is a generalized version of the CLAP-specific
`scripts/run_single_genre_analysis.py` script. It:

- loads an embeddings NPZ with shape (B, F, T) and a label array;
- applies an FFT (or DCT) along the time axis;
- trains a logistic regression probe on:
    * the full spectrum;
    * each contiguous frequency band (e.g. 4 coeffs per band);
    * cumulative bands from low to high;
- computes the learned spectral profile from the full-spectrum probe;
- saves:
    * a JSON metrics file (accuracies, band info, spectral profile);
    * a PNG plot with
        - left: bar chart of ORIG, B0..B{n-1}, AUTO_CMAX/AUTO_TR accuracies;
        - right: spectral profile (learned weights over frequency coeffs).

Can be used for:
- PaSST FMA / ASAP NPZs;
- BEATs FMA / ASAP NPZs;
- or any other (B, F, T) embeddings with a label array.
"""

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# Ensure the project root is on sys.path when invoked as
# `python scripts/run_generic_fft_band_probe.py ...`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.spectral import apply_transform
from src.training.probes import run_probe
from src.visualization.spectral_profile import (
    learned_weight_profile_stats,
    learned_weight_profile_stats_multiclass,
)


warnings.filterwarnings("ignore")


class _TorchLinearProbeModel:
    """Minimal sklearn-like wrapper exposing .coef_."""

    def __init__(self, coef: np.ndarray):
        self.coef_ = coef


def _run_probe_gpu_batched_same_dim(
    X_task_features: list[np.ndarray],
    y_labels: np.ndarray,
    random_state: int,
    test_size: float,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> tuple[list[float], list[_TorchLinearProbeModel]]:
    """Train multiple binary linear probes in parallel on GPU.

    All tasks must have the same feature dimension. This is used for
    band-specific probes where each band has identical width.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime env dependent
        raise RuntimeError(
            "GPU backend requested but PyTorch is not installed in this environment."
        ) from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU batched backend requested but CUDA is not available. "
            "Use --backend cpu or ensure CUDA-enabled torch is installed."
        )
    if not X_task_features:
        return [], []

    feat_dim = X_task_features[0].shape[1]
    if any(x.shape[1] != feat_dim for x in X_task_features):
        raise ValueError("All batched tasks must have the same feature dimension.")

    n_tasks = len(X_task_features)
    task_tensor = np.stack(X_task_features, axis=0).astype(np.float32)  # (K, N, D)
    labels = y_labels.astype(np.int64)

    # Split once and reuse across all tasks for consistency and speed.
    indices = np.arange(labels.shape[0])
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    X_train = task_tensor[:, train_idx, :]  # (K, Ntr, D)
    X_test = task_tensor[:, test_idx, :]  # (K, Nte, D)
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    # Task-wise standardization on GPU.
    X_train_t = torch.from_numpy(X_train).to(device)
    X_test_t = torch.from_numpy(X_test).to(device)
    y_train_t = torch.from_numpy(y_train.astype(np.float32)).to(device)

    mean = X_train_t.mean(dim=1, keepdim=True)
    std = X_train_t.std(dim=1, keepdim=True).clamp_min(1e-6)
    X_train_t = (X_train_t - mean) / std
    X_test_t = (X_test_t - mean) / std

    torch.manual_seed(random_state)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(random_state)

    # One linear head per task.
    W = torch.nn.Parameter(torch.zeros(n_tasks, feat_dim, device=device))
    b = torch.nn.Parameter(torch.zeros(n_tasks, device=device))
    optimizer = torch.optim.AdamW([W, b], lr=lr, weight_decay=weight_decay)

    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    pos_weight = torch.tensor(neg / max(pos, 1.0), device=device, dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    y_train_expand = y_train_t.unsqueeze(0).expand(n_tasks, -1)  # (K, Ntr)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = torch.einsum("knd,kd->kn", X_train_t, W) + b.unsqueeze(1)
        losses = criterion(logits, y_train_expand)  # (K, Ntr)
        loss = losses.mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits_test = torch.einsum("knd,kd->kn", X_test_t, W) + b.unsqueeze(1)
        preds = (torch.sigmoid(logits_test) >= 0.5).to(torch.int64).cpu().numpy()

    accuracies = (preds == y_test.reshape(1, -1)).mean(axis=1).astype(float).tolist()
    coef = W.detach().cpu().numpy()
    models = [_TorchLinearProbeModel(c.reshape(1, -1)) for c in coef]
    return accuracies, models


def _run_probe_gpu_batched_prefix(
    X_full_features: np.ndarray,
    prefix_lengths: list[int],
    y_labels: np.ndarray,
    random_state: int,
    test_size: float,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> tuple[list[float], list[_TorchLinearProbeModel]]:
    """Train cumulative-prefix probes in parallel on GPU.

    Each task uses a prefix of the same full feature vector. We train K linear heads
    in one loop and apply a per-head prefix mask.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime env dependent
        raise RuntimeError(
            "GPU backend requested but PyTorch is not installed in this environment."
        ) from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU batched cumulative backend requested but CUDA is not available."
        )

    n_tasks = len(prefix_lengths)
    if n_tasks == 0:
        return [], []

    labels = y_labels.astype(np.int64)
    indices = np.arange(labels.shape[0])
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    X_train = X_full_features[train_idx]
    X_test = X_full_features[test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    torch.manual_seed(random_state)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(random_state)

    X_train_t = torch.from_numpy(X_train_scaled).to(device)  # (Ntr, D)
    X_test_t = torch.from_numpy(X_test_scaled).to(device)  # (Nte, D)
    y_train_t = torch.from_numpy(y_train.astype(np.float32)).to(device)  # (Ntr,)

    d_full = X_train_t.shape[1]
    W = torch.nn.Parameter(torch.zeros(n_tasks, d_full, device=device))
    b = torch.nn.Parameter(torch.zeros(n_tasks, device=device))
    optimizer = torch.optim.AdamW([W, b], lr=lr, weight_decay=weight_decay)

    # Prefix mask: task k only sees first prefix_lengths[k] features.
    mask = torch.zeros((n_tasks, d_full), device=device, dtype=torch.float32)
    for k, plen in enumerate(prefix_lengths):
        plen_int = int(max(1, min(plen, d_full)))
        mask[k, :plen_int] = 1.0

    pos = float(np.sum(y_train == 1))
    neg = float(np.sum(y_train == 0))
    pos_weight = torch.tensor(neg / max(pos, 1.0), device=device, dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    y_train_expand = y_train_t.unsqueeze(0).expand(n_tasks, -1)  # (K, Ntr)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        W_masked = W * mask
        logits = X_train_t @ W_masked.T + b.unsqueeze(0)  # (Ntr, K)
        logits = logits.T  # (K, Ntr)
        losses = criterion(logits, y_train_expand)
        loss = losses.mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        W_masked = W * mask
        logits_test = X_test_t @ W_masked.T + b.unsqueeze(0)  # (Nte, K)
        probs_test = torch.sigmoid(logits_test).T  # (K, Nte)
        preds = (probs_test >= 0.5).to(torch.int64).cpu().numpy()

    accuracies = (preds == y_test.reshape(1, -1)).mean(axis=1).astype(float).tolist()
    coef = (W * mask).detach().cpu().numpy()
    models = [_TorchLinearProbeModel(c.reshape(1, -1)) for c in coef]
    return accuracies, models


def _run_probe_gpu(
    X_features: np.ndarray,
    y_labels: np.ndarray,
    random_state: int,
    test_size: float,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> tuple[float, _TorchLinearProbeModel, None]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime env dependent
        raise RuntimeError(
            "GPU backend requested but PyTorch is not installed in this environment."
        ) from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU backend requested but CUDA is not available. "
            "Use --backend cpu or ensure CUDA-enabled torch is installed."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X_features,
        y_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=y_labels,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    y_train_f = y_train.astype(np.float32)

    torch.manual_seed(random_state)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(random_state)

    X_train_t = torch.from_numpy(X_train_scaled).to(device)
    y_train_t = torch.from_numpy(y_train_f).to(device)
    X_test_t = torch.from_numpy(X_test_scaled).to(device)

    model = torch.nn.Linear(X_train_t.shape[1], 1).to(device)

    pos = float(np.sum(y_train_f == 1.0))
    neg = float(np.sum(y_train_f == 0.0))
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    y_train_t = y_train_t.view(-1, 1)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits_test = model(X_test_t)
        probs_test = torch.sigmoid(logits_test).squeeze(1)
        preds = (probs_test >= 0.5).to(torch.int64).cpu().numpy()

    accuracy = float((preds == y_test).mean())
    coef = model.weight.detach().cpu().numpy()
    wrapped_model = _TorchLinearProbeModel(coef=coef)
    return accuracy, wrapped_model, None


# ---------------------------------------------------------------------------
# Multiclass (multinomial logistic regression) GPU probes.
#
# Conventions:
#   - y_labels are integer encoded in [0, n_classes).
#   - Model wraps a (C, D) weight matrix in a .coef_ attribute so the existing
#     spectral-profile utilities (multiclass version) can consume it directly.
#   - Metrics returned:
#       * overall_accuracy  : float, top-1 accuracy over all test samples.
#       * per_class_recall  : dict[class_index -> float], recall (per-class
#                             accuracy conditional on that being the true class).
# ---------------------------------------------------------------------------


def _compute_balanced_class_weights_torch(
    y_train_np: np.ndarray, n_classes: int, device: str
):
    """Return a torch tensor (n_classes,) with sklearn-style 'balanced' weights.

    weight[c] = N / (n_classes * count[c]); unseen classes get weight 0.
    """
    import torch

    counts = np.bincount(y_train_np.astype(np.int64), minlength=n_classes).astype(
        np.float64
    )
    n = float(y_train_np.shape[0])
    weights = np.zeros_like(counts, dtype=np.float64)
    nonzero = counts > 0
    weights[nonzero] = n / (float(n_classes) * counts[nonzero])
    return torch.tensor(weights, device=device, dtype=torch.float32)


def _per_class_recall_from_preds(
    preds: np.ndarray, y_true: np.ndarray, n_classes: int
) -> list[float]:
    """Return per-class recall as a list of length n_classes (NaN -> 0 for absent classes)."""
    recalls: list[float] = []
    for c in range(n_classes):
        mask = y_true == c
        denom = int(mask.sum())
        if denom == 0:
            recalls.append(float("nan"))
        else:
            recalls.append(float((preds[mask] == c).mean()))
    return recalls


def _run_probe_gpu_multiclass(
    X_features: np.ndarray,
    y_labels: np.ndarray,
    n_classes: int,
    random_state: int,
    test_size: float,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> tuple[float, list[float], _TorchLinearProbeModel]:
    """Single-task multiclass logistic-regression probe on GPU.

    Returns (overall_accuracy, per_class_recall_list, wrapped_model).
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - runtime env dependent
        raise RuntimeError(
            "GPU backend requested but PyTorch is not installed in this environment."
        ) from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU backend requested but CUDA is not available. "
            "Ensure CUDA-enabled torch is installed (multiclass mode is GPU-only)."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X_features,
        y_labels,
        test_size=test_size,
        random_state=random_state,
        stratify=y_labels,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    y_train_i = y_train.astype(np.int64)
    y_test_i = y_test.astype(np.int64)

    torch.manual_seed(random_state)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(random_state)

    X_train_t = torch.from_numpy(X_train_scaled).to(device)
    y_train_t = torch.from_numpy(y_train_i).to(device)
    X_test_t = torch.from_numpy(X_test_scaled).to(device)

    model = torch.nn.Linear(X_train_t.shape[1], n_classes).to(device)
    class_weights = _compute_balanced_class_weights_torch(y_train_i, n_classes, device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits_test = model(X_test_t)
        preds = logits_test.argmax(dim=1).cpu().numpy()

    overall_acc = float((preds == y_test_i).mean())
    per_class_recall = _per_class_recall_from_preds(preds, y_test_i, n_classes)
    coef = model.weight.detach().cpu().numpy()  # (C, D)
    wrapped = _TorchLinearProbeModel(coef=coef)
    return overall_acc, per_class_recall, wrapped


def _run_probe_gpu_batched_same_dim_multiclass(
    X_task_features: list[np.ndarray],
    y_labels: np.ndarray,
    n_classes: int,
    random_state: int,
    test_size: float,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> tuple[list[float], list[list[float]], list[_TorchLinearProbeModel]]:
    """Train multiple multiclass probes with identical feature dim in parallel.

    Returns (overall_accs_per_task, per_class_recalls_per_task, models_per_task).
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "GPU backend requested but PyTorch is not installed in this environment."
        ) from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU batched backend requested but CUDA is not available."
        )
    if not X_task_features:
        return [], [], []

    feat_dim = X_task_features[0].shape[1]
    if any(x.shape[1] != feat_dim for x in X_task_features):
        raise ValueError("All batched tasks must have the same feature dimension.")

    n_tasks = len(X_task_features)
    task_tensor = np.stack(X_task_features, axis=0).astype(np.float32)  # (K, N, D)
    labels = y_labels.astype(np.int64)

    indices = np.arange(labels.shape[0])
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=labels
    )
    X_train = task_tensor[:, train_idx, :]
    X_test = task_tensor[:, test_idx, :]
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    X_train_t = torch.from_numpy(X_train).to(device)
    X_test_t = torch.from_numpy(X_test).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)  # (Ntr,)

    mean = X_train_t.mean(dim=1, keepdim=True)
    std = X_train_t.std(dim=1, keepdim=True).clamp_min(1e-6)
    X_train_t = (X_train_t - mean) / std
    X_test_t = (X_test_t - mean) / std

    torch.manual_seed(random_state)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(random_state)

    W = torch.nn.Parameter(
        torch.zeros(n_tasks, feat_dim, n_classes, device=device)
    )
    b = torch.nn.Parameter(torch.zeros(n_tasks, n_classes, device=device))
    optimizer = torch.optim.AdamW([W, b], lr=lr, weight_decay=weight_decay)

    class_weights = _compute_balanced_class_weights_torch(y_train, n_classes, device)

    Ntr = y_train_t.shape[0]
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        # logits: (K, Ntr, C)
        logits = torch.einsum("knd,kdc->knc", X_train_t, W) + b.unsqueeze(1)
        # Flatten tasks into the batch axis for cross_entropy so each task
        # gets equal contribution and we still respect class_weights.
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(n_tasks * Ntr, n_classes),
            y_train_t.repeat(n_tasks),
            weight=class_weights,
        )
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        logits_test = torch.einsum("knd,kdc->knc", X_test_t, W) + b.unsqueeze(1)
        preds_all = logits_test.argmax(dim=-1).cpu().numpy()  # (K, Nte)

    overall_accs = [float((preds_all[k] == y_test).mean()) for k in range(n_tasks)]
    per_class_recalls = [
        _per_class_recall_from_preds(preds_all[k], y_test, n_classes)
        for k in range(n_tasks)
    ]
    coef_np = W.detach().cpu().numpy()  # (K, D, C)
    models = [_TorchLinearProbeModel(coef=coef_np[k].T.copy()) for k in range(n_tasks)]
    return overall_accs, per_class_recalls, models


def _run_probe_gpu_batched_prefix_multiclass(
    X_full_features: np.ndarray,
    prefix_lengths: list[int],
    y_labels: np.ndarray,
    n_classes: int,
    random_state: int,
    test_size: float,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> tuple[list[float], list[list[float]], list[_TorchLinearProbeModel]]:
    """Train cumulative-prefix multiclass probes in parallel on GPU.

    May be memory hungry because W has shape (K, D, C); caller should be
    prepared to fall back to sequential single-task probes on OOM.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "GPU backend requested but PyTorch is not installed in this environment."
        ) from exc

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU batched cumulative backend requested but CUDA is not available."
        )

    n_tasks = len(prefix_lengths)
    if n_tasks == 0:
        return [], [], []

    labels = y_labels.astype(np.int64)
    indices = np.arange(labels.shape[0])
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=labels
    )

    X_train = X_full_features[train_idx]
    X_test = X_full_features[test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    torch.manual_seed(random_state)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(random_state)

    X_train_t = torch.from_numpy(X_train_scaled).to(device)  # (Ntr, D)
    X_test_t = torch.from_numpy(X_test_scaled).to(device)  # (Nte, D)
    y_train_t = torch.from_numpy(y_train).to(device)

    d_full = X_train_t.shape[1]
    W = torch.nn.Parameter(torch.zeros(n_tasks, d_full, n_classes, device=device))
    b = torch.nn.Parameter(torch.zeros(n_tasks, n_classes, device=device))
    optimizer = torch.optim.AdamW([W, b], lr=lr, weight_decay=weight_decay)

    mask = torch.zeros((n_tasks, d_full, 1), device=device, dtype=torch.float32)
    for k, plen in enumerate(prefix_lengths):
        plen_int = int(max(1, min(plen, d_full)))
        mask[k, :plen_int, 0] = 1.0

    class_weights = _compute_balanced_class_weights_torch(y_train, n_classes, device)

    Ntr = y_train_t.shape[0]
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        W_masked = W * mask  # (K, D, C)
        # logits_knc = sum_d X_train[n,d] * W_masked[k,d,c]
        logits = torch.einsum("nd,kdc->knc", X_train_t, W_masked) + b.unsqueeze(1)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(n_tasks * Ntr, n_classes),
            y_train_t.repeat(n_tasks),
            weight=class_weights,
        )
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        W_masked = W * mask
        logits_test = (
            torch.einsum("nd,kdc->knc", X_test_t, W_masked) + b.unsqueeze(1)
        )
        preds_all = logits_test.argmax(dim=-1).cpu().numpy()

    overall_accs = [float((preds_all[k] == y_test).mean()) for k in range(n_tasks)]
    per_class_recalls = [
        _per_class_recall_from_preds(preds_all[k], y_test, n_classes)
        for k in range(n_tasks)
    ]
    coef_np = (W * mask).detach().cpu().numpy()  # (K, D, C)
    models = [_TorchLinearProbeModel(coef=coef_np[k].T.copy()) for k in range(n_tasks)]
    return overall_accs, per_class_recalls, models


def _load_npz_generic(npz_path: str, label_key: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    if "embeddings" not in data.files:
        raise KeyError(f"NPZ at {npz_path} must contain 'embeddings' array.")
    if label_key not in data.files:
        raise KeyError(
            f"NPZ at {npz_path} does not contain label key '{label_key}'. "
            f"Available keys: {list(data.files)}"
        )
    X = data["embeddings"]  # (B, F, T)
    y = data[label_key].astype(str)
    if X.ndim != 3:
        raise ValueError(f"Expected embeddings of shape (B, F, T), got {X.shape}")
    return X, y


def _resolve_gpu_device(args) -> str:
    """Validate GPU availability and return the torch device string to use.

    Raises RuntimeError when backend=gpu/rocm/auto but no CUDA device is visible
    (multiclass mode has no CPU fallback).
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Multiclass mode requires PyTorch; install a CUDA-enabled torch."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Multiclass mode is GPU-only but CUDA is not available. "
            "Request a GPU allocation and activate the conda env before running."
        )

    if args.backend == "rocm":
        if not bool(getattr(torch.version, "hip", None)):
            raise RuntimeError(
                "Backend=rocm requested but torch.version.hip is not set; "
                "this torch build is not ROCm."
            )
    return args.gpu_device


def _run_multiclass_probe(args) -> None:
    """GPU-only multiclass spectral probing on (B, F, T) NPZ embeddings.

    Trains a single multinomial-logistic-regression head over all observed classes
    (per full embedding / per band / per cumulative prefix), and reports:
      * overall top-1 accuracy,
      * per-class recall,
      * overall spectral profile (mean_c mean_f |W[c, f, k]|),
      * per-class spectral profile (mean_f |W[cls, f, k]|).
    """
    from sklearn.preprocessing import LabelEncoder  # local import: avoid unused in binary path

    device = _resolve_gpu_device(args)
    print(
        f"[run_generic_fft_band_probe] Multiclass mode on device={device} "
        f"(backend={args.backend})."
    )

    print(
        f"[run_generic_fft_band_probe] Loading embeddings from {args.npz_path} "
        f"(label_key='{args.label_key}')"
    )
    X, y_str = _load_npz_generic(args.npz_path, args.label_key)

    # Encode labels as 0..C-1 (sorted unique strings for reproducibility).
    le = LabelEncoder()
    y_int = le.fit_transform(y_str).astype(np.int64)
    class_names: list[str] = [str(c) for c in le.classes_.tolist()]
    n_classes = len(class_names)
    if n_classes < 2:
        raise ValueError(
            f"Multiclass mode needs >=2 classes; got {n_classes} in label '{args.label_key}'."
        )
    counts = np.bincount(y_int, minlength=n_classes).tolist()
    n_total = int(y_int.shape[0])
    max_class_prop = float(max(counts)) / float(n_total)
    chance_uniform = 1.0 / float(n_classes)
    print(
        f"[run_generic_fft_band_probe] n_classes={n_classes}, n_total={n_total}, "
        f"class_counts={dict(zip(class_names, counts))}, "
        f"chance_uniform={chance_uniform:.4f}, max_class_prop={max_class_prop:.4f}"
    )

    print("[run_generic_fft_band_probe] Applying spectral transform...")
    coeffs = apply_transform(X, transform_type=args.transform_type, axis=2)
    n_coeffs = coeffs.shape[2]
    n_features = coeffs.shape[1]
    print(
        f"[run_generic_fft_band_probe] coeffs.shape={coeffs.shape} "
        f"(n_coeffs={n_coeffs}, n_features={n_features})"
    )

    X_full_flat = coeffs.reshape(coeffs.shape[0], -1)

    # ---- Full-spectrum multiclass probe (always sequential; we need its weights
    #      for the spectral profile). ----
    print("[run_generic_fft_band_probe] Training full-spectrum multiclass probe...")
    full_acc, full_recall, full_model = _run_probe_gpu_multiclass(
        X_features=X_full_flat,
        y_labels=y_int,
        n_classes=n_classes,
        random_state=args.random_state,
        test_size=0.2,
        device=device,
        epochs=args.gpu_epochs,
        lr=args.gpu_lr,
        weight_decay=args.gpu_weight_decay,
    )
    print(
        f"[run_generic_fft_band_probe] Full-spectrum overall_acc={full_acc:.4f} "
        f"(chance_uniform={chance_uniform:.4f}, max_class_prop={max_class_prop:.4f})"
    )

    # ---- Band-wise multiclass probes ----
    num_bands = n_coeffs // args.band_width
    if num_bands == 0:
        raise ValueError(
            f"band_width={args.band_width} too large for n_coeffs={n_coeffs}: need at least one band."
        )
    bands = {
        b: list(range(b * args.band_width, (b + 1) * args.band_width))
        for b in range(num_bands)
    }

    band_overall_accs: list[float] = []
    band_per_class_recalls: list[list[float]] = []
    use_batched_bands = args.gpu_batch_bands
    if use_batched_bands:
        try:
            band_task_features = [
                coeffs[:, :, bands[b]].reshape(coeffs.shape[0], -1)
                for b in range(num_bands)
            ]
            (
                band_overall_accs,
                band_per_class_recalls,
                _,
            ) = _run_probe_gpu_batched_same_dim_multiclass(
                X_task_features=band_task_features,
                y_labels=y_int,
                n_classes=n_classes,
                random_state=args.random_state,
                test_size=0.2,
                device=device,
                epochs=args.gpu_epochs,
                lr=args.gpu_lr,
                weight_decay=args.gpu_weight_decay,
            )
            print(
                "[run_generic_fft_band_probe] Batched GPU training enabled for "
                f"{num_bands} multiclass band probes."
            )
        except RuntimeError as exc:
            print(
                "[run_generic_fft_band_probe] Batched multiclass band training failed; "
                f"falling back to per-band training. Reason: {exc}"
            )
            band_overall_accs = []
            band_per_class_recalls = []

    if not band_overall_accs:
        for b in range(num_bands):
            band_coeffs = coeffs[:, :, bands[b]].reshape(coeffs.shape[0], -1)
            acc, recall, _ = _run_probe_gpu_multiclass(
                X_features=band_coeffs,
                y_labels=y_int,
                n_classes=n_classes,
                random_state=args.random_state,
                test_size=0.2,
                device=device,
                epochs=args.gpu_epochs,
                lr=args.gpu_lr,
                weight_decay=args.gpu_weight_decay,
            )
            band_overall_accs.append(float(acc))
            band_per_class_recalls.append(list(recall))

    # ---- Cumulative-prefix multiclass probes ----
    cumulative_overall_accs: list[float] = []
    cumulative_per_class_recalls: list[list[float]] = []
    use_batched_cumulative = args.gpu_batch_cumulative
    if use_batched_cumulative:
        try:
            prefix_lengths = [
                n_features * args.band_width * (b + 1) for b in range(num_bands)
            ]
            (
                cumulative_overall_accs,
                cumulative_per_class_recalls,
                _,
            ) = _run_probe_gpu_batched_prefix_multiclass(
                X_full_features=X_full_flat,
                prefix_lengths=prefix_lengths,
                y_labels=y_int,
                n_classes=n_classes,
                random_state=args.random_state,
                test_size=0.2,
                device=device,
                epochs=args.gpu_epochs,
                lr=args.gpu_lr,
                weight_decay=args.gpu_weight_decay,
            )
            print(
                "[run_generic_fft_band_probe] Batched GPU training enabled for "
                f"{num_bands} multiclass cumulative probes."
            )
        except RuntimeError as exc:
            print(
                "[run_generic_fft_band_probe] Batched multiclass cumulative training "
                f"failed (likely OOM); falling back to sequential cumulative probes. "
                f"Reason: {exc}"
            )
            cumulative_overall_accs = []
            cumulative_per_class_recalls = []

    if not cumulative_overall_accs:
        cumulative_coeffs = coeffs[:, :, bands[0]].reshape(coeffs.shape[0], -1)
        acc, recall, _ = _run_probe_gpu_multiclass(
            X_features=cumulative_coeffs,
            y_labels=y_int,
            n_classes=n_classes,
            random_state=args.random_state,
            test_size=0.2,
            device=device,
            epochs=args.gpu_epochs,
            lr=args.gpu_lr,
            weight_decay=args.gpu_weight_decay,
        )
        cumulative_overall_accs.append(float(acc))
        cumulative_per_class_recalls.append(list(recall))
        for b in range(1, num_bands):
            next_band_coeffs = coeffs[:, :, bands[b]].reshape(coeffs.shape[0], -1)
            cumulative_coeffs = np.concatenate(
                (cumulative_coeffs, next_band_coeffs), axis=-1
            )
            acc, recall, _ = _run_probe_gpu_multiclass(
                X_features=cumulative_coeffs,
                y_labels=y_int,
                n_classes=n_classes,
                random_state=args.random_state,
                test_size=0.2,
                device=device,
                epochs=args.gpu_epochs,
                lr=args.gpu_lr,
                weight_decay=args.gpu_weight_decay,
            )
            cumulative_overall_accs.append(float(acc))
            cumulative_per_class_recalls.append(list(recall))

    cumulative_auto_acc = (
        max(cumulative_overall_accs) if cumulative_overall_accs else 0.0
    )
    best_cum_idx = (
        int(np.argmax(cumulative_overall_accs)) if cumulative_overall_accs else None
    )

    # ---- AUTO_TR (trained on selected subset of bands) ----
    trained_auto_acc = None
    trained_auto_recall: list[float] | None = None
    trained_auto_bands: list[int] = []
    need_trained_auto = args.auto_mode in ("trained", "both")
    if need_trained_auto:
        if args.trained_auto_strategy == "best_cumulative_prefix":
            if best_cum_idx is None:
                trained_auto_bands = []
            else:
                trained_auto_bands = list(range(best_cum_idx + 1))
        else:
            k = int(max(1, min(args.trained_auto_top_k, num_bands)))
            topk_idx_desc = np.argsort(np.asarray(band_overall_accs))[::-1][:k]
            trained_auto_bands = sorted(int(i) for i in topk_idx_desc.tolist())

        if trained_auto_bands:
            auto_parts = [
                coeffs[:, :, bands[b]].reshape(coeffs.shape[0], -1)
                for b in trained_auto_bands
            ]
            auto_tr_features = (
                auto_parts[0]
                if len(auto_parts) == 1
                else np.concatenate(auto_parts, axis=-1)
            )
            acc_auto_tr, recall_auto_tr, _ = _run_probe_gpu_multiclass(
                X_features=auto_tr_features,
                y_labels=y_int,
                n_classes=n_classes,
                random_state=args.random_state,
                test_size=0.2,
                device=device,
                epochs=args.gpu_epochs,
                lr=args.gpu_lr,
                weight_decay=args.gpu_weight_decay,
            )
            trained_auto_acc = float(acc_auto_tr)
            trained_auto_recall = list(recall_auto_tr)

    # ---- Spectral profile from full-spectrum model ----
    profile_stats = learned_weight_profile_stats_multiclass(
        full_model,
        n_coeffs=n_coeffs,
        n_features=n_features,
        n_classes=n_classes,
        class_names=class_names,
    )
    overall_stats = profile_stats["overall"]
    per_class_stats = profile_stats["per_class"]

    # ---- Save metrics JSON ----
    metrics_filename = (
        f"{args.prefix}_multiclass_summary_{args.transform_type}_metrics.json"
    )
    metrics_path = os.path.join(args.results_dir, metrics_filename)
    metrics_data = {
        "label_mode": "multiclass",
        "label_key": args.label_key,
        "transform_type": args.transform_type,
        "class_names": class_names,
        "class_counts": {c: int(counts[i]) for i, c in enumerate(class_names)},
        "n_classes": int(n_classes),
        "n_total": int(n_total),
        "chance": {
            "uniform": float(chance_uniform),
            "max_class_prop": float(max_class_prop),
        },
        "accuracies": {
            "full_embedding": float(full_acc),
            "band_specific": band_overall_accs,
            "cumulative_per_band": cumulative_overall_accs,
            "cumulative_auto": float(cumulative_auto_acc),
            "trained_auto": trained_auto_acc,
        },
        "per_class_recall": {
            "full_embedding": dict(zip(class_names, full_recall)),
            "band_specific": [
                dict(zip(class_names, r)) for r in band_per_class_recalls
            ],
            "cumulative_per_band": [
                dict(zip(class_names, r)) for r in cumulative_per_class_recalls
            ],
            "trained_auto": (
                dict(zip(class_names, trained_auto_recall))
                if trained_auto_recall is not None
                else None
            ),
        },
        "auto_details": {
            "auto_mode": args.auto_mode,
            "cumulative_auto_best_band_idx": best_cum_idx,
            "trained_auto_strategy": (
                args.trained_auto_strategy if need_trained_auto else None
            ),
            "trained_auto_top_k": (
                int(args.trained_auto_top_k)
                if need_trained_auto and args.trained_auto_strategy == "topk_bands"
                else None
            ),
            "trained_auto_selected_bands": trained_auto_bands,
        },
        "spectral_profile_overall": {
            "mean_norm": overall_stats["mean_norm"].tolist(),
            "std_norm": overall_stats["std_norm"].tolist(),
            "median_norm": overall_stats["median_norm"].tolist(),
            "q25_norm": overall_stats["q25_norm"].tolist(),
            "q75_norm": overall_stats["q75_norm"].tolist(),
            "mean_raw": overall_stats["mean_raw"].tolist(),
            "std_raw": overall_stats["std_raw"].tolist(),
            "median_raw": overall_stats["median_raw"].tolist(),
            "q25_raw": overall_stats["q25_raw"].tolist(),
            "q75_raw": overall_stats["q75_raw"].tolist(),
        },
        "spectral_profile_per_class": {
            cls: {
                "mean_norm": s["mean_norm"].tolist(),
                "std_norm": s["std_norm"].tolist(),
                "median_norm": s["median_norm"].tolist(),
                "q25_norm": s["q25_norm"].tolist(),
                "q75_norm": s["q75_norm"].tolist(),
                "mean_raw": s["mean_raw"].tolist(),
                "std_raw": s["std_raw"].tolist(),
                "median_raw": s["median_raw"].tolist(),
                "q25_raw": s["q25_raw"].tolist(),
                "q75_raw": s["q75_raw"].tolist(),
            }
            for cls, s in per_class_stats.items()
        },
        "config": {
            "n_coeffs": int(n_coeffs),
            "n_features": int(n_features),
            "band_width": int(args.band_width),
            "random_state": int(args.random_state),
            "npz_path": args.npz_path,
            "results_dir": args.results_dir,
            "prefix": args.prefix,
            "backend": args.backend,
            "gpu_device": str(device),
            "gpu_epochs": int(args.gpu_epochs),
            "gpu_lr": float(args.gpu_lr),
            "gpu_weight_decay": float(args.gpu_weight_decay),
            "gpu_batch_bands": bool(args.gpu_batch_bands),
            "gpu_batch_cumulative": bool(args.gpu_batch_cumulative),
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"[run_generic_fft_band_probe] Multiclass metrics saved to {metrics_path}")

    # ---- Per-class spectral-profile CSVs (one per class, plus overall) ----
    csv_prefix = f"{args.prefix}_multiclass_feature_weight_stats_{args.transform_type}"

    def _dump_stats_csv(name: str, stats: dict) -> None:
        csv_path = os.path.join(args.results_dir, f"{csv_prefix}_{name}.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "freq_idx",
                    "mean_norm",
                    "std_norm",
                    "median_norm",
                    "q25_norm",
                    "q75_norm",
                    "mean_raw",
                    "std_raw",
                    "median_raw",
                    "q25_raw",
                    "q75_raw",
                ]
            )
            for i in range(n_coeffs):
                writer.writerow(
                    [
                        i,
                        float(stats["mean_norm"][i]),
                        float(stats["std_norm"][i]),
                        float(stats["median_norm"][i]),
                        float(stats["q25_norm"][i]),
                        float(stats["q75_norm"][i]),
                        float(stats["mean_raw"][i]),
                        float(stats["std_raw"][i]),
                        float(stats["median_raw"][i]),
                        float(stats["q25_raw"][i]),
                        float(stats["q75_raw"][i]),
                    ]
                )

    _dump_stats_csv("overall", overall_stats)
    for cls, s in per_class_stats.items():
        _dump_stats_csv(str(cls).replace(" ", "_").replace("/", "_"), s)

    # ---- Main summary plot: overall band accs + overall spectral profile ----
    bar_labels = ["ORIG"] + [f"B{b}" for b in range(num_bands)]
    bar_heights = [full_acc] + band_overall_accs
    colors = ["gray"] + ["lightskyblue"] * num_bands
    if args.auto_mode in ("cumulative", "both"):
        bar_labels.append("AUTO_CMAX")
        bar_heights.append(cumulative_auto_acc)
        colors.append("mediumpurple")
    if args.auto_mode in ("trained", "both"):
        bar_labels.append("AUTO_TR")
        bar_heights.append(trained_auto_acc if trained_auto_acc is not None else 0.0)
        colors.append("darkorange")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]}
    )
    title_model = args.prefix.upper() if args.prefix else "GENERIC"
    fig.suptitle(
        f"Multiclass Spectral Probing ({title_model}) — {args.label_key} "
        f"[{n_classes} classes, {args.transform_type.upper()}]",
        fontsize=16,
    )

    ax1.bar(bar_labels, bar_heights, color=colors)
    ax1.set_ylabel("Overall Top-1 Accuracy")
    ax1.set_ylim(0, 1)
    ax1.axhline(
        y=chance_uniform,
        color="r",
        linestyle="--",
        label=f"Chance ({chance_uniform:.2f})",
    )
    ax1.axhline(
        y=max_class_prop,
        color="orange",
        linestyle=":",
        label=f"Max-class prop ({max_class_prop:.2f})",
    )
    ax1.set_title("Probe Performance (bands, multiclass)")
    ax1.legend(loc="best", fontsize=8)
    ax1.tick_params(axis="x", rotation=45)

    x = np.arange(n_coeffs)
    overall_mean_norm = overall_stats["mean_norm"]
    overall_std_norm = overall_stats["std_norm"]
    lower = np.clip(overall_mean_norm - overall_std_norm, 0.0, 1.0)
    upper = np.clip(overall_mean_norm + overall_std_norm, 0.0, 1.0)
    ax2.fill_between(
        x,
        lower,
        upper,
        color="lightgray",
        alpha=0.5,
        label="mean ± std (over C×F)",
    )
    ax2.plot(x, overall_mean_norm, color="black", linewidth=1.2, label="mean weight")
    ax2.set_ylabel("Learned Weight")
    ax2.set_xlabel("Frequency Coefficient")
    ax2.set_xticks([0, n_coeffs // 2, n_coeffs - 1])
    ax2.set_xticklabels(["L", "M", "H"])
    ax2.set_ylim(0, 1)
    ax2.set_title("Overall Spectral Profile")
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = f"{args.prefix}_multiclass_summary_{args.transform_type}.png"
    plot_path = os.path.join(args.results_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"[run_generic_fft_band_probe] Main summary plot saved to {plot_path}")

    # ---- Per-class plots: (left) per-band per-class recall bar chart,
    #      (right) per-class spectral profile. ----
    for cls_idx, cls in enumerate(class_names):
        s = per_class_stats[cls]
        clean_cls = str(cls).replace(" ", "_").replace("/", "_")

        # Per-class recall per band (from multiclass probes).
        band_recall_cls = [float(r[cls_idx]) for r in band_per_class_recalls]
        full_recall_cls = float(full_recall[cls_idx])

        cls_bar_labels = ["ORIG"] + [f"B{b}" for b in range(num_bands)]
        cls_bar_heights = [full_recall_cls] + band_recall_cls
        cls_bar_colors = ["gray"] + ["lightskyblue"] * num_bands
        if args.auto_mode in ("cumulative", "both"):
            if best_cum_idx is not None and cumulative_per_class_recalls:
                auto_cmax_recall = float(
                    cumulative_per_class_recalls[best_cum_idx][cls_idx]
                )
            else:
                auto_cmax_recall = 0.0
            cls_bar_labels.append("AUTO_CMAX")
            cls_bar_heights.append(auto_cmax_recall)
            cls_bar_colors.append("mediumpurple")
        if args.auto_mode in ("trained", "both"):
            auto_tr_recall = (
                float(trained_auto_recall[cls_idx])
                if trained_auto_recall is not None
                else 0.0
            )
            cls_bar_labels.append("AUTO_TR")
            cls_bar_heights.append(auto_tr_recall)
            cls_bar_colors.append("darkorange")

        fig2, (ax1c, ax2c) = plt.subplots(
            1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]}
        )
        fig2.suptitle(
            f"Per-Class View ({title_model}) — {cls} "
            f"[multiclass probe, {args.transform_type.upper()}]",
            fontsize=15,
        )

        ax1c.bar(cls_bar_labels, cls_bar_heights, color=cls_bar_colors)
        ax1c.set_ylabel(f"Recall for class: {cls}")
        ax1c.set_ylim(0, 1)
        ax1c.axhline(
            y=chance_uniform,
            color="r",
            linestyle="--",
            label=f"Chance ({chance_uniform:.2f})",
        )
        ax1c.set_title(
            f"Per-Band Per-Class Recall — {cls}"
        )
        ax1c.legend(loc="best", fontsize=8)
        ax1c.tick_params(axis="x", rotation=45)

        mean_norm = s["mean_norm"]
        std_norm = s["std_norm"]
        lo = np.clip(mean_norm - std_norm, 0.0, 1.0)
        hi = np.clip(mean_norm + std_norm, 0.0, 1.0)
        ax2c.fill_between(
            x, lo, hi, color="lightgray", alpha=0.55, label="mean ± std (over F)"
        )
        ax2c.plot(x, mean_norm, color="black", linewidth=1.3, label="mean")
        ax2c.set_ylabel("Normalized Weight |W[cls, f, k]|")
        ax2c.set_xlabel("Frequency Coefficient")
        ax2c.set_xticks([0, n_coeffs // 2, n_coeffs - 1])
        ax2c.set_xticklabels(["L", "M", "H"])
        ax2c.set_ylim(0, 1)
        ax2c.set_title(f"Per-Class Spectral Profile — {cls}")
        ax2c.legend(loc="upper right", fontsize=8)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fname = (
            f"{args.prefix}_multiclass_per_class_summary_{clean_cls}_"
            f"{args.transform_type}.png"
        )
        out_path = os.path.join(args.results_dir, fname)
        fig2.savefig(out_path)
        plt.close(fig2)

        # Also keep a minimal profile-only panel for easy embedding in reports.
        fig3, axp = plt.subplots(1, 1, figsize=(7, 4))
        axp.fill_between(x, lo, hi, color="lightgray", alpha=0.55, label="mean ± std")
        axp.plot(x, mean_norm, color="black", linewidth=1.3, label="mean")
        axp.set_ylabel("Normalized Weight")
        axp.set_xlabel("Frequency Coefficient")
        axp.set_xticks([0, n_coeffs // 2, n_coeffs - 1])
        axp.set_xticklabels(["L", "M", "H"])
        axp.set_ylim(0, 1)
        axp.set_title(f"Per-Class Spectral Profile — {cls}")
        axp.legend(loc="upper right", fontsize=8)
        fig3.tight_layout()
        fname_only = (
            f"{args.prefix}_multiclass_feature_weight_mean_std_{clean_cls}_"
            f"{args.transform_type}.png"
        )
        fig3.savefig(os.path.join(args.results_dir, fname_only))
        plt.close(fig3)

    print(
        "[run_generic_fft_band_probe] Per-class plots written for "
        f"{len(class_names)} classes (summary + profile-only each)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generic FFT/DCT band-wise spectral probing for a single target label "
            "on (B, F, T) embeddings NPZ."
        )
    )
    parser.add_argument(
        "--npz_path",
        type=str,
        required=True,
        help=(
            "Path to embeddings NPZ with 'embeddings' and label arrays "
            "(e.g. data_artifacts/passt_embeddings_t64.npz, "
            "data_artifacts/passt_embeddings_asap_t32.npz, "
            "data_artifacts/beats_embeddings_t64.npz, "
            "data_artifacts/beats_embeddings_asap_t32.npz)."
        ),
    )
    parser.add_argument(
        "--label_key",
        type=str,
        default="genres",
        help="Name of the label array inside the NPZ (e.g. 'genres' or 'composers').",
    )
    parser.add_argument(
        "--target_label",
        type=str,
        default=None,
        help=(
            "The specific label value to use as positive class (e.g. 'Rock', 'Bach'). "
            "Required when --label_mode=binary; ignored when --label_mode=multiclass."
        ),
    )
    parser.add_argument(
        "--label_mode",
        type=str,
        default="binary",
        choices=["binary", "multiclass"],
        help=(
            "binary: one-vs-rest probe for --target_label (default, backwards compatible). "
            "multiclass: single multinomial probe over all observed classes (GPU-only)."
        ),
    )
    parser.add_argument(
        "--transform_type",
        type=str,
        default="fft",
        choices=["fft", "dct"],
        help="Spectral transform to apply along time axis (default: fft).",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/generic_fft_band_probe",
        help="Directory to store JSON metrics and plots.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="generic",
        help="Filename prefix for outputs (useful to indicate model, e.g. 'passt' or 'beats').",
    )
    parser.add_argument(
        "--band_width",
        type=int,
        default=4,
        help="Number of adjacent frequency coefficients per band (default: 4).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for probe training.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="cpu",
        choices=["cpu", "gpu", "rocm", "auto"],
        help="Probe training backend: cpu (sklearn), gpu (CUDA torch), rocm (AMD torch), or auto.",
    )
    parser.add_argument(
        "--gpu_device",
        type=str,
        default="cuda",
        help="Torch device for GPU backend (default: cuda).",
    )
    parser.add_argument(
        "--gpu_epochs",
        type=int,
        default=200,
        help="Training epochs for torch GPU probe.",
    )
    parser.add_argument(
        "--gpu_lr",
        type=float,
        default=0.05,
        help="Learning rate for torch GPU probe.",
    )
    parser.add_argument(
        "--gpu_weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for torch GPU probe.",
    )
    parser.add_argument(
        "--gpu_batch_bands",
        action="store_true",
        help="Batch all equal-width band probes into one GPU training loop.",
    )
    parser.add_argument(
        "--gpu_batch_cumulative",
        action="store_true",
        help="Batch cumulative prefix probes into one GPU training loop.",
    )
    parser.add_argument(
        "--auto_mode",
        type=str,
        default="both",
        choices=["cumulative", "trained", "both"],
        help=(
            "Which AUTO variant(s) to include in output/plot: "
            "'cumulative' (AUTO_CMAX), 'trained' (AUTO_TR), or 'both'."
        ),
    )
    parser.add_argument(
        "--trained_auto_strategy",
        type=str,
        default="best_cumulative_prefix",
        choices=["best_cumulative_prefix", "topk_bands"],
        help=(
            "How to build AUTO_TR features before separate probe training: "
            "best cumulative prefix, or top-k bands by single-band accuracy."
        ),
    )
    parser.add_argument(
        "--trained_auto_top_k",
        type=int,
        default=8,
        help="k for --trained_auto_strategy topk_bands (default: 8).",
    )
    args = parser.parse_args()

    # ---- Cross-argument validation ----
    if args.label_mode == "binary":
        if args.target_label is None:
            parser.error("--target_label is required when --label_mode=binary.")
    else:  # multiclass
        if args.backend not in ("gpu", "rocm", "auto"):
            parser.error(
                "--label_mode=multiclass is GPU-only; use --backend gpu/rocm/auto "
                "(CPU multiclass path is intentionally not implemented in this round)."
            )
        if args.target_label is not None:
            print(
                "[run_generic_fft_band_probe] NOTE: --target_label is ignored in "
                "multiclass mode."
            )

    os.makedirs(args.results_dir, exist_ok=True)

    if args.label_mode == "multiclass":
        _run_multiclass_probe(args)
        return

    print(
        f"[run_generic_fft_band_probe] Loading embeddings from {args.npz_path} "
        f"(label_key='{args.label_key}')"
    )
    X, y_str = _load_npz_generic(args.npz_path, args.label_key)

    target = args.target_label
    print(f"\n{'=' * 20} Processing target label: {target} {'=' * 20}")

    y_binary = (y_str == target).astype(int)
    n_pos = int(y_binary.sum())
    if n_pos < 10:
        raise ValueError(
            f"Target '{target}' has too few positives ({n_pos}); need at least 10."
        )
    if len(np.unique(y_binary)) < 2:
        raise ValueError(
            f"Target '{target}' must have both positive and negative samples."
        )

    print("[run_generic_fft_band_probe] Applying spectral transform...")
    # FFT / DCT along time axis (axis=2 for (B, F, T))
    coeffs = apply_transform(X, transform_type=args.transform_type, axis=2)
    n_coeffs = coeffs.shape[2]
    print(
        f"[run_generic_fft_band_probe] coeffs.shape={coeffs.shape} "
        f"(n_coeffs={n_coeffs})"
    )

    # Full-spectrum probe.
    probe_fn = run_probe
    if args.backend in ("gpu", "rocm", "auto"):
        use_gpu = args.backend in ("gpu", "rocm")
        if args.backend == "auto":
            try:
                import torch

                use_gpu = torch.cuda.is_available()
            except Exception:
                use_gpu = False
        elif args.backend == "rocm":
            try:
                import torch

                use_gpu = torch.cuda.is_available() and bool(getattr(torch.version, "hip", None))
            except Exception:
                use_gpu = False

        if use_gpu:
            print(
                f"[run_generic_fft_band_probe] Using GPU probe backend on {args.gpu_device}."
            )

            def probe_fn(
                X_probe: np.ndarray, y_probe: np.ndarray, random_state: int
            ) -> tuple[float, _TorchLinearProbeModel, None]:
                return _run_probe_gpu(
                    X_features=X_probe,
                    y_labels=y_probe,
                    random_state=random_state,
                    test_size=0.2,
                    device=args.gpu_device,
                    epochs=args.gpu_epochs,
                    lr=args.gpu_lr,
                    weight_decay=args.gpu_weight_decay,
                )

        else:
            print(
                "[run_generic_fft_band_probe] GPU backend unavailable; falling back to CPU."
            )

    X_full_flat = coeffs.reshape(coeffs.shape[0], -1)
    orig_accuracy, final_model, _ = probe_fn(
        X_full_flat, y_binary, random_state=args.random_state
    )
    if final_model is None:
        raise RuntimeError(f"Probe training failed for target '{target}'.")

    # Band-wise probing (contiguous bands along frequency axis).
    num_bands = n_coeffs // args.band_width
    if num_bands == 0:
        raise ValueError(
            f"band_width={args.band_width} too large for n_coeffs={n_coeffs}: need at least one band."
        )
    bands = {
        b: list(range(b * args.band_width, (b + 1) * args.band_width))
        for b in range(num_bands)
    }

    band_accuracies: list[float] = []
    use_batched_bands = args.gpu_batch_bands and args.backend in ("gpu", "rocm", "auto")
    if use_batched_bands:
        try:
            band_task_features = [
                coeffs[:, :, bands[b]].reshape(coeffs.shape[0], -1)
                for b in range(num_bands)
            ]
            band_accuracies, _ = _run_probe_gpu_batched_same_dim(
                X_task_features=band_task_features,
                y_labels=y_binary,
                random_state=args.random_state,
                test_size=0.2,
                device=args.gpu_device,
                epochs=args.gpu_epochs,
                lr=args.gpu_lr,
                weight_decay=args.gpu_weight_decay,
            )
            print(
                "[run_generic_fft_band_probe] Batched GPU training enabled for "
                f"{num_bands} band probes."
            )
        except Exception as exc:
            print(
                "[run_generic_fft_band_probe] Batched GPU band training failed; "
                f"falling back to per-band training. Reason: {exc}"
            )
            band_accuracies = []

    if not band_accuracies:
        for b in range(num_bands):
            band_coeffs = coeffs[:, :, bands[b]].reshape(coeffs.shape[0], -1)
            acc, _, _ = probe_fn(
                band_coeffs, y_binary, random_state=args.random_state
            )
            band_accuracies.append(float(acc))

    # Cumulative bands from low to high.
    cumulative_accuracies: list[float] = []
    use_batched_cumulative = args.gpu_batch_cumulative and args.backend in ("gpu", "rocm", "auto")
    if use_batched_cumulative:
        try:
            X_full_flat = coeffs.reshape(coeffs.shape[0], -1)
            prefix_lengths = [
                coeffs.shape[1] * args.band_width * (b + 1) for b in range(num_bands)
            ]
            cumulative_accuracies, _ = _run_probe_gpu_batched_prefix(
                X_full_features=X_full_flat,
                prefix_lengths=prefix_lengths,
                y_labels=y_binary,
                random_state=args.random_state,
                test_size=0.2,
                device=args.gpu_device,
                epochs=args.gpu_epochs,
                lr=args.gpu_lr,
                weight_decay=args.gpu_weight_decay,
            )
            print(
                "[run_generic_fft_band_probe] Batched GPU training enabled for "
                f"{num_bands} cumulative probes."
            )
        except Exception as exc:
            print(
                "[run_generic_fft_band_probe] Batched GPU cumulative training failed; "
                f"falling back to sequential cumulative probes. Reason: {exc}"
            )
            cumulative_accuracies = []

    if not cumulative_accuracies:
        cumulative_coeffs = coeffs[:, :, bands[0]].reshape(coeffs.shape[0], -1)
        acc, _, _ = probe_fn(
            cumulative_coeffs, y_binary, random_state=args.random_state
        )
        cumulative_accuracies.append(float(acc))

        for b in range(1, num_bands):
            next_band_coeffs = coeffs[:, :, bands[b]].reshape(coeffs.shape[0], -1)
            cumulative_coeffs = np.concatenate(
                (cumulative_coeffs, next_band_coeffs), axis=-1
            )
            acc, _, _ = probe_fn(
                cumulative_coeffs, y_binary, random_state=args.random_state
            )
            cumulative_accuracies.append(float(acc))

    cumulative_auto_accuracy = (
        max(cumulative_accuracies) if cumulative_accuracies else 0.0
    )
    best_cumulative_band_idx = (
        int(np.argmax(cumulative_accuracies)) if cumulative_accuracies else None
    )

    trained_auto_accuracy = None
    trained_auto_bands: list[int] = []
    need_trained_auto = args.auto_mode in ("trained", "both")
    if need_trained_auto:
        if args.trained_auto_strategy == "best_cumulative_prefix":
            if best_cumulative_band_idx is None:
                trained_auto_bands = []
            else:
                trained_auto_bands = list(range(best_cumulative_band_idx + 1))
        else:
            k = int(max(1, min(args.trained_auto_top_k, num_bands)))
            topk_idx_desc = np.argsort(np.asarray(band_accuracies))[::-1][:k]
            trained_auto_bands = sorted(int(i) for i in topk_idx_desc.tolist())

        if trained_auto_bands:
            auto_parts = [
                coeffs[:, :, bands[b]].reshape(coeffs.shape[0], -1)
                for b in trained_auto_bands
            ]
            auto_tr_features = (
                auto_parts[0]
                if len(auto_parts) == 1
                else np.concatenate(auto_parts, axis=-1)
            )
            acc_auto_tr, _, _ = probe_fn(
                auto_tr_features, y_binary, random_state=args.random_state
            )
            trained_auto_accuracy = float(acc_auto_tr)

    # Learned spectral profile from full-spectrum probe.
    raw_weights = final_model.coef_.flatten()
    n_features = coeffs.shape[1]
    weight_stats = learned_weight_profile_stats(
        final_model, n_coeffs=n_coeffs, n_features=n_features
    )
    normalized_weights = weight_stats["mean_norm"]
    std_norm = weight_stats["std_norm"]

    # ---- Save metrics JSON ----
    clean_label = str(target).replace(" ", "_").replace("/", "_")
    metrics_filename = (
        f"{args.prefix}_spectral_summary_{clean_label}_{args.transform_type}_metrics.json"
    )
    metrics_path = os.path.join(args.results_dir, metrics_filename)

    metrics_data = {
        "target_label": target,
        "label_key": args.label_key,
        "transform_type": args.transform_type,
        "accuracies": {
            "full_embedding": float(orig_accuracy),
            "band_specific": band_accuracies,
            "cumulative_auto": float(cumulative_auto_accuracy),
            "cumulative_per_band": cumulative_accuracies,
            "trained_auto": trained_auto_accuracy,
        },
        "auto_details": {
            "auto_mode": args.auto_mode,
            "cumulative_auto_best_band_idx": best_cumulative_band_idx,
            "trained_auto_strategy": (
                args.trained_auto_strategy if need_trained_auto else None
            ),
            "trained_auto_top_k": (
                int(args.trained_auto_top_k)
                if need_trained_auto and args.trained_auto_strategy == "topk_bands"
                else None
            ),
            "trained_auto_selected_bands": trained_auto_bands,
        },
        "model_weights": {
            "mean": float(np.mean(raw_weights)),
            "std_dev": float(np.std(raw_weights)),
            "max": float(np.max(raw_weights)),
            "min": float(np.min(raw_weights)),
        },
        "spectral_profile": normalized_weights.tolist(),
        "spectral_profile_stats": {
            "mean_norm": weight_stats["mean_norm"].tolist(),
            "std_norm": weight_stats["std_norm"].tolist(),
            "median_norm": weight_stats["median_norm"].tolist(),
            "q25_norm": weight_stats["q25_norm"].tolist(),
            "q75_norm": weight_stats["q75_norm"].tolist(),
            "mean_raw": weight_stats["mean_raw"].tolist(),
            "std_raw": weight_stats["std_raw"].tolist(),
            "median_raw": weight_stats["median_raw"].tolist(),
            "q25_raw": weight_stats["q25_raw"].tolist(),
            "q75_raw": weight_stats["q75_raw"].tolist(),
        },
        "config": {
            "n_coeffs": int(n_coeffs),
            "band_width": int(args.band_width),
            "random_state": int(args.random_state),
            "npz_path": args.npz_path,
            "results_dir": args.results_dir,
            "prefix": args.prefix,
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"[run_generic_fft_band_probe] Metrics saved to {metrics_path}")

    # ---- Save per-frequency weight stats (JSON + CSV) ----
    stats_filename = (
        f"{args.prefix}_feature_weight_stats_{clean_label}_{args.transform_type}.json"
    )
    stats_path = os.path.join(args.results_dir, stats_filename)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "target_label": target,
                "transform_type": args.transform_type,
                "n_coeffs": int(n_coeffs),
                "stats": metrics_data["spectral_profile_stats"],
            },
            f,
            indent=4,
        )
    print(f"[run_generic_fft_band_probe] Feature-weight stats saved to {stats_path}")

    stats_csv_filename = (
        f"{args.prefix}_feature_weight_stats_{clean_label}_{args.transform_type}.csv"
    )
    stats_csv_path = os.path.join(args.results_dir, stats_csv_filename)
    with open(stats_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "freq_idx",
                "mean_norm",
                "std_norm",
                "median_norm",
                "q25_norm",
                "q75_norm",
                "mean_raw",
                "std_raw",
                "median_raw",
                "q25_raw",
                "q75_raw",
            ]
        )
        for i in range(n_coeffs):
            writer.writerow(
                [
                    i,
                    float(weight_stats["mean_norm"][i]),
                    float(weight_stats["std_norm"][i]),
                    float(weight_stats["median_norm"][i]),
                    float(weight_stats["q25_norm"][i]),
                    float(weight_stats["q75_norm"][i]),
                    float(weight_stats["mean_raw"][i]),
                    float(weight_stats["std_raw"][i]),
                    float(weight_stats["median_raw"][i]),
                    float(weight_stats["q25_raw"][i]),
                    float(weight_stats["q75_raw"][i]),
                ]
            )
    print(f"[run_generic_fft_band_probe] Feature-weight CSV saved to {stats_csv_path}")

    # ---- Plot: bar chart (band accuracies) + spectral profile ----
    bar_labels = ["ORIG"] + [f"B{b}" for b in range(num_bands)]
    bar_heights = [orig_accuracy] + band_accuracies
    colors = ["gray"] + ["lightskyblue"] * num_bands
    if args.auto_mode in ("cumulative", "both"):
        bar_labels.append("AUTO_CMAX")
        bar_heights.append(cumulative_auto_accuracy)
        colors.append("mediumpurple")
    if args.auto_mode in ("trained", "both"):
        bar_labels.append("AUTO_TR")
        bar_heights.append(
            trained_auto_accuracy if trained_auto_accuracy is not None else 0.0
        )
        colors.append("darkorange")

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]}
    )
    title_model = args.prefix.upper() if args.prefix else "GENERIC"
    fig.suptitle(
        f"Spectral Probing Analysis ({title_model}) — Label: {target} ({args.transform_type.upper()})",
        fontsize=16,
    )

    # Left: probe performance per band.
    ax1.bar(bar_labels, bar_heights, color=colors)
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1)
    chance = float(y_binary.mean())
    ax1.axhline(
        y=chance,
        color="r",
        linestyle="--",
        label=f"Chance ({chance:.2f})",
    )
    ax1.set_title("Probe Performance (bands)")
    ax1.legend()
    ax1.tick_params(axis="x", rotation=45)

    # Right: spectral profile.
    x = np.arange(len(normalized_weights))
    lower = np.clip(normalized_weights - std_norm, 0.0, 1.0)
    upper = np.clip(normalized_weights + std_norm, 0.0, 1.0)
    ax2.fill_between(
        x,
        lower,
        upper,
        color="lightgray",
        alpha=0.5,
        label="mean ± std (over features)",
    )
    ax2.plot(x, normalized_weights, color="black", linewidth=1.2, label="mean weight")
    ax2.set_ylabel("Learned Weight")
    ax2.set_xlabel("Frequency Coefficient")
    ax2.set_xticks([0, n_coeffs // 2, n_coeffs - 1])
    ax2.set_xticklabels(["L", "M", "H"])
    ax2.set_ylim(0, 1)
    ax2.set_title("Spectral Profile")
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_filename = (
        f"{args.prefix}_spectral_summary_{clean_label}_{args.transform_type}.png"
    )
    plot_path = os.path.join(args.results_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"[run_generic_fft_band_probe] Plot saved to {plot_path}")

    # ---- Plot: feature weight mean±std only (for report panel use) ----
    fig2, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.fill_between(
        x,
        lower,
        upper,
        color="lightgray",
        alpha=0.55,
        label="mean ± std",
    )
    ax.plot(x, normalized_weights, color="black", linewidth=1.3, label="mean")
    ax.set_ylabel("Normalized Weight")
    ax.set_xlabel("Frequency Coefficient")
    ax.set_xticks([0, n_coeffs // 2, n_coeffs - 1])
    ax.set_xticklabels(["L", "M", "H"])
    ax.set_ylim(0, 1)
    ax.set_title(f"Feature Weight Mean±Std ({target})")
    ax.legend(loc="upper right", fontsize=8)
    fig2.tight_layout()
    mean_std_plot_filename = (
        f"{args.prefix}_feature_weight_mean_std_{clean_label}_{args.transform_type}.png"
    )
    mean_std_plot_path = os.path.join(args.results_dir, mean_std_plot_filename)
    fig2.savefig(mean_std_plot_path)
    plt.close(fig2)
    print(
        "[run_generic_fft_band_probe] Feature-weight mean±std plot saved to "
        f"{mean_std_plot_path}"
    )


if __name__ == "__main__":
    main()

