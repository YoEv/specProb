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
        - left: bar chart of ORIG, B0..B{n-1}, AUTO accuracies;
        - right: spectral profile (learned weights over frequency coeffs).

Can be used for:
- PaSST FMA / ASAP NPZs;
- BEATs FMA / ASAP NPZs;
- or any other (B, F, T) embeddings with a label array.
"""

import argparse
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
from src.visualization.spectral_profile import learned_weight_profile


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
        required=True,
        help="The specific label value to use as positive class (e.g. 'Rock', 'Bach').",
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
        choices=["cpu", "gpu", "auto"],
        help="Probe training backend: cpu (sklearn), gpu (torch), or auto.",
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
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

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
    if args.backend in ("gpu", "auto"):
        use_gpu = args.backend == "gpu"
        if args.backend == "auto":
            try:
                import torch

                use_gpu = torch.cuda.is_available()
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
    use_batched_bands = args.gpu_batch_bands and args.backend in ("gpu", "auto")
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
    use_batched_cumulative = args.gpu_batch_cumulative and args.backend in ("gpu", "auto")
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

    auto_accuracy = max(cumulative_accuracies) if cumulative_accuracies else 0.0

    # Learned spectral profile from full-spectrum probe.
    raw_weights = final_model.coef_.flatten()
    n_features = coeffs.shape[1]
    normalized_weights = learned_weight_profile(
        final_model, n_coeffs=n_coeffs, n_features=n_features
    )

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
            "cumulative_auto": float(auto_accuracy),
            "cumulative_per_band": cumulative_accuracies,
        },
        "model_weights": {
            "mean": float(np.mean(raw_weights)),
            "std_dev": float(np.std(raw_weights)),
            "max": float(np.max(raw_weights)),
            "min": float(np.min(raw_weights)),
        },
        "spectral_profile": normalized_weights.tolist(),
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

    # ---- Plot: bar chart (band accuracies) + spectral profile ----
    bar_labels = ["ORIG"] + [f"B{b}" for b in range(num_bands)] + ["AUTO"]
    bar_heights = [orig_accuracy] + band_accuracies + [auto_accuracy]
    colors = ["gray"] + ["lightskyblue"] * num_bands + ["mediumpurple"]

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
    ax2.plot(x, normalized_weights, color="black", linewidth=1)
    ax2.set_ylabel("Learned Weight")
    ax2.set_xlabel("Frequency Coefficient")
    ax2.set_xticks([0, n_coeffs // 2, n_coeffs - 1])
    ax2.set_xticklabels(["L", "M", "H"])
    ax2.set_ylim(0, 1)
    ax2.set_title("Spectral Profile")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_filename = (
        f"{args.prefix}_spectral_summary_{clean_label}_{args.transform_type}.png"
    )
    plot_path = os.path.join(args.results_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"[run_generic_fft_band_probe] Plot saved to {plot_path}")


if __name__ == "__main__":
    main()

