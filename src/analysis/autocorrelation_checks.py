import os
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from src.analysis.zero_padded_fft import get_embedding_and_config, _ensure_dir


def compute_autocorrelation_1d(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Compute normalized autocorrelation for a 1D sequence up to max_lag.

    Args:
        x: shape (T,)
        max_lag: maximum lag (inclusive)

    Returns:
        rho: shape (max_lag + 1,), rho[0] = 1
    """
    if x.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {x.shape}")
    if max_lag < 0:
        raise ValueError(f"max_lag must be >= 0, got {max_lag}")

    x_centered = x - np.mean(x)
    r_full = np.correlate(x_centered, x_centered, mode="full")
    mid = r_full.size // 2
    r = r_full[mid : mid + max_lag + 1]
    if r[0] == 0:
        return np.zeros_like(r)
    return r / r[0]


def aggregate_autocorrelation_over_features(
    x: np.ndarray,
    max_lag: int,
    n_features_sample: Optional[int] = None,
    random_state: int = 42,
) -> np.ndarray:
    """
    Aggregate autocorrelation over (samples, features) to get a global curve.

    Args:
        x: embeddings, shape (B, F, T)
        max_lag: maximum lag
        n_features_sample: if not None, randomly sample this many feature dims
            to reduce computation. If None, use all F.
        random_state: RNG seed for feature sampling.

    Returns:
        rho_mean: shape (max_lag + 1,)
    """
    if x.ndim != 3:
        raise ValueError(f"Expected (B, F, T), got {x.shape}")

    b, f, t = x.shape
    if max_lag >= t:
        raise ValueError(
            f"max_lag {max_lag} must be < T={t} (time length of embeddings)"
        )

    rng = np.random.RandomState(random_state)
    if n_features_sample is None or n_features_sample >= f:
        feat_indices = np.arange(f)
    else:
        feat_indices = rng.choice(f, size=n_features_sample, replace=False)

    acc = np.zeros(max_lag + 1, dtype=np.float64)
    count = 0

    total_iters = len(feat_indices) * b
    # Progress bar over all (feature, sample) pairs
    for feat_idx in tqdm(
        feat_indices,
        desc="autocorr: features",
        leave=False,
    ):
        for sample_idx in range(b):
            rho = compute_autocorrelation_1d(x[sample_idx, feat_idx, :], max_lag)
            acc += rho
            count += 1

    if count == 0:
        return np.zeros(max_lag + 1, dtype=np.float64)
    return acc / float(count)


def analyze_autocorrelation_for_dataset(
    config_name: str,
    max_lag: int,
    out_dir: str,
    subset_labels: Optional[List[str]] = None,
    n_features_sample: Optional[int] = 128,
) -> None:
    """
    Run autocorrelation analysis for a given dataset config.

    - Loads embeddings via EmbeddingConfig.
    - Computes global mean autocorrelation over (samples, features).
    - Optionally repeats for specific label subsets (genres / composers).

    Saves:
        - PNG plots of rho(lag) for each subset.
        - NPY files with rho(lag) values.
    """
    _ensure_dir(out_dir)
    x, labels, cfg = get_embedding_and_config(config_name)

    # Determine sensible max_lag if user passed something too large
    t = x.shape[2]
    if max_lag >= t:
        max_lag = t - 1

    # Global curve
    rho_all = aggregate_autocorrelation_over_features(
        x, max_lag=max_lag, n_features_sample=n_features_sample
    )
    _save_and_plot_autocorr(
        rho_all,
        out_dir=out_dir,
        tag="all",
        title=f"{config_name}: mean autocorrelation (all samples)",
        max_lag=max_lag,
    )

    # Per-subset curves (optional)
    if subset_labels is not None and len(subset_labels) > 0:
        labels = np.asarray(labels)
        for label in subset_labels:
            mask = labels == label
            if not np.any(mask):
                continue
            x_sub = x[mask]
            rho_sub = aggregate_autocorrelation_over_features(
                x_sub,
                max_lag=max_lag,
                n_features_sample=n_features_sample,
            )
            safe_label = str(label).replace(" ", "_").replace("/", "_")
            _save_and_plot_autocorr(
                rho_sub,
                out_dir=out_dir,
                tag=safe_label,
                title=f"{config_name}: mean autocorrelation ({label})",
                max_lag=max_lag,
            )


def _save_and_plot_autocorr(
    rho: np.ndarray,
    out_dir: str,
    tag: str,
    title: str,
    max_lag: int,
) -> None:
    """
    Save a single autocorrelation curve and generate a plot.
    """
    if rho.ndim != 1:
        raise ValueError(f"Expected 1D rho, got shape {rho.shape}")

    npy_path = os.path.join(out_dir, f"autocorr_{tag}.npy")
    np.save(npy_path, rho)

    lags = np.arange(rho.shape[0])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(lags, rho, color="C0")
    ax.set_xlabel("Lag (samples along time axis)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(title)

    # Highlight a few key lags: 4, 8, T/4, T/2, 3T/4 (if within range)
    key_lags: List[int] = [4, 8]
    # T is approximated by max_lag+1 here (global)
    t_est = max_lag + 1
    for frac in (0.25, 0.5, 0.75):
        k = int(round(frac * t_est))
        if 0 < k <= max_lag:
            key_lags.append(k)
    key_lags = sorted(set([k for k in key_lags if 0 <= k <= max_lag]))

    for k in key_lags:
        ax.axvline(k, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    plt.tight_layout()
    png_path = os.path.join(out_dir, f"autocorr_{tag}.png")
    plt.savefig(png_path)
    plt.close(fig)

