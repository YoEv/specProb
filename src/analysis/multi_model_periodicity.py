import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.config.multi_model_sanity import ModelEmbeddingSpec, get_model_spec
from src.analysis.zero_padded_fft import zero_pad_along_time, compute_fft_mean_magnitude
from src.analysis.autocorrelation_checks import (
    aggregate_autocorrelation_over_features,
    _ensure_dir,
)


def load_model_embeddings(spec: ModelEmbeddingSpec) -> Dict[str, np.ndarray]:
    """
    Load embeddings + labels from a NPZ file according to ModelEmbeddingSpec.

    Expected NPZ keys:
        - 'embeddings': (B, F, T)
        - one of 'labels', 'genres', 'composers'
    """
    with np.load(spec.embeddings_file) as data:
        raw = data["embeddings"]
        if "labels" in data:
            labels = data["labels"]
        elif "genres" in data:
            labels = data["genres"]
        elif "composers" in data:
            labels = data["composers"]
        else:
            raise KeyError(
                f"No label key found in {spec.embeddings_file}. "
                "Expected one of: 'labels', 'genres', 'composers'."
            )

    # Allow both (B, F, T) and raw CLAP shape (B, 768, 2, T_frames).
    if raw.ndim == 3:
        embeddings = raw
    elif raw.ndim == 4 and raw.shape[1] * raw.shape[2] == spec.feature_dim:
        # (B, 768, 2, T_frames) -> (B, 1536, T_frames)
        b, c, s, t = raw.shape
        embeddings = raw.reshape(b, c * s, t)
    else:
        raise ValueError(
            f"Expected embeddings with shape (B, F, T) or (B, 768, 2, T), "
            f"got {raw.shape}"
        )
    if embeddings.shape[1] != spec.feature_dim or embeddings.shape[2] != spec.time_length:
        # Soft warning: we don't raise, but log a mismatch for debugging.
        print(
            f"[multi_model_periodicity] Warning: embeddings shape {embeddings.shape} "
            f"does not match spec (F={spec.feature_dim}, T={spec.time_length})."
        )

    return {"embeddings": embeddings, "labels": labels}


def compare_fft_across_models(
    dataset: str,
    models: List[str],
    factors: List[int],
    window_type: str | None,
    out_dir: str,
) -> None:
    """
    For a given dataset, compare zero-padded mean FFT spectra across models.

    Currently only CLAP specs are guaranteed to exist; other models will
    raise if their NPZ files are missing.
    """
    _ensure_dir(out_dir)
    spectra_per_model: Dict[str, Dict[int, tuple[np.ndarray, np.ndarray]]] = {}

    for model in models:
        spec = get_model_spec(model, dataset)
        loaded = load_model_embeddings(spec)
        x = loaded["embeddings"]

        x_padded = zero_pad_along_time(x, factors)
        spectra = compute_fft_mean_magnitude(
            x_padded,
            window_type=None if window_type is None or window_type == "none" else window_type,
        )
        spectra_per_model[model] = spectra

    # (A) For each factor, plot all models on the same normalised frequency axis.
    for factor in sorted(factors):
        fig, ax = plt.subplots(figsize=(8, 4))
        for model in models:
            if model not in spectra_per_model:
                continue
            mean_spec, freq = spectra_per_model[model][factor]
            ax.plot(freq, mean_spec, label=model)

        ax.set_xlabel("Normalized frequency")
        ax.set_ylabel("Mean magnitude over (samples, features)")
        ax.set_xlim(0.0, 1.0)
        ax.set_title(f"{dataset}: zero-padded mean FFT (factor={factor})")
        ax.legend()
        plt.tight_layout()

        png_path = os.path.join(
            out_dir,
            f"fft_mean_spectrum_{dataset}_factor={factor}_window={window_type or 'none'}.png",
        )
        plt.savefig(png_path)
        plt.close(fig)

    # (B) For each model, plot all factors on the same figure (what the user requested).
    for model in models:
        if model not in spectra_per_model:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        for factor in sorted(factors):
            if factor not in spectra_per_model[model]:
                continue
            mean_spec, freq = spectra_per_model[model][factor]
            ax.plot(freq, mean_spec, label=f"factor={factor}")

        ax.set_xlabel("Normalized frequency")
        ax.set_ylabel("Mean magnitude over (samples, features)")
        ax.set_xlim(0.0, 1.0)
        ax.set_title(f"{dataset} ({model}): zero-padded mean FFT (all factors)")
        ax.legend()
        plt.tight_layout()

        png_path = os.path.join(
            out_dir,
            f"fft_mean_spectrum_{dataset}_model={model}_all_factors_window={window_type or 'none'}.png",
        )
        plt.savefig(png_path)
        plt.close(fig)


def compare_autocorrelation_across_models(
    dataset: str,
    models: List[str],
    max_lag: int,
    out_dir: str,
    n_features_sample: int = 128,
) -> None:
    """
    For a given dataset, compare mean autocorrelation curves across models.
    """
    _ensure_dir(out_dir)
    rho_per_model: Dict[str, np.ndarray] = {}

    for model in models:
        spec = get_model_spec(model, dataset)
        loaded = load_model_embeddings(spec)
        x = loaded["embeddings"]

        # Clip max_lag to be < T
        t = x.shape[2]
        lag = min(max_lag, t - 1)
        rho = aggregate_autocorrelation_over_features(
            x,
            max_lag=lag,
            n_features_sample=n_features_sample,
        )
        rho_per_model[model] = rho

    # Plot all models on the same figure.
    if not rho_per_model:
        return

    # Different models may have different effective max_lag (due to varying T).
    # Align all curves to the minimum available length to avoid shape mismatches.
    min_len = min(rho.shape[0] for rho in rho_per_model.values())
    lags = np.arange(min_len)

    fig, ax = plt.subplots(figsize=(8, 4))
    for model, rho in rho_per_model.items():
        ax.plot(lags, rho[:min_len], label=model)

    ax.set_xlabel("Lag (samples along time axis)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(f"{dataset}: mean autocorrelation across models")
    ax.legend()
    plt.tight_layout()

    png_path = os.path.join(out_dir, f"autocorr_mean_{dataset}.png")
    plt.savefig(png_path)
    plt.close(fig)

