import os
from typing import Dict, List, Tuple, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.config.spectral_experiments import EmbeddingConfig, get_embedding_config


def get_embedding_and_config(
    config_name: str,
) -> Tuple[np.ndarray, np.ndarray, EmbeddingConfig]:
    """
    Load embeddings and labels according to an EmbeddingConfig and return a
    unified (B, F, T) view together with the config.

    This does NOT depend on src.data_processing.loader so that we can handle
    both FMA (T=64) and ASAP (T=32) purely via EmbeddingConfig.
    """
    cfg = get_embedding_config(config_name)
    with np.load(cfg.embeddings_file) as data:
        embeddings = data["embeddings"]
        if "genres" in data:
            labels = data["genres"]
        elif "composers" in data:
            labels = data["composers"]
        elif "labels" in data:
            labels = data["labels"]
        else:
            raise KeyError(
                f"No known label key found in {cfg.embeddings_file}. "
                "Expected one of: 'genres', 'composers', 'labels'."
            )

    # Expect raw CLAP shape (B, 768, 2, T_frames)
    if embeddings.ndim != 4:
        raise ValueError(
            f"Expected 4D embeddings (B, 768, 2, T), got shape {embeddings.shape}"
        )

    b, n_channels, n_segments, t_frames = embeddings.shape
    if n_channels != cfg.n_channels or n_segments != cfg.n_segments:
        raise ValueError(
            f"Embedding shape {embeddings.shape} does not match config "
            f"(n_channels={cfg.n_channels}, n_segments={cfg.n_segments})."
        )
    if t_frames != cfg.frames_per_segment:
        raise ValueError(
            f"Time frames {t_frames} do not match config.frames_per_segment "
            f"{cfg.frames_per_segment} for '{cfg.name}'."
        )

    # (B, 768, 2, T) -> (B, 768*2, T)
    x = embeddings.reshape(b, n_channels * n_segments, t_frames)
    return x, labels, cfg


def zero_pad_along_time(
    x: np.ndarray,
    target_factors: List[int],
) -> Dict[int, np.ndarray]:
    """
    Zero-pad along the last axis (time) to multiple target length factors.

    Args:
        x: array of shape (B, F, T_orig)
        target_factors: e.g. [1, 2, 4] meaning T_pad = factor * T_orig

    Returns:
        dict mapping factor -> padded array of shape (B, F, factor * T_orig)
    """
    if x.ndim != 3:
        raise ValueError(f"Expected X with shape (B, F, T), got {x.shape}")

    b, f, t = x.shape
    padded: Dict[int, np.ndarray] = {}
    for factor in sorted(set(target_factors)):
        if factor < 1:
            raise ValueError(f"Padding factor must be >=1, got {factor}")
        t_pad = factor * t
        out = np.zeros((b, f, t_pad), dtype=x.dtype)
        out[:, :, :t] = x
        padded[factor] = out
    return padded


def compute_fft_mean_magnitude(
    x_dict: Dict[int, np.ndarray],
    window_type: Optional[str] = None,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    For each padded input, compute mean magnitude FFT over (B, F) and return
    both the spectrum and its corresponding normalized frequency axis.

    Returns:
        dict[factor] -> (mean_spectrum, freq_norm)
    """
    from src.analysis.spectral import apply_transform

    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for factor, x in x_dict.items():
        coeffs = apply_transform(
            x, transform_type="fft", axis=2, window_type=window_type
        )  # (B, F, n_coeffs)
        mean_spec = np.mean(coeffs, axis=(0, 1))  # (n_coeffs,)
        n_coeffs = mean_spec.shape[0]
        freq_norm = np.linspace(0.0, 1.0, num=n_coeffs, endpoint=True)
        out[factor] = (mean_spec, freq_norm)
    return out


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def analyze_zero_padding_for_dataset(
    config_name: str,
    factors: List[int],
    window_type: Optional[str],
    out_dir: str,
    subset_labels: Optional[List[str]] = None,
    max_samples: Optional[int] = 5000,
) -> None:
    """
    Run zero-padding + FFT mean spectrum analysis for a given dataset config.

    - Loads embeddings using EmbeddingConfig.
    - Zero-pads along time for each factor.
    - Computes mean FFT spectrum over (B, F).
    - Optionally repeats for specific label subsets (genres / composers).

    Saves:
        - PNG plots of mean spectra vs normalized frequency for each subset.
        - NPY files with mean spectra and freq grids for reproducibility.
    """
    _ensure_dir(out_dir)
    x, labels, cfg = get_embedding_and_config(config_name)

    # To keep memory bounded, optionally subsample along batch axis.
    if max_samples is not None and x.shape[0] > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(x.shape[0], size=max_samples, replace=False)
        x = x[idx]
        labels = np.asarray(labels)[idx]

    # Full dataset (possibly subsampled)
    x_padded = zero_pad_along_time(x, factors)
    spectra = compute_fft_mean_magnitude(x_padded, window_type=window_type)
    _save_and_plot_spectra(
        spectra,
        out_dir=out_dir,
        tag=f"all_window={window_type or 'none'}",
        title=f"{config_name}: zero-padded mean FFT (all samples)",
    )

    # Per-subset (optional)
    if subset_labels is not None and len(subset_labels) > 0:
        labels = np.asarray(labels)
        from tqdm.auto import tqdm  # local import to avoid hard dep for other callers

        for label in tqdm(
            subset_labels,
            desc=f"zero-padded FFT subsets ({config_name})",
            leave=False,
        ):
            mask = labels == label
            if not np.any(mask):
                continue
            x_sub = x[mask]
            x_padded_sub = zero_pad_along_time(x_sub, factors)
            spectra_sub = compute_fft_mean_magnitude(
                x_padded_sub, window_type=window_type
            )
            safe_label = str(label).replace(" ", "_").replace("/", "_")
            _save_and_plot_spectra(
                spectra_sub,
                out_dir=out_dir,
                tag=f"{safe_label}_window={window_type or 'none'}",
                title=f"{config_name}: zero-padded mean FFT ({label})",
            )


def _save_and_plot_spectra(
    spectra: Dict[int, Tuple[np.ndarray, np.ndarray]],
    out_dir: str,
    tag: str,
    title: str,
) -> None:
    """
    Helper to save spectra dict as .npz and generate a multi-factor plot.

    spectra: dict[factor] -> (mean_spectrum, freq_norm)
    """
    # Save raw numeric data
    npz_path = os.path.join(out_dir, f"zero_padded_mean_spectrum_{tag}.npz")
    np.savez(
        npz_path,
        **{
            f"factor_{factor}": spec
            for factor, (spec, _) in spectra.items()
        },
        **{
            f"freq_{factor}": freq
            for factor, (_, freq) in spectra.items()
        },
    )

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    for factor in sorted(spectra.keys()):
        mean_spec, freq = spectra[factor]
        ax.plot(freq, mean_spec, label=f"factor={factor}")

    ax.set_xlabel("Normalized frequency")
    ax.set_ylabel("Mean magnitude over (samples, features)")
    ax.set_xlim(0.0, 1.0)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    png_path = os.path.join(out_dir, f"zero_padded_mean_spectrum_{tag}.png")
    plt.savefig(png_path)
    plt.close(fig)

