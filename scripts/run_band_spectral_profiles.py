"""
Band-level spectral probing profiles for generic (B, F, T) embeddings.

This script mirrors the PaSST / CLAP spectral probing setup, but instead of
operating on individual FFT bins, it first aggregates neighbouring bins into
fixed-width bands (e.g. 4 bins per band), and then:

1. Runs rFFT along the time axis to obtain magnitude spectra.
2. Aggregates frequency coefficients into bands.
3. For each class (genre / composer) with enough samples, trains a one-vs-rest
   logistic regression probe on flattened band-wise spectral features.
4. Derives the learned band-level spectral profile (weight per band) and saves:
     - a JSON summary with band-wise profiles and accuracies;
     - a PNG plot visualising mean band magnitude and learned weight curves.

Can be used with:
  - PaSST embeddings NPZ (e.g. passt_embeddings_t64.npz / passt_embeddings_asap_t32.npz)
  - BEATs embeddings NPZ (e.g. beats_embeddings_t64.npz / beats_embeddings_asap_t32.npz)
  - any other (B, F, T) embeddings NPZ with a label array.
"""

import argparse
import json
import os
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.analysis.spectral import apply_transform
from src.training.probes import run_probe
from src.visualization.spectral_profile import learned_weight_profile


MIN_SAMPLES_PER_CLASS = 10
RESULTS_DIR_DEFAULT = "results/band_spectral"


def _load_npz_generic(path: str, label_key: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "embeddings" not in data.files:
        raise KeyError(f"NPZ at {path} must contain 'embeddings' array.")
    if label_key not in data.files:
        raise KeyError(
            f"NPZ at {path} does not contain label key '{label_key}'. "
            f"Available keys: {list(data.files)}"
        )
    X = data["embeddings"]  # (B, F, T)
    y = data[label_key].astype(str)
    if X.ndim != 3:
        raise ValueError(f"Expected embeddings of shape (B, F, T), got {X.shape}")
    return X, y


def _safe_suffix(name: str) -> str:
    return str(name).replace(" ", "_").replace("/", "_")


def _build_bands(n_coeffs: int, band_width: int) -> list[tuple[int, int]]:
    """
    Construct linearly spaced bands over [0, n_coeffs), each of width band_width.

    The final (partial) band is dropped so that each band has identical width.
    This matches the CLAP FFT analysis convention where, e.g. 33 coeffs with
    band_width=4 yield 8 bands using coeffs 0..31 and leave the last bin unused.
    """
    num_bands = n_coeffs // band_width
    bands = []
    for b in range(num_bands):
        start = b * band_width
        end = (b + 1) * band_width
        bands.append((start, end))
    return bands


def _aggregate_into_bands(coeffs: np.ndarray, bands: list[tuple[int, int]]) -> np.ndarray:
    """
    Aggregate FFT magnitudes into bands by averaging over bins within each band.

    Parameters
    ----------
    coeffs : np.ndarray
        Array of shape (B, F, K) containing per-bin magnitudes.
    bands : list of (start, end)
        Inclusive-exclusive index ranges along the last axis.

    Returns
    -------
    band_coeffs : np.ndarray
        Shape (B, F, n_bands), where each slice [:, :, b] is the mean magnitude
        over coeffs[:, :, start:end] for the corresponding band.
    """
    B, F, _ = coeffs.shape
    band_vals = []
    for start, end in bands:
        # Mean over frequency bins within the band, keep a singleton band axis.
        band_mean = coeffs[:, :, start:end].mean(axis=2, keepdims=True)  # (B, F, 1)
        band_vals.append(band_mean)
    band_coeffs = np.concatenate(band_vals, axis=2)  # (B, F, n_bands)
    return band_coeffs


def _compute_and_plot_band_profiles(
    X: np.ndarray,
    y: np.ndarray,
    label_name: str,
    out_dir: str,
    prefix: str,
    band_width: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    B, F, T = X.shape
    # rFFT along time axis
    coeffs = apply_transform(X, transform_type="fft", axis=2, window_type=None)
    n_coeffs = coeffs.shape[2]

    bands = _build_bands(n_coeffs=n_coeffs, band_width=band_width)
    n_bands = len(bands)
    if n_bands == 0:
        raise ValueError(
            f"band_width={band_width} too large for n_coeffs={n_coeffs} "
            f"(need at least one full band)."
        )

    # Aggregate per-bin magnitudes into bands.
    band_coeffs = _aggregate_into_bands(coeffs, bands)  # (B, F, n_bands)

    classes = sorted(set(y.tolist()))
    counts = {c: int((y == c).sum()) for c in classes}
    print(f"[run_band_spectral] Found {len(classes)} {label_name}: {counts}")

    # Flatten band-wise features for probing: (B, F * n_bands)
    X_flat = band_coeffs.reshape(B, -1)

    mean_magnitude = {}
    learned_profiles = {}
    probe_accuracies = {}

    for cls in classes:
        mask = y == cls
        n_cls = int(mask.sum())
        if n_cls < MIN_SAMPLES_PER_CLASS:
            print(
                f"[run_band_spectral] Skipping {label_name} '{cls}': "
                f"only {n_cls} samples (< {MIN_SAMPLES_PER_CLASS})."
            )
            continue

        print(
            f"[run_band_spectral] Processing {label_name}='{cls}' "
            f"(n={n_cls})..."
        )

        # Mean band magnitude over samples of this class and feature dims.
        # band_coeffs: (B, F, n_bands)
        mean_mag = band_coeffs[mask].mean(axis=(0, 1))  # (n_bands,)
        mean_magnitude[cls] = mean_mag.tolist()

        # One-vs-rest binary labels built on the same full set as in FFT-bin probing.
        y_binary = (y == cls).astype(int)
        if np.sum(y_binary) < MIN_SAMPLES_PER_CLASS or len(np.unique(y_binary)) < 2:
            print(
                f"[run_band_spectral] Skipping probing for {cls}: "
                f"insufficient positives or only one class."
            )
            continue

        acc, final_model, _ = run_probe(X_flat, y_binary)
        probe_accuracies[cls] = float(acc)

        # Learned band-level spectral profile: treat each band as a "frequency bin".
        prof = learned_weight_profile(
            final_model, n_coeffs=n_bands, n_features=F
        )
        learned_profiles[cls] = prof.tolist()

        # Plot per-class band-level spectral profile (mean magnitude + learned weight).
        x = np.arange(n_bands)
        fig, (ax_mag, ax_w) = plt.subplots(
            1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 1]}
        )

        ax_mag.plot(x, mean_mag, label=f"{cls} (n={n_cls})")
        ax_mag.set_xlabel("Band index")
        ax_mag.set_ylabel("Mean band magnitude")
        ax_mag.legend()
        ax_mag.set_title(f"Mean band magnitude — {cls}")

        ax_w.plot(x, prof, label=f"{cls} (acc={acc:.3f})")
        ax_w.set_xlabel("Band index")
        ax_w.set_ylabel("Normalized weight")
        ax_w.set_ylim(0.0, 1.0)
        ax_w.legend()
        ax_w.set_title(f"Band-level spectral profile — {cls}")

        fig.suptitle(
            f"Band-level spectral probing — {label_name}='{cls}' "
            f"(F={F}, T={T}, n_coeffs={n_coeffs}, n_bands={n_bands}, band_width={band_width})",
            fontsize=11,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.94])

        cls_suffix = _safe_suffix(cls)
        png_path = os.path.join(
            out_dir,
            f"{prefix}_band_spectral_profile_{label_name}_{cls_suffix}.png",
        )
        plt.savefig(png_path)
        plt.close(fig)
        print(f"[run_band_spectral] Saved plot to {png_path}")

    summary = {
        "shape": {
            "B": int(B),
            "F": int(F),
            "T": int(T),
            "n_coeffs": int(n_coeffs),
            "n_bands": int(n_bands),
            "band_width": int(band_width),
        },
        "bands": [{"start": int(s), "end": int(e)} for (s, e) in bands],
        "label_name": label_name,
        "class_counts": counts,
        "probe_accuracies": probe_accuracies,
        "mean_band_magnitude_profiles": mean_magnitude,
        "learned_band_weight_profiles": learned_profiles,
    }
    json_path = os.path.join(
        out_dir, f"{prefix}_band_spectral_summary_{label_name}.json"
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[run_band_spectral] Saved JSON summary to {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run band-level spectral probing and plot band spectral profiles "
            "from a generic embeddings NPZ (B, F, T)."
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
        "--results_dir",
        type=str,
        default=RESULTS_DIR_DEFAULT,
        help="Directory to store JSON summaries and plots.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="band",
        help="Filename prefix for outputs (useful when running multiple configs).",
    )
    parser.add_argument(
        "--band_width",
        type=int,
        default=4,
        help=(
            "Number of adjacent FFT bins per band. "
            "Matches CLAP FFT analysis default of 4."
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print(
        f"[run_band_spectral] Loading embeddings from {args.npz_path} "
        f"(label_key='{args.label_key}', band_width={args.band_width})"
    )
    X, y = _load_npz_generic(args.npz_path, label_key=args.label_key)
    print(f"[run_band_spectral] X.shape={X.shape}, n_labels={len(y)}")

    _compute_and_plot_band_profiles(
        X=X,
        y=y,
        label_name=args.label_key.rstrip("s"),
        out_dir=args.results_dir,
        prefix=args.prefix,
        band_width=args.band_width,
    )


if __name__ == "__main__":
    main()

