"""
Compute spectral probing profiles using PaSST embeddings NPZ files.

This script is a light-weight reimplementation of the CLAP-based spectral probing,
but operating directly on PaSST timestamp embeddings saved as:

    - embeddings: np.ndarray, shape (B, F_passt, T)
    - labels:     np.ndarray[str], e.g. 'genres' or 'composers'

It will:
    1. Load the NPZ.
    2. Run rFFT along the time axis.
    3. For each class (genre / composer) with enough samples, train a one-vs-rest
       logistic regression probe on flattened spectral features.
    4. Derive the learned spectral profile (weight per frequency) and save:
         - a JSON summary with spectral profiles and accuracies;
         - a PNG plot visualising mean magnitude and learned weight curves.
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
RESULTS_DIR_DEFAULT = "results/passt_spectral"


def _load_passt_npz(path: str, label_key: str) -> Tuple[np.ndarray, np.ndarray]:
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


def _compute_and_plot_profiles(
    X: np.ndarray,
    y: np.ndarray,
    label_name: str,
    out_dir: str,
    prefix: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    B, F, T = X.shape
    # rFFT along time axis
    coeffs = apply_transform(X, transform_type="fft", axis=2, window_type=None)
    if coeffs.shape[2] != (T // 2 + 1):
        # Not fatal, but warn in case conventions change.
        print(
            f"[run_passt_spectral] Warning: coeffs.shape[2]={coeffs.shape[2]} "
            f"!= T//2+1={T//2+1}"
        )
    n_coeffs = coeffs.shape[2]
    n_features = F

    classes = sorted(set(y.tolist()))
    counts = {c: int((y == c).sum()) for c in classes}
    print(f"[run_passt_spectral] Found {len(classes)} {label_name}: {counts}")

    X_flat = coeffs.reshape(B, -1)

    mean_magnitude = {}
    learned_profiles = {}
    probe_accuracies = {}

    for cls in classes:
        mask = y == cls
        n_cls = int(mask.sum())
        if n_cls < MIN_SAMPLES_PER_CLASS:
            print(
                f"[run_passt_spectral] Skipping {label_name} '{cls}': "
                f"only {n_cls} samples (< {MIN_SAMPLES_PER_CLASS})."
            )
            continue

        print(
            f"[run_passt_spectral] Processing {label_name}='{cls}' "
            f"(n={n_cls})..."
        )

        # Mean magnitude over samples of this class and feature dims.
        mean_mag = coeffs[mask].mean(axis=(0, 1))  # (n_coeffs,)
        mean_magnitude[cls] = mean_mag.tolist()

        # One-vs-rest binary labels.
        y_binary = (y == cls).astype(int)
        if np.sum(y_binary) < MIN_SAMPLES_PER_CLASS or len(np.unique(y_binary)) < 2:
            print(
                f"[run_passt_spectral] Skipping probing for {cls}: "
                f"insufficient positives or only one class."
            )
            continue

        acc, final_model, _ = run_probe(X_flat, y_binary)
        probe_accuracies[cls] = float(acc)

        prof = learned_weight_profile(
            final_model, n_coeffs=n_coeffs, n_features=n_features
        )
        learned_profiles[cls] = prof.tolist()

        # Plot per-class spectral profile (mean magnitude + learned weight).
        x = np.arange(n_coeffs)
        fig, (ax_mag, ax_w) = plt.subplots(
            1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 1]}
        )

        ax_mag.plot(x, mean_mag, label=f"{cls} (n={n_cls})")
        ax_mag.set_xlabel("FFT coefficient index")
        ax_mag.set_ylabel("Mean magnitude")
        ax_mag.legend()
        ax_mag.set_title(f"Mean FFT magnitude — {cls}")

        ax_w.plot(x, prof, label=f"{cls} (acc={acc:.3f})")
        ax_w.set_xlabel("FFT coefficient index")
        ax_w.set_ylabel("Normalized weight")
        ax_w.set_ylim(0.0, 1.0)
        ax_w.legend()
        ax_w.set_title(f"Spectral profile (learned weight) — {cls}")

        fig.suptitle(
            f"PaSST spectral probing — {label_name}='{cls}' "
            f"(F={F}, T={T}, n_coeffs={n_coeffs})",
            fontsize=11,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.94])

        cls_suffix = _safe_suffix(cls)
        png_path = os.path.join(
            out_dir, f"{prefix}_spectral_profile_{label_name}_{cls_suffix}.png"
        )
        plt.savefig(png_path)
        plt.close(fig)
        print(f"[run_passt_spectral] Saved plot to {png_path}")

    summary = {
        "shape": {"B": int(B), "F": int(F), "T": int(T), "n_coeffs": int(n_coeffs)},
        "label_name": label_name,
        "class_counts": counts,
        "probe_accuracies": probe_accuracies,
        "mean_magnitude_profiles": mean_magnitude,
        "learned_weight_profiles": learned_profiles,
    }
    json_path = os.path.join(out_dir, f"{prefix}_spectral_summary_{label_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[run_passt_spectral] Saved JSON summary to {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run spectral probing and plot spectral profiles from a PaSST "
            "embeddings NPZ."
        )
    )
    parser.add_argument(
        "--npz_path",
        type=str,
        required=True,
        help=(
            "Path to PaSST embeddings NPZ "
            "(e.g. data_artifacts/passt_embeddings_t64.npz or "
            "data_artifacts/passt_embeddings_asap_t32.npz)."
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
        default="passt",
        help="Filename prefix for outputs (useful when running multiple configs).",
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print(
        f"[run_passt_spectral] Loading embeddings from {args.npz_path} "
        f"(label_key='{args.label_key}')"
    )
    X, y = _load_passt_npz(args.npz_path, label_key=args.label_key)
    print(f"[run_passt_spectral] X.shape={X.shape}, n_labels={len(y)}")

    _compute_and_plot_profiles(
        X=X,
        y=y,
        label_name=args.label_key.rstrip("s"),
        out_dir=args.results_dir,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()

