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

# Ensure the project root is on sys.path when invoked as
# `python scripts/run_generic_fft_band_probe.py ...`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.spectral import apply_transform
from src.training.probes import run_probe
from src.visualization.spectral_profile import learned_weight_profile


warnings.filterwarnings("ignore")


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
    X_full_flat = coeffs.reshape(coeffs.shape[0], -1)
    orig_accuracy, final_model, _ = run_probe(
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
    for b in range(num_bands):
        band_coeffs = coeffs[:, :, bands[b]].reshape(coeffs.shape[0], -1)
        acc, _, _ = run_probe(
            band_coeffs, y_binary, random_state=args.random_state
        )
        band_accuracies.append(float(acc))

    # Cumulative bands from low to high.
    cumulative_accuracies: list[float] = []
    cumulative_coeffs = coeffs[:, :, bands[0]].reshape(coeffs.shape[0], -1)
    acc, _, _ = run_probe(
        cumulative_coeffs, y_binary, random_state=args.random_state
    )
    cumulative_accuracies.append(float(acc))

    for b in range(1, num_bands):
        next_band_coeffs = coeffs[:, :, bands[b]].reshape(coeffs.shape[0], -1)
        cumulative_coeffs = np.concatenate(
            (cumulative_coeffs, next_band_coeffs), axis=-1
        )
        acc, _, _ = run_probe(
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

