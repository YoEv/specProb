#!/usr/bin/env python3

"""
Confusion-matrix–focused analysis script.

This script ties together the following components:
- Data loading        -> src.data_processing.loader.load_data
- Spectral transforms -> src.analysis.spectral.apply_transform
- Probing             -> src.training.probes.run_probe_with_predictions
- Metrics             -> src.analysis.metrics.compute_confusion
- Visualisation       -> src.visualization.confusion.{plot_confusion_matrix, plot_confusion_grid}

It generates:
- Full multi-class confusion matrices for time-domain and spectral features.
- Band-wise confusion matrices for spectral features, arranged as a heatmap grid.
"""

import argparse
import os
from typing import List

import numpy as np

from src.data_processing.loader import load_data
from src.analysis.spectral import apply_transform
from src.training.probes import run_probe_with_predictions
from src.analysis.metrics import compute_confusion
from src.visualization.confusion import plot_confusion_matrix, plot_confusion_grid


EMBEDDINGS_PATH = "data_artifacts/clap_embeddings_t64.npz"
OUTPUT_DIR = "results/confusion_analysis"
RANDOM_STATE = 42


def compute_time_domain_confusion(
    X: np.ndarray,
    y_str: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, list[str], float]:
    """Flatten raw embeddings over all non-batch dims and run a probe."""
    n_samples = X.shape[0]
    X_flat = X.reshape(n_samples, -1)

    accuracy, _, _, y_test, y_pred, label_names = run_probe_with_predictions(
        X_flat,
        y_str,
        random_state=random_state,
    )
    cm, labels = compute_confusion(y_test, y_pred, label_names)
    return cm, labels, accuracy


def compute_spectral_confusions(
    X: np.ndarray,
    y_str: np.ndarray,
    transform_type: str,
    n_bands: int,
    random_state: int,
) -> tuple[np.ndarray, list[str], float, list[np.ndarray], list[str]]:
    """
    Compute:
    - Full-spectrum confusion matrix for spectral features.
    - Band-wise confusion matrices by splitting the spectral coefficients.
    """
    # X is expected to be (B, F, T); transform along the temporal axis.
    coeffs = apply_transform(X, transform_type=transform_type, axis=2)
    n_samples, n_features, n_coeffs = coeffs.shape

    # Full-spectrum probe
    X_full = coeffs.reshape(n_samples, -1)
    full_acc, _, _, y_test_full, y_pred_full, label_names = run_probe_with_predictions(
        X_full,
        y_str,
        random_state=random_state,
    )
    full_cm, labels = compute_confusion(y_test_full, y_pred_full, label_names)

    # Band-wise probes along the coefficient dimension
    band_cms: list[np.ndarray] = []
    band_titles: list[str] = []

    band_size = n_coeffs // n_bands
    for b in range(n_bands):
        start = b * band_size
        end = (b + 1) * band_size if b < n_bands - 1 else n_coeffs
        band_coeffs = coeffs[:, :, start:end]
        X_band = band_coeffs.reshape(n_samples, -1)

        band_acc, _, _, y_test_band, y_pred_band, _ = run_probe_with_predictions(
            X_band,
            y_str,
            random_state=random_state,
        )
        band_cm, _ = compute_confusion(y_test_band, y_pred_band, label_names)

        band_cms.append(band_cm)
        band_titles.append(f"Band {b} [{start}:{end}] (acc={band_acc:.2f})")

    return full_cm, labels, full_acc, band_cms, band_titles


def save_detailed_spectral_confusions(
    cm: np.ndarray,
    labels: list[str],
    output_dir: str,
    transform_type: str,
    top_k: int = 3,
) -> None:
    """
    For each genre, create a detailed spectral confusion matrix showing the
    target genre against all available genres (full set of classes).
    """
    os.makedirs(output_dir, exist_ok=True)
    n_classes = len(labels)

    for i in range(n_classes):
        target_label = labels[i]
        # Reorder so that the target genre appears first, followed by all others.
        idxs = [i] + [j for j in range(n_classes) if j != i]
        cm_sub = cm[np.ix_(idxs, idxs)]
        sub_labels = [labels[j] for j in idxs]

        transform_name = transform_type.upper()
        title = f"{transform_name} detailed confusion for {target_label}"
        clean_label = target_label.replace(" ", "_").replace("/", "_")
        save_path = os.path.join(
            output_dir,
            f"detailed_confusion_{transform_type}_{clean_label}.png",
        )
        plot_confusion_matrix(cm_sub, sub_labels, title, save_path)


def save_band_confusions(
    band_cms: list[np.ndarray],
    labels: list[str],
    band_titles: list[str],
    output_dir: str,
    transform_type: str,
) -> None:
    """
    Save one full 8x8 confusion matrix per spectral band.

    Each image corresponds to a specific band (e.g. low to high frequency),
    and compares all classes within that band.
    """
    os.makedirs(output_dir, exist_ok=True)
    transform_name = transform_type.upper()

    for idx, cm_band in enumerate(band_cms):
        if idx < len(band_titles):
            subtitle = band_titles[idx]
        else:
            subtitle = f"Band {idx}"

        title = f"{transform_name} confusion - {subtitle}"
        safe_band = subtitle.replace(" ", "_").replace("/", "_").replace("[", "").replace("]", "").replace(":", "-")
        filename = f"confusion_{transform_type}_band_{idx}_{safe_band}.png"
        save_path = os.path.join(output_dir, filename)
        plot_confusion_matrix(cm_band, labels, title, save_path)


def main():
    parser = argparse.ArgumentParser(
        description="Run confusion-matrix analysis for time vs spectral probes."
    )
    parser.add_argument(
        "--embeddings-path",
        type=str,
        default=EMBEDDINGS_PATH,
        help="Path to the NPZ file containing embeddings and genres.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory to store confusion matrix figures.",
    )
    parser.add_argument(
        "--transform-type",
        type=str,
        default="fft",
        choices=["dct", "fft", "dft"],
        help="Spectral transform to apply for the frequency-domain probe.",
    )
    parser.add_argument(
        "--n-bands",
        type=int,
        default=8,
        help="Number of spectral bands to split coefficients into.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed for train/test splits.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading embeddings from {args.embeddings_path}...")
    X, y_str = load_data(args.embeddings_path)
    print(f"Embeddings shape after loading: {X.shape}")
    print(f"Number of samples: {X.shape[0]}")

    # --- 1. Time-domain confusion ---
    print("\n[1/3] Computing time-domain confusion matrix...")
    time_cm, labels, time_acc = compute_time_domain_confusion(
        X,
        y_str,
        random_state=args.random_state,
    )
    time_title = f"Time-domain probe (acc={time_acc:.2f})"
    time_path = os.path.join(args.output_dir, "confusion_time_domain.png")
    plot_confusion_matrix(time_cm, labels, time_title, time_path)
    print(f"Saved time-domain confusion matrix to {time_path}")

    # --- 2. Spectral full-spectrum + band-wise confusion ---
    print("\n[2/3] Computing spectral (full-spectrum) confusion matrix...")
    (
        freq_cm,
        labels_freq,
        freq_acc,
        band_cms,
        band_titles,
    ) = compute_spectral_confusions(
        X,
        y_str,
        transform_type=args.transform_type,
        n_bands=args.n_bands,
        random_state=args.random_state,
    )
    assert labels == labels_freq, "Label ordering mismatch between time and spectral probes."

    freq_title = f"{args.transform_type.upper()} probe (acc={freq_acc:.2f})"
    freq_path = os.path.join(
        args.output_dir,
        f"confusion_{args.transform_type}_full_spectrum.png",
    )
    plot_confusion_matrix(freq_cm, labels_freq, freq_title, freq_path)
    print(f"Saved spectral full-spectrum confusion matrix to {freq_path}")

    # --- 3. Band-wise spectral confusion matrices (one figure per band) ---
    print("\n[3/3] Generating band-wise spectral confusion matrices...")
    save_band_confusions(
        band_cms,
        labels,
        band_titles,
        output_dir=args.output_dir,
        transform_type=args.transform_type,
    )
    print("Saved band-wise spectral confusion matrices.")


if __name__ == "__main__":
    main()

