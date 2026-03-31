"""
FFT window ablation: for each genre, compare spectral profile and probe accuracy for
no window vs Hann vs Hamming. Saves one total plot and one total metrics JSON per genre
under results/fft_window_ablation/ with genre in the filename.
"""
import json
import os
import time
from datetime import datetime
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_processing.loader import load_data
from src.analysis.spectral import apply_transform
from src.training.probes import run_probe
from src.visualization.spectral_profile import learned_weight_profile


def _log(msg: str, t0: Optional[float] = None) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    elapsed = f" (elapsed {time.time() - t0:.1f}s)" if t0 is not None else ""
    print(f"[{ts}]{elapsed} {msg}", flush=True)


def _genre_to_suffix(genre: str) -> str:
    """Safe filename suffix: e.g. 'Hip-Hop' -> 'Hip-Hop', spaces/slashes -> underscore."""
    return genre.replace(" ", "_").replace("/", "_")


EMBEDDINGS_PATH = "data_artifacts/clap_embeddings_t64.npz"
OUTPUT_DIR = "results/fft_window_ablation"
RANDOM_STATE = 42
MIN_SAMPLES_PER_GENRE = 10

window_configs = [
    (None, "No window (rectangular)"),
    ("hann", "Hann"),
    ("hamming", "Hamming"),
]
n_coeffs = 33


def run_ablation_for_genre(X: np.ndarray, y_binary: np.ndarray, genre: str, t0: float) -> None:
    mean_spectra = []
    learned_weights = []  # per-window learned weight (spectral profile)
    labels = []
    accuracies = []
    n_steps = len(window_configs)
    n_features = X.shape[1]  # 1536

    # Only average magnitude over this genre's samples so each genre gets its own curve
    mask_genre = y_binary.astype(bool)

    for i, (window_type, label) in enumerate(window_configs):
        _log(f"  [{genre}] Step {i + 1}/{n_steps}: {label}...", t0)
        step_start = time.time()
        X_t = apply_transform(X, transform_type="fft", axis=2, window_type=window_type)
        assert X_t.shape[2] == n_coeffs
        # Per-genre mean magnitude: average only over samples of this genre
        mean_spec = np.mean(X_t[mask_genre], axis=(0, 1))
        mean_spectra.append(mean_spec)
        labels.append(label)

        X_flat = X_t.reshape(X_t.shape[0], -1)
        acc, model, _ = run_probe(X_flat, y_binary, random_state=RANDOM_STATE)
        accuracies.append(acc)
        lw = learned_weight_profile(model, n_coeffs, n_features=n_features)
        learned_weights.append(lw)
        _log(f"  [{genre}] Step {i + 1}/{n_steps} done: acc={acc:.4f} ({time.time() - step_start:.1f}s)", t0)

    suffix = _genre_to_suffix(genre)
    x = np.arange(n_coeffs)

    # One figure, two panels: left = mean magnitude, right = learned weight
    fig, (ax_mag, ax_w) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 1]})
    for mean_spec, label in zip(mean_spectra, labels):
        ax_mag.plot(x, mean_spec, label=label)
    ax_mag.set_xlabel("Frequency coefficient")
    ax_mag.set_ylabel("Mean magnitude")
    ax_mag.legend()
    ax_mag.set_xticks([0, n_coeffs // 2, n_coeffs - 1])
    ax_mag.set_xticklabels(["L", "M", "H"])
    n_genre = int(np.sum(mask_genre))
    ax_mag.set_title(f"Mean FFT magnitude by window (n={n_genre})")

    for lw, label in zip(learned_weights, labels):
        ax_w.plot(x, lw, label=label)
    ax_w.set_xlabel("Frequency coefficient")
    ax_w.set_ylabel("Learned weight")
    ax_w.legend()
    ax_w.set_xticks([0, n_coeffs // 2, n_coeffs - 1])
    ax_w.set_xticklabels(["L", "M", "H"])
    ax_w.set_ylim(0, 1)
    ax_w.set_title("Spectral profile (learned weight) by window")

    fig.suptitle(f"FFT window ablation — {genre} task", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(OUTPUT_DIR, f"spectral_profile_by_window_{suffix}.png")
    plt.savefig(out_path)
    plt.close(fig)
    _log(f"  [{genre}] Saved {out_path}", t0)

    # Diagnostic: per-genre mean magnitude should differ (e.g. sum of first 5 coeffs)
    mean_mag_diagnostic = {label: float(np.sum(mean_spectra[i][:5])) for i, label in enumerate(labels)}

    summary = {
        "reference_genre": genre,
        "n_genre_samples": int(np.sum(mask_genre)),
        "accuracies": {labels[i]: accuracies[i] for i in range(len(labels))},
        "mean_magnitude_sum_first5_coeffs": mean_mag_diagnostic,
        "config": {"random_state": RANDOM_STATE},
    }
    json_path = os.path.join(OUTPUT_DIR, f"probe_accuracy_by_window_{suffix}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    _log(f"  [{genre}] Saved {json_path}", t0)


def main():
    t0 = time.time()
    _log("Starting FFT window ablation (all genres).")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    _log("Loading data...", t0)
    X, y_str = load_data(EMBEDDINGS_PATH)
    _log(f"Loaded X.shape={X.shape}", t0)

    all_genres = np.unique(y_str)
    _log(f"Found {len(all_genres)} genres: {', '.join(all_genres)}", t0)

    for genre in all_genres:
        y_binary = (y_str == genre).astype(int)
        if np.sum(y_binary) < MIN_SAMPLES_PER_GENRE:
            _log(f"Skipping {genre}: too few samples ({np.sum(y_binary)}).", t0)
            continue
        if len(np.unique(y_binary)) < 2:
            _log(f"Skipping {genre}: need both classes.", t0)
            continue
        _log(f"Processing genre: {genre}", t0)
        run_ablation_for_genre(X, y_binary, genre, t0)

    _log(f"Done. Total time: {time.time() - t0:.1f}s", t0)


if __name__ == "__main__":
    main()
