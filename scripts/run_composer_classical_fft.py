import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analysis.spectral import apply_transform
from src.config.spectral_experiments import get_embedding_config
from src.features.extraction import (
    EMBEDDINGS_ASAP_FILE,
    extract_embeddings_asap_composer,
    load_model,
)
from src.training.probes import run_probe
from src.visualization.spectral_profile import learned_weight_profile
from src.visualization.plotting import create_and_save_plot


AUDIO_DIR = "/home/evev/noiseloss/datasets/D_asap_100"
RESULTS_DIR = "results/composer_classical"
# Only these five classical composers are considered for Task 4.
TARGET_COMPOSERS = ["Bach", "Beethoven", "Liszt", "Schubert", "Chopin"]


def _log(msg: str):
    print(f"[composer_fft] {msg}")


def _ensure_embeddings(npz_path: str):
    """
    Ensure that ASAP composer embeddings exist on disk.

    If the NPZ does not exist yet, run CLAP over D_asap_100 and save:
        - embeddings: (B, 768, 2, 32)
        - composers: array[str]
        - file_paths: array[str]
    """
    if os.path.exists(npz_path):
        _log(f"Found existing embeddings at {npz_path}")
        return

    from transformers import logging as hf_logging
    import torch

    hf_logging.set_verbosity_error()
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(f"Embeddings NPZ not found. Extracting ASAP composer embeddings on device={device}...")
    model, processor = load_model()
    if model is None or processor is None:
        raise RuntimeError("Failed to load CLAP model/processor for ASAP composer extraction.")

    embeddings, composers, file_paths = extract_embeddings_asap_composer(
        AUDIO_DIR, model, processor, device
    )
    _log(f"Saving ASAP composer embeddings with shape {embeddings.shape} to {npz_path}")
    np.savez_compressed(
        npz_path,
        embeddings=embeddings,
        composers=np.array(composers),
        file_paths=np.array(file_paths),
    )


def _load_embeddings(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    X = data["embeddings"]  # (B, 768, 2, 32)
    composers = data["composers"].astype(str)
    file_paths = data["file_paths"].astype(str)
    return X, composers, file_paths


def compute_spectral_profiles():
    """
    Main pipeline for the classical composer Task 4.

    Steps:
        1. Ensure embeddings exist for D_asap_100 (CLAP last_hidden_state).
        2. Reshape to (B, 1536, 32) and run FFT on the time axis (32 → 17 coeffs).
        3. For each composer:
            - compute mean magnitude profile over samples and feature dims;
            - train a one-vs-rest probe and derive learned weight profile.
        4. Save JSON summary and png plots under results/composer_classical/.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg = get_embedding_config("asap_composer")
    npz_path = EMBEDDINGS_ASAP_FILE

    _ensure_embeddings(npz_path)
    X_raw, composers, file_paths = _load_embeddings(npz_path)
    _log(f"Loaded embeddings X_raw.shape={X_raw.shape}, n_tracks={len(composers)}")

    if X_raw.ndim != 4 or X_raw.shape[1:] != (cfg.n_channels, cfg.n_segments, cfg.frames_per_segment):
        raise ValueError(
            f"Expected embeddings shape (B, {cfg.n_channels}, {cfg.n_segments}, {cfg.frames_per_segment}), "
            f"got {X_raw.shape}"
        )

    B = X_raw.shape[0]
    # (B, 768, 2, 32) → (B, 1536, 32)
    X = X_raw.reshape(B, cfg.feature_dim, cfg.frames_per_segment)

    _log("Applying FFT (rfft) along time axis...")
    coeffs = apply_transform(X, transform_type="fft", axis=2, window_type=None)
    # coeffs: (B, 1536, n_coeffs=17)
    if coeffs.shape[2] != cfg.n_coeffs:
        raise ValueError(
            f"Expected n_coeffs={cfg.n_coeffs} along axis=2 after FFT, got {coeffs.shape[2]}"
        )

    all_composers = sorted(set(composers.tolist()))
    counts = {c: int((composers == c).sum()) for c in all_composers}
    _log(f"Composers and counts: {counts}")

    # Restrict analysis to the five target composers and require at least 5 samples.
    composers_unique = [
        c for c in TARGET_COMPOSERS if counts.get(c, 0) >= 5
    ]
    _log(f"Target composers used in profiles (n>=5): {composers_unique}")

    # Flatten for probing.
    X_flat = coeffs.reshape(B, -1)  # (B, 1536 * n_coeffs)
    n_features = cfg.feature_dim

    mean_magnitude = {}
    learned_profiles = {}
    probe_accuracies = {}
    band_specific_acc = {}
    cumulative_acc = {}

    for comp in composers_unique:
        mask = composers == comp
        n_comp = int(mask.sum())

        # Mean magnitude profile over samples and feature dims.
        mean_mag = coeffs[mask].mean(axis=(0, 1))  # (n_coeffs,)
        mean_magnitude[comp] = mean_mag.tolist()

        # Binary labels: this composer vs. others.
        y_binary = (composers == comp).astype(int)
        if np.sum(y_binary) < 5 or len(np.unique(y_binary)) < 2:
            _log(f"Skipping composer {comp} in probing due to insufficient positives or classes.")
            continue

        # Full-spectrum probe.
        orig_accuracy, final_model, _ = run_probe(X_flat, y_binary)
        probe_accuracies[comp] = float(orig_accuracy)

        # Per-band and cumulative band accuracies (as in FMA scripts).
        num_bands = cfg.n_coeffs // 4
        bands = {b: list(range(b * 4, (b + 1) * 4)) for b in range(num_bands)}

        band_accuracies = []
        for b in range(num_bands):
            band_coeffs = coeffs[:, :, bands[b]].reshape(B, -1)
            acc_b, _, _ = run_probe(band_coeffs, y_binary)
            band_accuracies.append(float(acc_b))

        cumulative_accuracies = []
        cumulative_coeffs = coeffs[:, :, bands[0]].reshape(B, -1)
        acc_c, _, _ = run_probe(cumulative_coeffs, y_binary)
        cumulative_accuracies.append(float(acc_c))
        for b in range(1, num_bands):
            next_band_coeffs = coeffs[:, :, bands[b]].reshape(B, -1)
            cumulative_coeffs = np.concatenate((cumulative_coeffs, next_band_coeffs), axis=-1)
            acc_c, _, _ = run_probe(cumulative_coeffs, y_binary)
            cumulative_accuracies.append(float(acc_c))

        auto_accuracy = max(cumulative_accuracies) if cumulative_accuracies else 0.0
        band_specific_acc[comp] = band_accuracies
        cumulative_acc[comp] = cumulative_accuracies

        # Learned weight spectral profile (full spectrum).
        prof = learned_weight_profile(final_model, cfg.n_coeffs, n_features=n_features)
        learned_profiles[comp] = prof.tolist()
        _log(
            f"Composer {comp}: n={n_comp}, full_acc={orig_accuracy:.4f}, "
            f"auto_acc={auto_accuracy:.4f}"
        )

        # Per-composer spectral probing plot (band accuracies + learned weight),
        # reusing the same visualization code as for FMA.
        bar_labels = ["ORIG"] + [f"B{b}" for b in range(num_bands)] + ["AUTO"]
        bar_heights = [orig_accuracy] + band_accuracies + [auto_accuracy]
        colors = ["gray"] + ["lightskyblue"] * num_bands + ["mediumpurple"]
        chance = float(np.mean(y_binary))
        create_and_save_plot(
            target_genre=comp,
            bar_labels=bar_labels,
            bar_heights=bar_heights,
            colors=colors,
            normalized_weights=prof,
            chance=chance,
            n_coeffs=cfg.n_coeffs,
            output_dir=RESULTS_DIR,
        )

    # Save JSON summary.
    summary = {
        "config": {
            "embedding_mode": cfg.name,
            "feature_dim": cfg.feature_dim,
            "frames_per_segment": cfg.frames_per_segment,
            "n_fft": cfg.n_fft,
            "n_coeffs": cfg.n_coeffs,
        },
        "n_samples": int(B),
        "composer_counts": counts,
        "composer_probe_accuracies": probe_accuracies,
        "mean_magnitude_profiles": mean_magnitude,
        "learned_weight_profiles": learned_profiles,
        "composer_band_accuracies": band_specific_acc,
        "composer_cumulative_accuracies": cumulative_acc,
    }
    json_path = os.path.join(RESULTS_DIR, "composer_spectral_summary_fft_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _log(f"Saved JSON summary to {json_path}")

    # Plot mean magnitude profiles.
    fig, ax = plt.subplots(figsize=(8, 5))
    for comp in composers_unique:
        if comp not in mean_magnitude:
            continue
        label = f"{comp} (n={counts[comp]})"
        ax.plot(range(cfg.n_coeffs), mean_magnitude[comp], label=label)
    ax.set_xlabel("FFT coefficient index (0 .. n_coeffs-1)")
    ax.set_ylabel("Mean magnitude")
    ax.set_title("Mean FFT magnitude by composer (ASAP, Task 4)")
    ax.legend()
    fig.tight_layout()
    mean_png = os.path.join(RESULTS_DIR, "composer_mean_magnitude_fft.png")
    fig.savefig(mean_png)
    plt.close(fig)
    _log(f"Saved mean magnitude plot to {mean_png}")

    # Plot learned weight profiles.
    fig, ax = plt.subplots(figsize=(8, 5))
    for comp in composers_unique:
        if comp not in learned_profiles:
            continue
        label = f"{comp} (n={counts[comp]})"
        ax.plot(range(cfg.n_coeffs), learned_profiles[comp], label=label)
    ax.set_xlabel("FFT coefficient index (0 .. n_coeffs-1)")
    ax.set_ylabel("Normalized weight")
    ax.set_title("Learned weight spectral profiles by composer (ASAP, Task 4)")
    ax.legend()
    fig.tight_layout()
    weight_png = os.path.join(RESULTS_DIR, "composer_learned_weight_fft.png")
    fig.savefig(weight_png)
    plt.close(fig)
    _log(f"Saved learned weight plot to {weight_png}")


def main():
    compute_spectral_profiles()


if __name__ == "__main__":
    main()

