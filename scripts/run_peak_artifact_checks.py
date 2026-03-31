"""
Entry script for Task 1: investigate the origin of the 3 peaks.

Runs:
- Single-sample FFT sanity check (manual rfft vs apply_transform).
- Random Gaussian FFT spectra (to see if 3 peaks are algorithmic artifacts).
- Per-sample spectra for a few representative genres (to see if peaks are pervasively present).
- Learned spectral profiles (probe weights per frequency) for the same genres,
  using src/visualization/spectral_profile.learned_weight_profile.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.analysis.peak_artifact_checks import (
    EMBEDDINGS_PATH,
    N_COEFFS,
    describe_band_layout,
    plot_per_sample_spectra_for_genre,
    run_layout_variants_experiment,
    run_random_vector_fft_pipeline,
    run_single_sample_fft_check,
    run_time_permutation_experiment,
)
from src.data_processing.loader import load_data
from src.analysis.spectral import apply_transform
from src.training.probes import run_probe
from src.visualization.spectral_profile import learned_weight_profile


OUTPUT_DIR = "results/peak_artifact_investigation"


def _plot_learned_profiles_for_genres(genres):
    """For the given genres, train FFT probes and plot learned spectral profiles."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X, y_str = load_data(EMBEDDINGS_PATH)

    profiles = {}
    for genre in genres:
        # one-vs-rest on the full dataset,保持和主 FFT 分析脚本一致
        y_binary = (y_str == genre).astype(int)
        if np.sum(y_binary) < 10 or len(np.unique(y_binary)) < 2:
            print(
                f"[run_peak_artifact_checks] Skipping learned profile for {genre}: "
                f"need >=10 positives and both classes."
            )
            continue

        coeffs = apply_transform(X, transform_type="fft", axis=2, window_type=None)
        X_flat = coeffs.reshape(coeffs.shape[0], -1)
        acc, model, _ = run_probe(X_flat, y_binary, random_state=42)
        n_features = coeffs.shape[1]  # 1536
        prof = learned_weight_profile(model, N_COEFFS, n_features=n_features)
        profiles[genre] = (prof, acc)
        print(f"[run_peak_artifact_checks] Learned profile for {genre}: acc={acc:.4f}")

    if not profiles:
        print("[run_peak_artifact_checks] No learned profiles to plot.")
        return

    x = np.arange(N_COEFFS)
    fig, ax = plt.subplots(figsize=(8, 4))
    for genre, (prof, acc) in profiles.items():
        ax.plot(x, prof, label=f"{genre} (acc={acc:.2f})")

    ax.set_xlabel("Frequency coefficient")
    ax.set_ylabel("Learned weight (normalized)")
    ax.set_xticks([0, N_COEFFS // 2, N_COEFFS - 1])
    ax.set_xticklabels(["L", "M", "H"])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Learned spectral profiles across genres (FFT probe)")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "learned_spectral_profiles_selected_genres.png")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"[run_peak_artifact_checks] Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Run peak artifact checks (Task 1).")
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default=EMBEDDINGS_PATH,
        help="Path to NPZ with embeddings (default: data_artifacts/clap_embeddings_t64.npz)",
    )
    parser.add_argument(
        "--genres",
        nargs="*",
        default=["Rock", "Pop", "Electronic", "Experimental"],
        help="Subset of genres to visualise per-sample spectra for.",
    )
    parser.add_argument(
        "--sample_idx", type=int, default=0, help="Sample index for single-sample FFT check."
    )
    parser.add_argument(
        "--dim_idx", type=int, default=0, help="Embedding dimension index for single-sample FFT check."
    )
    args = parser.parse_args()

    # 1) Implementation / numeric sanity: single-sample FFT
    run_single_sample_fft_check(
        embeddings_path=args.embeddings_path,
        sample_idx=args.sample_idx,
        dim_idx=args.dim_idx,
        window_type=None,
    )

    # 2) Random Gaussian pipeline to see if 3 peaks are algorithmic
    run_random_vector_fft_pipeline(n_trials=5, shape=(1, 1536, 64), window_type=None)

    # 2.5) Band layout / coverage description
    describe_band_layout()

    # 3) Per-sample spectra for a few genres (no window for now)
    for g in args.genres:
        try:
            plot_per_sample_spectra_for_genre(g, max_samples=200, window_type=None)
        except ValueError as e:
            print(f"[run_peak_artifact_checks] Skipping genre {g}: {e}")

    # 4) Learned spectral profiles (using src/visualization/spectral_profile)
    _plot_learned_profiles_for_genres(args.genres)

    # 5) Time permutation experiment (subset of samples)
    run_time_permutation_experiment(max_samples=500, random_state=42, window_type=None)

    # 6) Layout variants experiment (baseline vs time-only FFT)
    run_layout_variants_experiment(max_samples=500, random_state=42, window_type=None)


if __name__ == "__main__":
    main()

