"""
High-dimensional spectral visualisation (Task 2).

Steps:
- load_data -> X (B, 1536, 64), genres
- apply_transform (FFT) -> X_freq (B, 1536, 33)
- compute per-dimension mean spectra and plot heatmap
- cluster dimensions by spectral shape and plot cluster mean spectra
- for a few representative dims, plot per-sample spectra curves
"""
import argparse
import os

import numpy as np

from src.data_processing.loader import load_data
from src.analysis.spectral import apply_transform
from src.visualization.spectral_highdim import (
    cluster_dimensions_kmeans,
    compute_dimension_spectrum_matrix,
    plot_cluster_mean_spectra,
    plot_heatmap_dim_x_freq,
    plot_per_sample_curves_for_dims,
    plot_3d_sample_freq_surface,
    plot_3d_dim_freq_surface,
)


EMBEDDINGS_PATH = "data_artifacts/clap_embeddings_t64.npz"
RESULTS_DIR = "results/highdim_spectral"
N_COEFFS = 33


def main():
    parser = argparse.ArgumentParser(description="Run high-dimensional spectral visualisation.")
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default=EMBEDDINGS_PATH,
        help="Path to embeddings NPZ.",
    )
    parser.add_argument(
        "--genres",
        nargs="*",
        default=["Electronic", "Experimental", "Folk", "Hip-Hop", "Instrumental", "International", "Pop", "Rock"],
        help="Genres to use for per-sample and single-sample plots.",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=8,
        help="Number of clusters for KMeans on dimension spectra.",
    )
    parser.add_argument(
        "--plot_sample_surface",
        action="store_true",
        help="If set, plot a 3D surface (sample index × frequency × mean-over-dims) for up to 200 samples.",
    )
    parser.add_argument(
        "--per_sample_heatmap",
        action="store_true",
        help="If set, for each genre randomly pick one sample and plot its full dim×freq FFT magnitude heatmap (no averaging over dims).",
    )
    parser.add_argument(
        "--per_sample_3d",
        action="store_true",
        help="If set, for each genre randomly pick samples and plot 3D dim×freq surfaces (no averaging over dims).",
    )
    parser.add_argument(
        "--per_sample_n",
        type=int,
        default=3,
        help="Number of random samples per genre for single-sample 2D/3D visualisations (default: 3).",
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("[run_highdim_spectral] Loading data...")
    X, y_str = load_data(args.embeddings_path)  # (B, 1536, 64)
    print(f"[run_highdim_spectral] X.shape={X.shape}")

    print("[run_highdim_spectral] Applying FFT transform...")
    X_freq = apply_transform(X, transform_type="fft", axis=2, window_type=None)  # (B, 1536, 33)
    assert X_freq.shape[2] == N_COEFFS, f"Expected {N_COEFFS} coeffs, got {X_freq.shape[2]}"

    print("[run_highdim_spectral] Computing per-dimension spectra matrix...")
    M_mean, M_std = compute_dimension_spectrum_matrix(X_freq)
    print(f"[run_highdim_spectral] M_mean.shape={M_mean.shape}")

    # Heatmap over all dims
    heatmap_path = os.path.join(RESULTS_DIR, "heatmap_dims_x_freq.png")
    plot_heatmap_dim_x_freq(
        M_mean,
        heatmap_path,
        title="Per-dimension mean FFT magnitude (all genres, dims=1536)",
    )
    print(f"[run_highdim_spectral] Saved heatmap to {heatmap_path}")

    if args.plot_sample_surface:
        print("[run_highdim_spectral] Plotting 3D sample x freq surface...")
        surface_path = os.path.join(RESULTS_DIR, "surface_sample_x_freq_mean_over_dims.png")
        plot_3d_sample_freq_surface(
            X_freq,
            surface_path,
            title="3D surface: sample x frequency x mean FFT magnitude over dims",
            max_samples=200,
        )
        print(f"[run_highdim_spectral] Saved 3D surface to {surface_path}")

    # Cluster dimensions by spectral pattern
    print(f"[run_highdim_spectral] Clustering dimensions into {args.n_clusters} clusters...")
    labels = cluster_dimensions_kmeans(M_mean, n_clusters=args.n_clusters)
    cluster_plot_path = os.path.join(RESULTS_DIR, "cluster_mean_spectra.png")
    plot_cluster_mean_spectra(
        M_mean,
        labels,
        cluster_plot_path,
        title="Cluster mean spectra (dimension clusters)",
    )
    print(f"[run_highdim_spectral] Saved cluster mean spectra to {cluster_plot_path}")

    # For a couple of representative dimensions, plot per-sample curves for selected genres
    energy = np.linalg.norm(M_mean, axis=1)
    strongest_dims = np.argsort(-energy)[:3].tolist()
    weakest_dims = np.argsort(energy)[:3].tolist()
    rep_dims = strongest_dims + weakest_dims
    print(f"[run_highdim_spectral] Representative dims (strongest+weakest): {rep_dims}")

    for genre in args.genres:
        mask = y_str == genre
        if not np.any(mask):
            print(f"[run_highdim_spectral] No samples for genre {genre}; skipping per-sample curves.")
            continue
        out_path = os.path.join(
            RESULTS_DIR,
            f"per_sample_curves_dims_{'_'.join(map(str, rep_dims))}_{genre.replace(' ', '_').replace('/', '_')}.png",
        )
        title = f"Per-sample spectra for dims {rep_dims} (genre={genre})"
        plot_per_sample_curves_for_dims(X_freq, rep_dims, mask, out_path, title, max_samples=200)
        print(f"[run_highdim_spectral] Saved per-sample curves to {out_path}")

    # Optional: single-sample dim×freq heatmap / 3D per genre (no averaging over dims)
    if args.per_sample_heatmap or args.per_sample_3d:
        rng = np.random.default_rng(42)
        for genre in args.genres:
            mask = y_str == genre
            indices = np.where(mask)[0]
            if indices.size == 0:
                print(f"[run_highdim_spectral] No samples for genre {genre}; skipping single-sample visualisations.")
                continue
            n_pick = min(args.per_sample_n, indices.size)
            picked = rng.choice(indices, size=n_pick, replace=False)
            clean_genre = genre.replace(" ", "_").replace("/", "_")
            for b in picked:
                S = X_freq[b]  # (n_dims, n_coeffs) = (1536, 33)
                if args.per_sample_heatmap:
                    out_path = os.path.join(
                        RESULTS_DIR,
                        f"single_sample_dim_freq_{clean_genre}_idx{b}.png",
                    )
                    plot_heatmap_dim_x_freq(
                        S,
                        out_path,
                        title=f"Single-sample dim×freq FFT magnitude (genre={genre}, sample_idx={b})",
                        normalize_per_dim=True,
                        sort_by_energy=True,
                    )
                    print(f"[run_highdim_spectral] Saved single-sample heatmap for {genre} to {out_path}")

                if args.per_sample_3d:
                    out_path_3d = os.path.join(
                        RESULTS_DIR,
                        f"single_sample_dim_freq_3d_{clean_genre}_idx{b}.png",
                    )
                    plot_3d_dim_freq_surface(
                        S,
                        out_path_3d,
                        title=f"Single-sample 3D dim×freq surface (genre={genre}, sample_idx={b})",
                        dim_step=4,
                    )
                    print(f"[run_highdim_spectral] Saved single-sample 3D surface for {genre} to {out_path_3d}")


if __name__ == "__main__":
    main()

