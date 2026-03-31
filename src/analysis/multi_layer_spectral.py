import os
from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.analysis.spectral import apply_transform
from src.training.probes import run_probe
from src.visualization.spectral_profile import learned_weight_profile


MULTI_EMBEDDINGS_PATH = "data_artifacts/clap_embeddings_t64_multilayer.npz"
RESULTS_DIR = "results/multi_layer_spectral"
N_COEFFS = 33  # rfft(64)


def load_multi_layer_embeddings(path: str = MULTI_EMBEDDINGS_PATH):
    """
    Load multi-layer embeddings saved by `extract_embeddings_multi_layer`.

    Expected npz keys:
        - embeddings: (B, L, 768, 2, 32)
        - layers: (L,)  # encoder layer indices
        - genres: (B,)
        - track_ids: (B,)

    Returns:
        X: np.ndarray, shape (B, L, 1536, 64)
        layers: np.ndarray, shape (L,)
        genres: np.ndarray, shape (B,)
        track_ids: np.ndarray, shape (B,)
    """
    data = np.load(path, allow_pickle=True)
    emb = data["embeddings"]  # (B, L, 768, 2, 32)
    layers = data["layers"]
    genres = data["genres"]
    track_ids = data["track_ids"]

    B, L, D, G, T = emb.shape
    assert D == 768 and G == 2 and T == 32, f"Unexpected embedding shape: {emb.shape}"
    # Merge (D, G) into feature dim, keep time=64 (2*32)
    emb_reshaped = emb.reshape(B, L, D * G, T)
    return emb_reshaped, layers, genres, track_ids


def compute_multi_layer_spectral_profiles(
    target_genres: Sequence[str],
    path: str = MULTI_EMBEDDINGS_PATH,
    transform_type: str = "fft",
) -> None:
    """
    For each target genre and each encoder layer, run FFT-based probes and compute
    learned spectral profiles.

    This addresses plan 1.2: 对比不同层的 embedding 是否都有相似的三峰结构。
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    X, layers, genres, track_ids = load_multi_layer_embeddings(path)
    B, L, F, T = X.shape
    print(f"[multi_layer_spectral] Loaded multi-layer embeddings: {X.shape}, layers={layers}")

    for genre in target_genres:
        mask = genres == genre
        if np.sum(mask) < 10:
            print(f"[multi_layer_spectral] Skipping {genre}: too few samples ({np.sum(mask)}).")
            continue

        y_binary = (genres == genre).astype(int)
        profiles = []
        accuracies = []

        for li in range(L):
            X_layer = X[:, li, :, :]  # (B, F, T)
            coeffs = apply_transform(X_layer, transform_type=transform_type, axis=2, window_type=None)
            assert coeffs.shape[2] == N_COEFFS, f"Expected {N_COEFFS} coeffs, got {coeffs.shape[2]}"

            X_flat = coeffs.reshape(coeffs.shape[0], -1)
            acc, model, _ = run_probe(X_flat, y_binary, random_state=42)
            lw = learned_weight_profile(model, N_COEFFS, n_features=coeffs.shape[1])
            profiles.append(lw)
            accuracies.append(acc)
            print(
                f"[multi_layer_spectral] Genre={genre}, layer={layers[li]}: acc={acc:.4f}"
            )

        profiles = np.stack(profiles, axis=0)  # (L, N_COEFFS)

        # Save JSON metrics
        clean_genre = genre.replace(" ", "_").replace("/", "_")
        metrics = {
            "genre": genre,
            "layers": layers.tolist(),
            "accuracies": [float(a) for a in accuracies],
            "spectral_profiles": profiles.tolist(),
            "config": {
                "n_coeffs": N_COEFFS,
                "transform_type": transform_type,
            },
        }
        json_path = os.path.join(
            RESULTS_DIR, f"multi_layer_spectral_profiles_{clean_genre}.json"
        )
        import json

        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[multi_layer_spectral] Saved metrics to {json_path}")

        # Plot per-genre multi-layer spectral profiles
        x = np.arange(N_COEFFS)
        fig, ax = plt.subplots(figsize=(8, 4))
        for i in range(L):
            ax.plot(
                x,
                profiles[i],
                label=f"layer {layers[i]} (acc={accuracies[i]:.2f})",
            )

        ax.set_xlabel("Frequency coefficient")
        ax.set_ylabel("Learned weight (normalized)")
        ax.set_xticks([0, N_COEFFS // 2, N_COEFFS - 1])
        ax.set_xticklabels(["L", "M", "H"])
        ax.set_ylim(0, 1)
        ax.legend(fontsize="small")
        ax.set_title(f"Multi-layer spectral profiles — {genre}")
        plt.tight_layout()

        png_path = os.path.join(
            RESULTS_DIR, f"multi_layer_spectral_profiles_{clean_genre}.png"
        )
        plt.savefig(png_path)
        plt.close(fig)
        print(f"[multi_layer_spectral] Saved plot to {png_path}")


if __name__ == "__main__":
    # Example usage: adjust genres as needed when you have a multi-layer npz.
    default_genres: List[str] = ["Rock", "Pop", "Electronic", "Experimental"]
    compute_multi_layer_spectral_profiles(default_genres, path=MULTI_EMBEDDINGS_PATH)

