"""
FFT-based spectral probing for all genres. Uses unified src loader, transform, and probes.
Output: results/fft_genre_specific_analysis/ (metrics JSON per genre; plotting can be added).
"""
import numpy as np
import os
from tqdm import tqdm
import json

from src.data_processing.loader import load_data
from src.analysis.spectral import apply_transform
from src.training.probes import run_probe
from src.visualization.spectral_profile import learned_weight_profile

# --- Constants (match 2nd_analysis_plan: (B, 1536, 64) -> n_coeffs=33) ---
EMBEDDINGS_PATH = "data_artifacts/clap_embeddings_t64.npz"
OUTPUT_DIR = "results/fft_genre_specific_analysis"
TRANSFORM_TYPE = "fft"
N_COEFFS = 33
RANDOM_STATE = 42


def run_fft_analysis_for_all_genres():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    X, y_str = load_data(EMBEDDINGS_PATH)
    assert X.ndim == 3 and X.shape[2] == 64, f"Expected (B, 1536, 64), got {X.shape}"
    all_genres = np.unique(y_str)
    print(f"Found genres to analyze: {', '.join(all_genres)}")

    for TARGET_GENRE in all_genres:
        print(f"\n{'='*20} Processing Genre: {TARGET_GENRE} {'='*20}")

        y_binary = (y_str == TARGET_GENRE).astype(int)

        if np.sum(y_binary) < 10:
            print(f"Skipping genre '{TARGET_GENRE}' due to insufficient samples ({np.sum(y_binary)} found).")
            continue
        if len(np.unique(y_binary)) < 2:
            print(f"Skipping genre '{TARGET_GENRE}': need both classes.")
            continue

        X_transformed = apply_transform(X, transform_type=TRANSFORM_TYPE, axis=2)
        assert X_transformed.shape[2] == N_COEFFS, f"Shape mismatch: expected {N_COEFFS} coeffs, got {X_transformed.shape[2]}"

        print("Calculating metrics...")
        X_full_flat = X_transformed.reshape(X_transformed.shape[0], -1)
        orig_accuracy, final_model, _ = run_probe(X_full_flat, y_binary, random_state=RANDOM_STATE)

        num_bands = N_COEFFS // 4
        bands = {b: list(range(b * 4, (b + 1) * 4)) for b in range(num_bands)}
        band_accuracies = []
        for b in range(num_bands):
            band_coeffs = X_transformed[:, :, bands[b]].reshape(X_transformed.shape[0], -1)
            acc, _, _ = run_probe(band_coeffs, y_binary, random_state=RANDOM_STATE)
            band_accuracies.append(acc)

        cumulative_accuracies = []
        cumulative_coeffs = X_transformed[:, :, bands[0]].reshape(X_transformed.shape[0], -1)
        acc, _, _ = run_probe(cumulative_coeffs, y_binary, random_state=RANDOM_STATE)
        cumulative_accuracies.append(acc)
        for b in range(1, num_bands):
            next_band_coeffs = X_transformed[:, :, bands[b]].reshape(X_transformed.shape[0], -1)
            cumulative_coeffs = np.concatenate((cumulative_coeffs, next_band_coeffs), axis=-1)
            acc, _, _ = run_probe(cumulative_coeffs, y_binary, random_state=RANDOM_STATE)
            cumulative_accuracies.append(acc)

        auto_accuracy = max(cumulative_accuracies) if cumulative_accuracies else 0.0

        raw_weights = final_model.coef_.flatten()
        n_features = X_transformed.shape[1]
        normalized_weights = learned_weight_profile(final_model, N_COEFFS, n_features=n_features)

        clean_genre_name = TARGET_GENRE.replace(" ", "_").replace("/", "_")
        metrics_data = {
            "target_genre": TARGET_GENRE,
            "transform_type": TRANSFORM_TYPE,
            "accuracies": {
                "full_embedding": orig_accuracy,
                "band_specific": band_accuracies,
                "cumulative_auto": auto_accuracy,
                "cumulative_per_band": cumulative_accuracies,
            },
            "model_weights": {
                "mean": float(np.mean(raw_weights)),
                "std_dev": float(np.std(raw_weights)),
                "max": float(np.max(raw_weights)),
                "min": float(np.min(raw_weights)),
            },
            "spectral_profile": normalized_weights.tolist(),
            "config": {"n_coeffs": N_COEFFS, "random_state": RANDOM_STATE},
        }
        metrics_save_path = os.path.join(OUTPUT_DIR, f"spectral_summary_{clean_genre_name}_{TRANSFORM_TYPE}_metrics.json")
        with open(metrics_save_path, "w") as f:
            json.dump(metrics_data, f, indent=4)
        print(f"Metrics for {TARGET_GENRE} saved to {metrics_save_path}")


if __name__ == "__main__":
    run_fft_analysis_for_all_genres()
