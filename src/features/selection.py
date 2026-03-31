
import numpy as np
import json
import os
from tqdm import tqdm

# Import the necessary functions from our consolidated script
from spectral_analysis import run_probe, get_raw_band_features

# --- Constants ---
DCT_CACHE_FILE = "data_artifacts/dct_coefficients.npz"
RESULTS_FILE = "results/band_selection_results.json"
N_INITIAL_BANDS = 8

# --- Main Execution ---
if __name__ == "__main__":
    # --- 1. Load Data ---
    print(f"Loading data from DCT cache file ({DCT_CACHE_FILE})...")
    with np.load(DCT_CACHE_FILE) as data:
        dct_coeffs = data['dct_coeffs']
        genres = data['genres']
    
    n_samples, n_freqs, n_features = dct_coeffs.shape
    print(f"Loaded DCT Coeffs with shape: {dct_coeffs.shape}")

    # --- 2. Initial Band Partitioning ---
    band_size = n_freqs // N_INITIAL_BANDS
    initial_bands = [(i * band_size, (i + 1) * band_size) for i in range(N_INITIAL_BANDS)]
    print(f"\nPartitioned spectrum into {N_INITIAL_BANDS} initial bands of size {band_size}.")
    print(initial_bands)

    # --- 3. Independent Band Evaluation ---
    print(f"\n--- Evaluating {N_INITIAL_BANDS} individual bands ---")
    band_performance = []
    for i, band in enumerate(tqdm(initial_bands, desc="Evaluating Bands")):
        features = get_raw_band_features(dct_coeffs, band)
        accuracy, _ = run_probe(features, genres)
        band_performance.append({'band_index': i, 'band_range': band, 'accuracy': accuracy})
        print(f"Band {i} {band}: Accuracy = {accuracy:.2f}")

    # --- 4. Sort Bands by Performance ---
    sorted_bands = sorted(band_performance, key=lambda x: x['accuracy'], reverse=True)
    print("\n--- Bands sorted by individual performance ---")
    for item in sorted_bands:
        print(f"Band {item['band_index']} {item['band_range']}: Accuracy = {item['accuracy']:.2f}")

    # --- 5. Greedy Cumulative Probing ---
    print("\n--- Performing greedy cumulative probing based on performance ranking ---")
    cumulative_accuracies = []
    cumulative_features = None
    
    # Get the indices of bands in their sorted order
    sorted_band_indices = [item['band_index'] for item in sorted_bands]

    for i, band_idx in enumerate(tqdm(sorted_band_indices, desc="Cumulative Probing")):
        band_range = initial_bands[band_idx]
        current_features = get_raw_band_features(dct_coeffs, band_range)
        
        if cumulative_features is None:
            cumulative_features = current_features
        else:
            # Concatenate features along the feature axis (axis=1)
            cumulative_features = np.concatenate((cumulative_features, current_features), axis=1)
        
        accuracy, _ = run_probe(cumulative_features, genres)
        cumulative_accuracies.append({
            'num_bands': i + 1,
            'bands_in_combination': [sorted_bands[j]['band_index'] for j in range(i + 1)],
            'cumulative_accuracy': accuracy
        })
        print(f"Accumulating Band {band_idx} (Top {i+1}). Total Features: {cumulative_features.shape[1]}. Accuracy: {accuracy:.2f}")

    # --- 6. Save Results ---
    final_results = {
        'initial_band_evaluation': band_performance,
        'sorted_band_performance': sorted_bands,
        'cumulative_probing_curve': cumulative_accuracies
    }
    
    print(f"\nSaving band selection analysis results to {RESULTS_FILE}...")
    with open(RESULTS_FILE, 'w') as f:
        json.dump(final_results, f, indent=4)
    print("Results saved successfully.")
