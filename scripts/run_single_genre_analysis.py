'''
This script is designed for single-genre spectral analysis of audio embeddings.
It performs a binary classification task for a specified genre against all others,
using spectral features derived from the embeddings. The script supports both DCT and FFT
as spectral transforms.

Key functionalities include:
- Loading audio embeddings and genre labels.
- Applying a spectral transform (DCT or FFT) to the embeddings.
- Training a logistic regression probe on the full spectrum and on specific frequency bands.
- Calculating and inspecting the weights of the trained model.
- Saving detailed metrics and analysis plots for each genre.
- Automatically iterating through all available genres in the dataset.
'''
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import warnings

from src.data_processing.loader import load_data
from src.analysis.spectral import apply_transform
from src.training.probes import run_probe
from src.visualization.spectral_profile import learned_weight_profile

warnings.filterwarnings('ignore')

# --- Constants ---
TRANSFORM_TYPE = 'fft'  # 'dct' or 'fft'
EMBEDDINGS_PATH = 'data_artifacts/clap_embeddings_t64.npz'
OUTPUT_DIR = 'results/fft_genre_specific_analysis'
if TRANSFORM_TYPE == 'fft':
    N_COEFFS = 33  # rfft(T=64) -> 33
else:
    N_COEFFS = 64
RANDOM_STATE = 42

# --- Main Execution ---
if __name__ == '__main__':
    # This script is for single genre analysis.
    # The TARGET_GENRE needs to be set manually.
    TARGET_GENRE = 'Pop' # <-- CHANGE THIS TO THE DESIRED GENRE

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print('Loading data...')
    X, y_str = load_data(EMBEDDINGS_PATH)

    print(f"\n{'='*20} Processing Genre: {TARGET_GENRE} {'='*20}")

    y_binary = (y_str == TARGET_GENRE).astype(int)

    if np.sum(y_binary) < 10:
        print(f"Skipping genre '{TARGET_GENRE}' due to insufficient samples ({np.sum(y_binary)} found).")
        exit()
    if len(np.unique(y_binary)) < 2:
        print(f"Skipping genre '{TARGET_GENRE}': need both classes for stratification.")
        exit()

    X_transformed = apply_transform(X, transform_type=TRANSFORM_TYPE, axis=2)
    assert X_transformed.shape[2] == N_COEFFS, f"Shape mismatch after transform: expected {N_COEFFS} coeffs, got {X_transformed.shape[2]}"

    print("Calculating metrics...")
    X_full_flat = X_transformed.reshape(X_transformed.shape[0], -1)
    orig_accuracy, final_model, _ = run_probe(X_full_flat, y_binary, random_state=RANDOM_STATE)

    if final_model is None:
        print(f"Could not train a model for {TARGET_GENRE}. Skipping.")
        exit()

    num_bands = N_COEFFS // 4
    bands = {b: list(range(b*4, (b+1)*4)) for b in range(num_bands)}
    band_accuracies = []
    pbar = tqdm(range(num_bands), desc=f"Band accuracies for {TARGET_GENRE}", leave=False)
    for b in pbar:
        band_coeffs = X_transformed[:, :, bands[b]].reshape(X_transformed.shape[0], -1)
        acc, _, _ = run_probe(band_coeffs, y_binary, random_state=RANDOM_STATE)
        band_accuracies.append(acc)

    cumulative_accuracies = []
    cumulative_coeffs = X_transformed[:, :, bands[0]].reshape(X_transformed.shape[0], -1)
    acc, _, _ = run_probe(cumulative_coeffs, y_binary, random_state=RANDOM_STATE)
    cumulative_accuracies.append(acc)

    pbar_cum = tqdm(range(1, num_bands), desc=f"Cumulative accuracies for {TARGET_GENRE}", leave=False)
    for b in pbar_cum:
        next_band_coeffs = X_transformed[:, :, bands[b]].reshape(X_transformed.shape[0], -1)
        cumulative_coeffs = np.concatenate((cumulative_coeffs, next_band_coeffs), axis=-1)
        acc, _, _ = run_probe(cumulative_coeffs, y_binary, random_state=RANDOM_STATE)
        cumulative_accuracies.append(acc)

    auto_accuracy = max(cumulative_accuracies) if cumulative_accuracies else 0

    raw_weights = final_model.coef_.flatten()
    n_features = X_transformed.shape[1]
    normalized_weights = learned_weight_profile(final_model, N_COEFFS, n_features=n_features)

    print(f'\nSaving metrics and generating plot for {TARGET_GENRE}...')
    metrics_data = {
        'target_genre': TARGET_GENRE,
        'transform_type': TRANSFORM_TYPE,
        'accuracies': {
            'full_embedding': orig_accuracy,
            'band_specific': band_accuracies,
            'cumulative_auto': auto_accuracy,
            'cumulative_per_band': cumulative_accuracies
        },
        'model_weights': {
            'mean': np.mean(raw_weights),
            'std_dev': np.std(raw_weights),
            'max': np.max(raw_weights),
            'min': np.min(raw_weights)
        },
        'spectral_profile': normalized_weights.tolist(),
        'config': {
            'n_coeffs': N_COEFFS,
            'random_state': RANDOM_STATE
        }
    }
    clean_genre_name = TARGET_GENRE.replace(' ', '_').replace('/', '_')
    metrics_filename = f'spectral_summary_{clean_genre_name}_{TRANSFORM_TYPE}_metrics.json'
    metrics_save_path = os.path.join(OUTPUT_DIR, metrics_filename)
    with open(metrics_save_path, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    print(f'Metrics for {TARGET_GENRE} saved to {metrics_save_path}')

    bar_labels = ['ORIG'] + [f'B{b}' for b in range(num_bands)] + ['AUTO']
    bar_heights = [orig_accuracy] + band_accuracies + [auto_accuracy]
    colors = ['gray'] + ['lightskyblue'] * num_bands + ['mediumpurple']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})
    fig.suptitle(f'Spectral Probing Analysis for Genre: {TARGET_GENRE} ({TRANSFORM_TYPE.upper()})', fontsize=16)

    ax1.bar(bar_labels, bar_heights, color=colors)
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    chance = np.mean(y_binary)
    ax1.axhline(y=chance, color='r', linestyle='--', label=f'Chance ({chance:.2f})')
    ax1.set_title('Probe Performance')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)

    x = np.arange(len(normalized_weights))
    ax2.plot(x, normalized_weights, color='black', linewidth=1)
    ax2.set_ylabel('Learned Weight')
    ax2.set_xlabel('Frequency Coefficient')
    ax2.set_xticks([0, N_COEFFS//2, N_COEFFS-1])
    ax2.set_xticklabels(['L', 'M', 'H'])
    ax2.set_ylim(0, 1)
    ax2.set_title('Spectral Profile')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_filename = f'spectral_summary_{clean_genre_name}_{TRANSFORM_TYPE}.png'
    save_path = os.path.join(OUTPUT_DIR, plot_filename)

    plt.savefig(save_path)
    plt.close(fig)
    print(f'Plot for {TARGET_GENRE} saved to {save_path}')
