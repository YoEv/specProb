import numpy as np
import os
from scipy.fftpack import dct
import warnings
import argparse

# Adjust the path to import from the src directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_data
from src.analysis import run_probe
from src.visualization.plotting import create_and_save_plot
from src.visualization.spectral_profile import learned_weight_profile

warnings.filterwarnings('ignore')

# --- Constants ---
EMBEDDINGS_PATH = 'data_artifacts/clap_embeddings_t64.npz'
OUTPUT_DIR = 'results/genre_specific_plots'
N_COEFFS = 64
RANDOM_STATE = 42

def main(target_genre):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f'Loading data and preparing for genre: {target_genre}')
    X, y_str = load_data(EMBEDDINGS_PATH)
    X_dct = dct(X, type=2, axis=-1, norm='ortho')
    
    y_binary = (y_str == target_genre).astype(int)
    
    print('Calculating metrics...')
    # --- 1. Calculate accuracies ---
    X_full_flat = X_dct.reshape(X_dct.shape[0], -1)
    orig_accuracy, final_model = run_probe(X_full_flat, y_binary, RANDOM_STATE)

    num_bands = N_COEFFS // 4
    bands = {b: list(range(b*4, (b+1)*4)) for b in range(num_bands)}
    band_accuracies = []
    for b in range(num_bands):
        band_coeffs = X_dct[:, :, bands[b]].reshape(X_dct.shape[0], -1)
        acc, _ = run_probe(band_coeffs, y_binary, RANDOM_STATE)
        band_accuracies.append(acc)
    
    cumulative_accuracies = []
    cumulative_coeffs = X_dct[:, :, bands[0]]
    for b in range(1, num_bands):
        acc, _ = run_probe(cumulative_coeffs.reshape(cumulative_coeffs.shape[0], -1), y_binary, RANDOM_STATE)
        cumulative_accuracies.append(acc)
        cumulative_coeffs = np.concatenate((cumulative_coeffs, X_dct[:, :, bands[b]]), axis=-1)
    cumulative_accuracies.append(orig_accuracy)
    auto_accuracy = max(cumulative_accuracies)

    bar_labels = ['ORIG'] + [f'B{b}' for b in range(num_bands)] + ['AUTO']
    bar_heights = [orig_accuracy] + band_accuracies + [auto_accuracy]
    colors = ['gray'] + ['lightskyblue'] * num_bands + ['mediumpurple']

    # --- 2. Calculate weights ---
    normalized_weights = learned_weight_profile(final_model, N_COEFFS)

    # --- 3. Create and Save Plot ---
    print('Generating plot...')
    chance = np.mean(y_binary)
    create_and_save_plot(target_genre, bar_labels, bar_heights, colors, normalized_weights, chance, N_COEFFS, OUTPUT_DIR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run spectral probing analysis for a specific music genre.')
    parser.add_argument('genre', type=str, help='The target genre to analyze.')
    args = parser.parse_args()
    
    main(args.genre)
