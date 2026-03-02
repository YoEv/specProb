import os
import numpy as np
import pandas as pd
from scipy.fftpack import dct
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import json

# --- Constants ---
EMBEDDINGS_PATH = "data_artifacts/clap_embeddings.npz"
RESULTS_DIR = "results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "spectral_analysis_results.json")
OUTPUT_DIR = "data_artifacts"

# --- Spectral Functions ---
def apply_dct(embeddings):
    """Applies DCT to the temporal dimension of the embeddings."""
    # Assuming embeddings are (n_samples, n_time_steps, n_features)
    # We apply DCT along the time_steps axis (axis=1)
    return dct(embeddings, axis=1, type=2, norm='ortho')

def get_raw_band_features(dct_coeffs, band):
    """Extracts and flattens the raw DCT coefficients for a specific band."""
    start, end = band
    band_coeffs = dct_coeffs[:, start:end, :]
    n_samples = band_coeffs.shape[0]
    # Flatten the band's time and feature dimensions into a single feature vector
    return band_coeffs.reshape(n_samples, -1)

# def get_band_energy(dct_coeffs, band):
#     """Computes the energy for a specific DCT coefficient band."""
#     start, end = band
#     band_coeffs = dct_coeffs[:, start:end, :]
#     # Energy = sum of squares of coefficients
#     energy = np.sum(band_coeffs**2, axis=(1, 2))
#     return energy

def get_spectral_bands(dct_coeffs, n_bands=4):
    """Divides DCT coefficients into bands and computes energy for each."""
    n_samples, n_freqs, n_features = dct_coeffs.shape
    band_size = n_freqs // n_bands
    
    band_energies_list = []
    for i in range(n_bands):
        start = i * band_size
        end = (i + 1) * band_size if i < n_bands - 1 else n_freqs
        energy = get_band_energy(dct_coeffs, (start, end))
        band_energies_list.append(energy)
        
    # Stack the energies into a single (n_samples, n_bands) array
    return np.stack(band_energies_list, axis=1)

# --- Probing Functions ---
def run_probe(X, y, test_size=0.2, random_state=42):
    """Trains and evaluates a logistic regression probe."""

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy, le.classes_, model

# --- Main Execution ---
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # --- Load Data ---
    print(f"Loading embeddings from {EMBEDDINGS_PATH}...")
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    embeddings = data['embeddings']
    genres = data['genres']
    
    print(f"Original embeddings shape: {embeddings.shape}")
    # The user correctly pointed out the dimensions were swapped.
    # We permute from (samples, d1, t1, t2) -> (samples, t1, t2, d1)
    # The original shape is (7997, 768, 2, 32)
    # We want (7997, 64, 768) where 64 is time and 768 is features.
    if embeddings.ndim == 4:
        embeddings = embeddings.transpose(0, 2, 3, 1) # -> (7997, 2, 32, 768)
        print(f"Transposed embeddings to: {embeddings.shape}")
        
        # Reshape to (samples, time, features)
        n_samples, t1, t2, n_features = embeddings.shape
        n_time_steps = t1 * t2 # -> 64
        embeddings = embeddings.reshape(n_samples, n_time_steps, n_features) # -> (7997, 64, 768)
        print(f"Reshaped embeddings to: {embeddings.shape}")

    # Reshape embeddings if they are 2D (n_samples, features) -> (n_samples, 1, features)
    if embeddings.ndim == 2:
        embeddings = embeddings[:, np.newaxis, :]    


    # --- Experiment 1: Raw DCT Probing ---
    print("\n--- Running Experiment 1: Raw DCT Coefficient Probing ---")
    dct_coeffs = apply_dct(embeddings)
    # Flatten the DCT coefficients for the classifier
    n_samples, n_freqs, n_features = dct_coeffs.shape
    X_dct_flat = dct_coeffs.reshape(n_samples, -1)
    
    print(f"Feature shape for raw DCT probe: {X_dct_flat.shape}")
    accuracy_dct, class_names = run_probe(X_dct_flat, genres)
    print(f"Accuracy with raw DCT coefficients: {accuracy_dct:.4f}")
    
    # --- Experiment 2: Spectral Band Probing ---
    print("\n--- Running Experiment 2: Spectral Band Probing ---")
    N_BANDS = 4 # e.g., Low, Mid-Low, Mid-High, High
    band_energies = get_spectral_bands(dct_coeffs, n_bands=N_BANDS)
    
    print(f"Feature shape for spectral band probe: {band_energies.shape}")
    band_accuracies = []
    for i in range(N_BANDS):
        X_band = band_energies[:, i:i+1] # Keep it as a 2D array
        accuracy_band, _ = run_probe(X_band, genres)
        band_accuracies.append(accuracy_band)
        print(f"  - Accuracy for Band {i+1}/{N_BANDS}: {accuracy_band:.4f}")

    # --- Experiment 3: Cumulative Spectral Band Probing ---
    print("\n--- Running Experiment 3: Cumulative Spectral Band Probing ---")
    cumulative_accuracies = []
    for i in range(1, N_BANDS + 1):
        X_cumulative = band_energies[:, :i]
        accuracy_cumulative, _ = run_probe(X_cumulative, genres)
        cumulative_accuracies.append(accuracy_cumulative)
        print(f"  - Accuracy for Bands 1-{i}: {accuracy_cumulative:.4f}")

    # --- Save DCT Coefficients Cache ---
    print(f"\nSaving DCT coefficients cache to data_artifacts/dct_coefficients.npz...")
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "dct_coefficients.npz"),
        dct_coeffs=dct_coeffs,
        genres=genres
    )

    # --- Save Experiment Results ---
    results = {
        'experiment_1_raw_dct': {
            'accuracy': accuracy_dct,
            'feature_shape': list(X_dct_flat.shape),
            'classes': class_names.tolist()
        },
        'experiment_2_spectral_bands': {
            'n_bands': N_BANDS,
            'band_accuracies': band_accuracies,
            'feature_shape': list(band_energies.shape)
        },
        'experiment_3_cumulative_bands': {
            'n_bands': N_BANDS,
            'cumulative_accuracies': cumulative_accuracies,
        }
    }
    
    print(f"\nSaving analysis results to {RESULTS_FILE}...")
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    print("Results saved successfully.")
