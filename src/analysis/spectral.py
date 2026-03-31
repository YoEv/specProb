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
EMBEDDINGS_PATH = "data_artifacts/clap_embeddings_t64.npz"
RESULTS_DIR = "results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "spectral_analysis_results.json")
OUTPUT_DIR = "data_artifacts"

# --- Spectral Functions ---
def _apply_window(x: np.ndarray, axis: int, window_type: str) -> np.ndarray:
    """Apply a 1D window along the given axis. window_type: 'hann', 'hamming', 'blackman'."""
    ax = axis if axis >= 0 else x.ndim + axis
    n = x.shape[ax]
    if window_type == 'hann':
        w = np.hanning(n)
    elif window_type == 'hamming':
        w = np.hamming(n)
    elif window_type == 'blackman':
        w = np.blackman(n)
    else:
        raise ValueError(f"Unknown window_type: {window_type}")
    # Broadcast w to (1,...,1,n,1,...,1) so it multiplies along axis
    axes_expand = tuple(i for i in range(x.ndim) if i != ax)
    w = np.expand_dims(w, axes_expand)
    return x * w


def apply_transform(embeddings, transform_type='dct', axis=-1, window_type=None):
    """Applies a specified transform to the temporal dimension of the embeddings.

    Args:
        embeddings: (..., T) array; transform is applied along axis.
        transform_type: 'dct', 'fft', or 'dft'.
        axis: axis along which the sequence lies (default -1, i.e. T=64).
        window_type: For FFT only. None = rectangular (no window); 'hann', 'hamming', 'blackman'.
            FFT is applied along axis; length T=64 yields n_coeffs=33 (rfft).

    Supported values:
        - 'dct': Discrete Cosine Transform (type-II)
        - 'fft' or 'dft': Magnitude of the real FFT (DFT), optionally windowed.
    """
    if transform_type == 'dct':
        return dct(embeddings, axis=axis, type=2, norm='ortho')
    elif transform_type in ('fft', 'dft'):
        x = embeddings
        if window_type is not None:
            x = _apply_window(x, axis, window_type)
        return np.abs(np.fft.rfft(x, axis=axis, norm='ortho'))
    else:
        raise ValueError(f"Unknown transform_type: {transform_type}")

def get_raw_band_features(coeffs, band):
    """Extracts and flattens the raw coefficients for a specific band (no energy or other aggregation)."""
    start, end = band
    band_coeffs = coeffs[:, start:end, :]
    n_samples = band_coeffs.shape[0]
    return band_coeffs.reshape(n_samples, -1)


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
def run_spectral_analysis_pipeline(config):
    """Runs the full spectral analysis pipeline based on a config dict."""
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # --- Load Data ---
    print(f"Loading embeddings from {config['embeddings_path']}...")
    data = np.load(config['embeddings_path'], allow_pickle=True)
    embeddings = data['embeddings']
    genres = data['genres']
    
    print(f"Original embeddings shape: {embeddings.shape}")
    if embeddings.ndim == 4:
        embeddings = embeddings.transpose(0, 2, 3, 1)
        n_samples, t1, t2, n_features = embeddings.shape
        n_time_steps = t1 * t2
        embeddings = embeddings.reshape(n_samples, n_time_steps, n_features)
        print(f"Reshaped embeddings to: {embeddings.shape}")

    if embeddings.ndim == 2:
        embeddings = embeddings[:, np.newaxis, :]

    # --- Run Analysis ---
    print(f"\n--- Running Analysis with transform: {config['transform_type']} ---")
    coeffs = apply_transform(embeddings, transform_type=config['transform_type'])
    n_samples, n_freqs, n_features = coeffs.shape
    X_flat = coeffs.reshape(n_samples, -1)
    
    accuracy, class_names, _ = run_probe(X_flat, genres)
    print(f"Accuracy with full coefficients: {accuracy:.2f}")

    # --- Save Results ---
    results = {
        'config': config,
        'full_accuracy': accuracy,
        'classes': class_names.tolist(),
    }
    
    results_file = os.path.join(config['results_dir'], f"{config['experiment_name']}_results.json")
    print(f"\nSaving analysis results to {results_file}...")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print("Results saved successfully.")
    return results
