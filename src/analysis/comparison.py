import numpy as np
from scipy.fftpack import dct
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

EMBEDDINGS_PATH = "data_artifacts/clap_embeddings_t64.npz"

# --- Utility Functions ---

def load_and_prepare_data(file_path=EMBEDDINGS_PATH):
    """
    Loads and prepares the data from the .npz file.
    """
    with np.load(file_path) as data:
        raw_embeddings = data['embeddings']
        genres = data['genres']
    X = raw_embeddings.transpose(0, 2, 1, 3).reshape(raw_embeddings.shape[0], -1, 32)
    y = np.array(genres)
    print(f"Data loaded. Reshaped X to: {X.shape}")
    return X, y

def run_probe(X_flat, y):
    """
    Runs a standard linear probe using Logistic Regression.
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    return accuracy_score(y_test, y_pred)

# --- Filter Bank Analysis Functions ---

def run_granular_band_analysis(X_freq, y):
    """
    Analyzes performance of linearly spaced frequency bands.
    """
    print("\n--- Running Filter Bank Comparison: 1. Granular (Linear) Bands ---")
    n_samples, n_features, n_coeffs = X_freq.shape
    bands = {f"Band {i}": (i*4, i*4+4) for i in range(n_coeffs // 4)}
    
    acc_full = run_probe(X_freq.reshape(n_samples, -1), y)
    print(f"Baseline Accuracy (All Bands): {acc_full:.2f}")

    for band_name, (start, end) in bands.items():
        X_freq_band = np.zeros_like(X_freq)
        X_freq_band[:, :, start:end] = X_freq[:, :, start:end]
        accuracy = run_probe(X_freq_band.reshape(n_samples, -1), y)
        print(f"  - Accuracy for {band_name}: {accuracy:.2f}")
    print("---------------------------------------------------------------------")

def run_mel_scale_analysis(X_freq, y):
    """
    Analyzes performance after applying Mel-scale filter banks.
    """
    print("\n--- Running Filter Bank Comparison: 2. Mel-Scale Filter Bank ---")
    n_samples, n_features, n_coeffs = X_freq.shape
    n_mels = 12
    sr_conceptual = 22050
    mel_filters = librosa.filters.mel(sr=sr_conceptual, n_fft=(n_coeffs-1)*2, n_mels=n_mels)
    
    X_mel = np.einsum('ijk,lk->ijl', X_freq, mel_filters)
    accuracy = run_probe(X_mel.reshape(n_samples, -1), y)
    print(f"Accuracy on {n_mels} Mel-Scale features: {accuracy:.2f}")
    print("---------------------------------------------------------------------")

def run_log_scale_analysis(X_freq, y):
    """
    Analyzes performance after applying custom Log-scale filter banks.
    """
    print("\n--- Running Filter Bank Comparison: 3. Log-Scale Filter Bank ---")
    n_samples, n_features, n_coeffs = X_freq.shape
    n_log_bands = 12

    # Create log-spaced frequencies. Add a small epsilon to avoid log(0).
    log_freqs = np.logspace(0, np.log10(n_coeffs), n_log_bands + 2)
    # Convert log-spaced points to linear indices
    linear_indices = np.floor(log_freqs).astype(int)
    linear_indices = np.minimum(linear_indices, n_coeffs - 1)
    linear_indices = np.unique(linear_indices) # Remove duplicates

    # Create triangular filters
    log_filters = np.zeros((len(linear_indices) - 2, n_coeffs))
    for i in range(len(linear_indices) - 2):
        start, center, end = linear_indices[i:i+3]
        log_filters[i, start:center] = np.linspace(0, 1, center - start, endpoint=False)
        log_filters[i, center:end] = np.linspace(1, 0, end - center, endpoint=False)

    print(f"Created Log filter bank of shape: {log_filters.shape}")

    X_log = np.einsum('ijk,lk->ijl', X_freq, log_filters)
    accuracy = run_probe(X_log.reshape(n_samples, -1), y)
    print(f"Accuracy on {log_filters.shape[0]} Log-Scale features: {accuracy:.2f}")
    print("---------------------------------------------------------------------")

if __name__ == "__main__":
    X_time, y = load_and_prepare_data()
    
    print("\nConverting to frequency domain via DCT...")
    X_freq = dct(X_time, axis=2, type=2, norm='ortho')
    
    # Run all filter bank comparisons
    run_granular_band_analysis(X_freq, y)
    run_mel_scale_analysis(X_freq, y)
    run_log_scale_analysis(X_freq, y)
