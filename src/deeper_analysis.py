import numpy as np
from scipy.fftpack import dct
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import librosa
import itertools

EMBEDDINGS_PATH = "data_artifacts/clap_embeddings.npz"

# --- Helper Functions (from previous script) ---

def load_and_prepare_data(file_path):
    """
    Loads embeddings and reshapes to the (B, 1536, 32) format.
    """
    with np.load(file_path) as data:
        raw_embeddings = data['embeddings']
        genres = data['genres']
    # Reshape [B, 768, 2, 32] -> [B, 1536, 32]
    X = raw_embeddings.transpose(0, 2, 1, 3).reshape(raw_embeddings.shape[0], -1, 32)
    y = np.array(genres)
    print(f"Data loaded. Reshaped X to: {X.shape}")
    return X, y

def run_probe(X_flat, y, use_scaler=True):
    """
    Runs a standard linear probe.
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    if use_scaler:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

# --- Deeper Analysis Functions ---

def run_granular_band_analysis(X_freq, y):
    """
    3.1. Granular Band Analysis (Low vs. High Frequency)
    """
    print("\n--- Running Deeper Analysis 3.1: Granular Band Analysis ---")
    n_samples = X_freq.shape[0]
    n_coeffs = X_freq.shape[2]
    
    # Define bands. We use 8 bands of 4 coefficients each.
    bands = {f"Band {i} (Coeffs {i*4}-{i*4+3})": (i*4, i*4+4) for i in range(n_coeffs // 4)}
    
    band_accuracies = {}

    for band_name, (start, end) in bands.items():
        # Create a copy and zero out coefficients outside the current band
        X_freq_band = np.zeros_like(X_freq)
        X_freq_band[:, :, start:end] = X_freq[:, :, start:end]
        
        # Flatten and run probe
        X_freq_band_flat = X_freq_band.reshape(n_samples, -1)
        accuracy = run_probe(X_freq_band_flat, y)
        
        print(f"Accuracy for {band_name}: {accuracy:.4f}")
        band_accuracies[band_name] = accuracy
        
    print("-------------------------------------------------------------")
    return band_accuracies

def run_mel_scale_analysis(X_freq, y):
    """
    3.2. Mel-Scale Filter Banks
    """
    print("\n--- Running Deeper Analysis 3.2: Mel-Scale Filter Banks ---")
    n_samples = X_freq.shape[0]
    n_coeffs = X_freq.shape[2] # 32
    n_mels = 12 # A standard number for Mel features

    # Create a Mel filter bank. The number of columns in the filter bank should match
    # the number of frequency bins we have (32).
    # librosa.filters.mel has `n_fft // 2 + 1` columns. We need `n_fft // 2 + 1 = 32`,
    # which means `n_fft = 62`.
    sr_conceptual = 22050 # This is arbitrary as we are already in frequency domain
    mel_filters = librosa.filters.mel(sr=sr_conceptual, n_fft=62, n_mels=n_mels)
    mel_filters = mel_filters[:, :n_coeffs] # Trim to shape (n_mels, 32)
    
    print(f"Created Mel filter bank of shape: {mel_filters.shape}")

    # Apply the filter bank. X_freq is (B, 1536, 32)
    # We transform the 32 coefficients for each of the 1536 features.
    X_mel = np.einsum('ijk,lk->ijl', X_freq, mel_filters)
    
    print(f"Shape after applying Mel filters: {X_mel.shape}")

    # Flatten and run probe
    X_mel_flat = X_mel.reshape(n_samples, -1)
    accuracy = run_probe(X_mel_flat, y)
    
    print(f"Accuracy on Mel-Scale features: {accuracy:.4f}")
    print("-------------------------------------------------------------")
    return accuracy

def run_band_connection_analysis(X_freq, y, band_accuracies):
    """
    3.3. Band Connection Analysis
    """
    print("\n--- Running Deeper Analysis 3.3: Band Connection Analysis ---")
    n_samples = X_freq.shape[0]
    n_coeffs = X_freq.shape[2]
    bands = {i: (i*4, i*4+4) for i in range(n_coeffs // 4)}
    band_names = {i: f"Band {i}" for i in range(n_coeffs // 4)}

    # Analyze pairs of bands
    for i, j in itertools.combinations(bands.keys(), 2):
        start_i, end_i = bands[i]
        start_j, end_j = bands[j]
        
        # Create a mask for the two bands
        X_freq_pair = np.zeros_like(X_freq)
        X_freq_pair[:, :, start_i:end_i] = X_freq[:, :, start_i:end_i]
        X_freq_pair[:, :, start_j:end_j] = X_freq[:, :, start_j:end_j]
        
        # Flatten and run probe
        acc_pair = run_probe(X_freq_pair.reshape(n_samples, -1), y)
        
        # Compare with individual accuracies
        acc_i = band_accuracies[f"{band_names[i]} (Coeffs {start_i}-{end_i-1})"]
        acc_j = band_accuracies[f"{band_names[j]} (Coeffs {start_j}-{end_j-1})"]
        
        print(f"Accuracy for {band_names[i]} + {band_names[j]}: {acc_pair:.4f} (Individual Accuracies: {acc_i:.4f}, {acc_j:.4f})")

    print("-------------------------------------------------------------")


if __name__ == "__main__":
    # This script is ready for review. 
    # The main execution block is commented out to prevent running.
    # Once approved, these lines can be uncommented to run the analysis.
    
    # print("Loading data...")
    # X, y = load_and_prepare_data(EMBEDDINGS_PATH)
    
    # print("\nConverting to frequency domain via DCT...")
    # X_freq = dct(X, axis=2, type=2, norm='ortho')
    
    # # 1. Granular Band Analysis
    # band_accuracies = run_granular_band_analysis(X_freq, y)
    
    # # 2. Mel-Scale Analysis
    # run_mel_scale_analysis(X_freq, y)
    
    # # 3. Band Connection Analysis
    # run_band_connection_analysis(X_freq, y, band_accuracies)
    pass
