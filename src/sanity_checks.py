import numpy as np
from scipy.fftpack import dct
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

EMBEDDINGS_PATH = "data_artifacts/clap_embeddings.npz"

def load_and_prepare_data(file_path):
    """
    Loads embeddings and returns both original and reshaped versions.
    """
    with np.load(file_path) as data:
        raw_embeddings = data['embeddings'] # Shape: (B, 768, 2, 32)
        genres = data['genres']
        
    # Reshaped version for combined analysis: [B, 1536, 32]
    X_reshaped = raw_embeddings.transpose(0, 2, 1, 3).reshape(raw_embeddings.shape[0], -1, 32)
    y = np.array(genres)
    
    print(f"Data loaded. Original shape: {raw_embeddings.shape}, Reshaped X to: {X_reshaped.shape}")
    return raw_embeddings, X_reshaped, y

def run_probe(X_flat, y):
    """
    Runs a standard linear probe.
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

def check_equivalence(X, y):
    """
    Performs the time vs. frequency equivalence check on (B, 1536, 32) data.
    """
    print("\n--- Running Sanity Check: Time vs. Frequency Equivalence (on 32-step time axis) ---")
    n_samples = X.shape[0]

    # 1. Time-domain probe
    X_time_flat = X.reshape(n_samples, -1)
    acc_time = run_probe(X_time_flat, y)
    print(f"Accuracy on raw time-domain data: {acc_time:.4f}")

    # 2. Frequency-domain probe
    # Apply DCT along the time axis (axis=2 for shape (B, 1536, 32))
    X_freq = dct(X, axis=2, type=2, norm='ortho')
    X_freq_flat = X_freq.reshape(n_samples, -1)
    acc_freq = run_probe(X_freq_flat, y)
    print(f"Accuracy on frequency-domain (DCT) data: {acc_freq:.4f}")

    # 3. Compare accuracies
    if np.isclose(acc_time, acc_freq, atol=1e-3):
        print("✅ Result: Success! Accuracies are equivalent as expected.")
    else:
        print(f"❌ Result: Failure! Accuracies are significantly different by {abs(acc_time - acc_freq):.4f}.")
        
    print("-------------------------------------------------------------------------------------")

def check_dc_ablation(X, y):
    """
    Sanity Check 2.2: DC Component Ablation.
    Checks the impact of removing the mean from the time-series signal before DCT.
    """
    print("\n--- Running Sanity Check 2.2: DC Component Ablation ---")
    
    # For each sample and each feature, subtract the mean of its time-series
    # X has shape (B, 1536, 32). We compute mean over axis 2.
    X_mean = np.mean(X, axis=2, keepdims=True)
    X_centered = X - X_mean
    
    print(f"Subtracted time-series mean. Shape of X_centered: {X_centered.shape}")
    
    # Apply DCT to the mean-centered data
    X_freq_centered = dct(X_centered, axis=2, type=2, norm='ortho')
    
    # The 0-th DCT coefficient of a mean-centered signal should be close to zero.
    # We can verify this as a quick check.
    mean_zeroth_coeff = np.mean(np.abs(X_freq_centered[:, :, 0]))
    print(f"Mean of absolute 0-th DCT coefficient after centering: {mean_zeroth_coeff:.6f}")

    # Run the probe on the flattened, centered frequency coefficients
    n_samples = X_freq_centered.shape[0]
    X_freq_centered_flat = X_freq_centered.reshape(n_samples, -1)
    acc_centered = run_probe(X_freq_centered_flat, y)
    
    print(f"Accuracy on mean-centered frequency-domain data: {acc_centered:.4f}")
    print("-------------------------------------------------------------------------------------")

def check_lane_ablation(X_raw, y):
    """
    Sanity Check 2.3: Feature Lane Ablation.
    Analyzes the performance of each feature lane independently.
    """
    print("\n--- Running Sanity Check 2.3: Feature Lane Ablation ---")
    
    # X_raw has shape (B, 768, 2, 32)
    # Lane 0: (B, 768, 32)
    X_lane0 = X_raw[:, :, 0, :]
    # Lane 1: (B, 768, 32)
    X_lane1 = X_raw[:, :, 1, :]
    
    print(f"Shape of Lane 0: {X_lane0.shape}")
    print(f"Shape of Lane 1: {X_lane1.shape}")

    # --- Probe Lane 0 ---
    X_freq_lane0 = dct(X_lane0, axis=2, type=2, norm='ortho')
    n_samples = X_freq_lane0.shape[0]
    X_freq_lane0_flat = X_freq_lane0.reshape(n_samples, -1)
    acc_lane0 = run_probe(X_freq_lane0_flat, y)
    print(f"Accuracy on frequency-domain data for Lane 0 ONLY: {acc_lane0:.4f}")

    # --- Probe Lane 1 ---
    X_freq_lane1 = dct(X_lane1, axis=2, type=2, norm='ortho')
    X_freq_lane1_flat = X_freq_lane1.reshape(n_samples, -1)
    acc_lane1 = run_probe(X_freq_lane1_flat, y)
    print(f"Accuracy on frequency-domain data for Lane 1 ONLY: {acc_lane1:.4f}")
    print("-------------------------------------------------------------------------------------")

if __name__ == "__main__":
    X_raw, X_reshaped, y = load_and_prepare_data(EMBEDDINGS_PATH)
    check_equivalence(X_reshaped, y)
    check_dc_ablation(X_reshaped, y)
    check_lane_ablation(X_raw, y)
