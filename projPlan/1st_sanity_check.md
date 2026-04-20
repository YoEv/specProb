
# Sanity Checks & Deeper Analysis: Implementation Plan

**Objective**: To rigorously validate our experimental setup and findings, ensuring that the observed "low-frequency dominance" is a genuine phenomenon and not an artifact of our methodology. This document outlines a concrete implementation plan for all sanity checks and future analyses.

---

## 1. Core Premise Correction: Tensor Semantics & Code Structure

Our most critical realization is the potential misinterpretation of the raw embedding shape `[B, 768, 2, 32]`. All future experiments will be based on the corrected semantics.

**Correct Interpretation**:
-   `B`: Batch size
-   `768`: Feature dimension
-   `32`: Time steps
-   `2`: Parallel feature lanes

**Correct Reshape**: `[B, 768, 2, 32]` -> `[B, 1536, 32]` (Concatenate in feature space)

### 1.1. Code Implementation

We will create a new, dedicated script `src/sanity_checks.py` to systematically implement all validation experiments.

```python
# src/sanity_checks.py

import numpy as np
from scipy.fftpack import dct
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# ... other imports

# --- Data Loading & Preprocessing ---
def load_and_reshape_data(path="data_artifacts/clap_embeddings.npz"):
    # ... (Loads and reshapes to [B, 1536, 32])
    return X, genres, labels # labels are string names

# --- Core Probing Function (Expanded) ---
def run_probe(features, labels):
    """Trains a probe and returns a dictionary of detailed metrics."""
    # ... (Train/test split, scaling, logistic regression)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        'accuracy': accuracy,
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'report': report,
        'confusion_matrix': cm
    }
    return metrics, model

# ... (Functions for each Sanity Check and Deeper Analysis) ...
```

---

## 2. Priority Sanity Checks: Implementation Details

*(This section details the three core validation experiments: Time vs. Freq Equivalence, DC Component Ablation, and Feature Lane Ablation, with their specific logic as previously discussed.)*
### 2.1. Information Equivalence Check (Time vs. Frequency)

-   **Equation/Logic**:
    1.  `X_time = flatten(X)` where `X` has shape `[B, 1536, 32]`
    2.  `X_freq = flatten(DCT(X, axis=2))` where DCT is applied along the time axis (axis=2).
    3.  `Acc_time = run_probe(X_time, genres)`
    4.  `Acc_freq = run_probe(X_freq, genres)`
-   **Expected Result**: `Acc_time` should be approximately equal to `Acc_freq`.
-   **Implementation**: This will be the `check_equivalence` function in `sanity_checks.py`.

### 2.2. DC Component Ablation (Mean Removal)

-   **Clarification**: This check addresses whether the signal's **mean value (DC component)**, captured by the 0-th DCT coefficient, is the primary source of our results, as opposed to low-frequency **dynamics**.
-   **Equation/Logic**:
    1.  For each feature `f` in `1536` and each sample `b` in `B`:
        `X_centered_b,f(t) = X_b,f(t) - mean(X_b,f(t) for all t)`
    2.  `X_freq_centered = DCT(X_centered, axis=2)`
    3.  Run the full band analysis (or at least probe Band 0) on `X_freq_centered`.
-   **Expected Result**: If the accuracy of Band 0 drops significantly, our previous findings were heavily reliant on the simple mean of the signal. If it remains high, the low-frequency dynamics are genuinely important.
-   **Implementation**: This will be the `check_dc_ablation` function.

### 2.3. Feature Lane Ablation

-   **Equation/Logic**:
    1.  `X_lane0 = embeddings[:, :, 0, :].reshape(B, 768, 32)`
    2.  `X_lane1 = embeddings[:, :, 1, :].reshape(B, 768, 32)`
    3.  `Acc_lane0 = run_probe(flatten(DCT(X_lane0, axis=2)), genres)`
    4.  `Acc_lane1 = run_probe(flatten(DCT(X_lane1, axis=2)), genres)`
-   **Expected Result**: Comparing `Acc_lane0`, `Acc_lane1`, and the accuracy from the concatenated `1536`-feature probe will reveal the relationship between the two lanes.
-   **Implementation**: This will be the `check_lane_ablation` function.

---

## 3. Visualization Update

### 3.1. Spectral Profile Plot Modification

-   **Objective**: Modify the spectral profile plot for greater clarity.
-   **Specification**: The color should only be applied **underneath** the learned weight curve, filling the area between the curve and the x-axis.
-   **Implementation**: Use `matplotlib`'s `fill_between` function.

    ```python
    # In plotting.py
    x = np.arange(len(normalized_weights))
    ax2.plot(x, normalized_weights, color='black', linewidth=0.8)
    # Use a loop with fill_between to create the gradient fill
    for i in range(len(x) - 1):
        ax2.fill_between([i, i+1], [normalized_weights[i], normalized_weights[i+1]], color=cmap(norm(i)), alpha=0.5)
    ```

---

## 4. Deeper Analysis & Future Work: Implementation Details

Once the sanity checks confirm a solid foundation, we will proceed with these deeper analyses, implemented as functions within `sanity_checks.py`.

### 4.1. Granular Performance Metrics

-   **Objective**: To get a nuanced view of performance beyond simple accuracy.
-   **Implementation**: The core `run_probe` function will be permanently modified to return a dictionary of metrics, including `accuracy`, `macro_f1`, a full `classification_report`, and the `confusion_matrix`. A helper function, `print_metrics(metrics)`, will be created to pretty-print this information for every experiment run.

### 4.2. Advanced Filter Banks

-   **Objective**: To test if the "low-frequency dominance" is robust to more perceptually relevant frequency scaling.
-   **Implementation**: A function `check_mel_filterbank(X_freq, genres)` will be created.

    ```python
    import librosa

    def check_mel_filterbank(X_freq, genres):
        n_samples, n_features, n_freqs = X_freq.shape # [B, 1536, 32]
        
        # 1. Create Mel filterbank matching our frequency resolution
        # sample_rate can be arbitrary as we operate on DCT coeffs, not Hz
        mel_filters = librosa.filters.mel(sr=22050, n_fft=n_freqs*2-2, n_mels=16)
        # librosa's filters are for power spectrum, so we square our coeffs
        power_spectrum = X_freq**2

        # 2. Apply filterbank to each of the 1536 feature channels
        # We need to loop or use einsum for this: [B, 1536, 32] x [16, 32] -> [B, 1536, 16]
        mel_energies = np.einsum('ijk,lk->ijl', power_spectrum, mel_filters)

        # 3. Probe the resulting 16 Mel-band energy features
        metrics, _ = run_probe(mel_energies.reshape(n_samples, -1), genres)
        print("--- Mel Filterbank Results ---")
        print(f"Accuracy with 16 Mel Bands: {metrics['accuracy']:.4f}")
    ```

### 4.3. Analyze Band Connections

-   **Objective**: To discover potential complementary relationships between frequency bands.
-   **Implementation**: A function `check_band_connections(X_freq, genres)` will be created.

    ```python
    def check_band_connections(X_freq, genres):
        n_samples, _, n_freqs = X_freq.shape
        
        # Define bands (e.g., 8 bands of 4 coeffs each)
        bands = [(i*4, (i+1)*4) for i in range(8)]
        B0_features = X_freq[:, :, bands[0][0]:bands[0][1]].reshape(n_samples, -1)
        B7_features = X_freq[:, :, bands[7][0]:bands[7][1]].reshape(n_samples, -1)

        # Combine B0 and B7 features
        combined_features = np.concatenate((B0_features, B7_features), axis=1)
        
        metrics, _ = run_probe(combined_features, genres)
        print("--- Band Connection (B0+B7) Results ---")
        print_metrics(metrics)
    ```

### 4.4. Explore Higher Temporal Resolution

-   **Objective**: To see if more time steps reveal the importance of mid/high frequencies.
-   **Implementation Plan**: This is a future task that requires modifying the data generation step, not just the analysis script.
    1.  **Investigate Model**: Analyze the `laion/clap-htsat-unfused` model architecture in the `transformers` library to see if the internal STFT or patching mechanism allows for configuration (e.g., changing `hop_size`).
    2.  **Re-run `extract_features.py`**: If configuration is possible, re-run the entire feature extraction with settings that produce a longer time axis (e.g., `T > 32`).
    3.  **Re-run Sanity Checks**: Repeat all sanity checks on this new high-resolution dataset.

### 4.5. FFT vs. DCT Analysis

-   **Objective**: To explore the role of phase information.
-   **Status**: This is a future research direction and will not be implemented at this stage.
