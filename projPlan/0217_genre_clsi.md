# Project Plan: Spectral Probing for Genre Classification

**Date**: 2026-02-17
**Objective**: To investigate how musical genre information is distributed across the temporal frequency spectrum of CLAP's hidden representations.

---

## 📂 File & Directory Structure

```
specProb/
├── data/fma_small/               # Raw audio files
├── data_artifacts/
│   ├── fma_metadata.csv            # Simplified metadata for the project
│   ├── baseline_embeddings.npy     # Mean-pooled embeddings for baseline
│   └── spectral_features/          # Directory for all spectral feature sets
├── plots/
│   ├── filter_responses.png
│   ├── signal_decomposition.png
│   ├── accuracy_comparison.png
│   └── genre_spectral_heatmap.png
├── projPlan/
│   └── 0217_genre_clsi.md        # This project plan
├── reports/
│   └── genre_probe_report.md     # Final summary report
└── src/
    ├── prep_fma_data.py            # Phase 1: Data preparation
    ├── extract_features.py         # Phase 1 & 2: Feature extraction
    ├── train_probe.py              # Phase 1 & 2: Training and evaluation
    ├── analysis.py                 # Phase 3: Deeper analysis
    ├── plotting.py                 # Phase 3: Visualization scripts
    └── experiments.py              # Phase 3: Advanced experiments
```

---

## Phase 1: Baseline & Data Setup

### 1.1. Goal
Establish a baseline genre classification accuracy using the standard mean-pooled CLAP embedding from 30-second audio clips.

### 1.2. Implementation

- **`src/prep_fma_data.py`**
  - `load_fma_metadata(path)`: Loads the official `tracks.csv` from FMA, which contains metadata for all splits of the dataset.
  - `create_project_metadata(df, audio_dir)`: Filters for the 'small' subset, verifies that each audio file exists in `data/fma_small/`, and creates our simplified `data_artifacts/fma_metadata.csv` with columns `[track_id, genre, audio_path]`.

- **`src/extract_features.py`** (Mode: `baseline`)
  - `load_audio_segment(path, duration=30)`: Loads the audio file, converts it to mono, and extracts the central 30-second segment. If the file is shorter than 30s, it will be padded with silence.
  - `get_baseline_embedding(audio, model, processor)`: Takes the 30s audio segment and computes the single 768-dimension global mean-pooled embedding.
  - `main(mode='baseline')`: The main function will loop through `fma_metadata.csv`, process each track to get its baseline embedding, collect them into a single NumPy array, and save it to `data_artifacts/baseline_embeddings.npy`.

- **`src/train_probe.py`** (Mode: `baseline`)
  - `load_data(features_path, metadata_path)`: Loads the `baseline_embeddings.npy` and the `fma_metadata.csv` to align features with genre labels.
  - `train_and_evaluate(X, y)`: Performs a stratified train/test split on the data to ensure genre balance. It will train a Logistic Regression classifier on the training set and print a detailed classification report (including precision, recall, F1-score) from the test set evaluation.
  - `main(mode='baseline')`: Orchestrates the entire baseline experiment by calling the above functions and logging the final performance.

---

## Phase 2: Spectral Probing Experiments

### 2.1. Goal
Decompose the full, un-pooled embeddings into spectral components and probe them individually and cumulatively to discover where genre information is most concentrated.

### 2.2. Implementation

- **`src/extract_features.py`** (Mode: `spectral`)
  - `get_clap_sequence(audio, model, processor)`: Extracts the full hidden state sequence from a 30s audio clip. For the `laion/clap-htsat-unfused` model, this is a 4D tensor of shape `(1, D, T1, T2)`.
  - `main(mode='spectral')`: The main script now processes this 4D tensor by first permuting its dimensions to `(1, T1, T2, D)` and then reshaping it into a 3D tensor of `(1, T, D)`, where `T = T1 * T2` (e.g., 64 time steps) and `D` is the feature dimension (e.g., 768). This becomes the input for the spectral analysis.

- **`src/spectral_analysis.py`**: **(Consolidated Probe Runner)**
  - This script is now the central place for all probing experiments, superseding the need for a separate `train_probe.py`.
  - `run_probe(X, y)`: The core linear probe function. Trains and evaluates a logistic regression classifier on given features `X` and labels `y`.
  - `apply_dct(sequence)`: Applies a DCT-II transformation along the corrected temporal axis (`T=64`) of the sequence.
  - `main()`: Orchestrates all experiments: 
    1. Probing raw DCT coefficients.
    2. Probing individual spectral bands.
    3. Probing cumulative spectral bands.
    - It saves all results into a single `spectral_analysis_results.json`.

---

## Phase 3: Analysis, Visualization & Future Work

### 3.1. Goal
Perform deeper analysis to pinpoint critical frequencies, explore advanced methods to enhance performance, and then visualize all findings.

### 3.2. Implementation: Deeper Analysis & Experiments

- **`src/analysis.py`**
  - `find_critical_sub_band(dct_coeffs, labels, band_to_split)`: Implements the binary search ("二分法"). It takes the DCT coefficients, programmatically splits the frequency spectrum in half, and runs a probing experiment on each to determine which half retains more information. This is called recursively for a set number of iterations (e.g., 10) to find the most informative narrow band. The final result is saved to `critical_band_results.json`.

- **`src/experiments.py`**
  - **High-Frequency Probing & Enhancement**:
    - `probe_high_freq_only(coeffs, labels)`: A straightforward experiment that trains a probe using only the energy from the highest frequency bands to establish a high-frequency baseline.
    - `train_adapter_with_spectral_loss(...)`: A more complex experiment. This will implement a small, trainable adapter module that sits on top of the frozen CLAP, using a custom loss to focus on high-frequency content.
  - **Alternative Filter Banks**:
    - `get_mel_filtered_energy(dct_coeffs)`: Applies a set of triangular Mel-spaced filter masks.
    - `get_wavelet_features(sequence)`: Replaces DCT with a DWT.
    - `train_learnable_filters(sequence, labels)`: Implements a simple 1D CNN to act as a learnable filter.

### 3.3. Implementation: Visualization (Final Step)

- **`src/plotting.py`**: A dedicated script to generate all key visualizations for the final report *after* the analysis is complete.
  - `plot_accuracy_results()`: Generates a bar chart comparing the probing accuracy of all feature sets (raw, individual bands, cumulative bands).
  - `plot_genre_spectral_heatmap()`: Creates a heatmap showing the average spectral energy per frequency coefficient for each genre.
  - `plot_signal_decomposition()`: For a sample track, creates a heatmap of its DCT coefficients to visualize the spectrum.
  - `plot_filter_responses()`: Creates a plot showing the frequency masks for spectral bands.

---

## 4. Q&A and Technical Notes

- **DCT Windowing**: The DCT is applied globally across the entire sequence for a given clip. There is **no sliding window or hop size**. The model's output sequence is already a highly compressed representation, making a global transform the most direct and appropriate first step.
- **Information & Aggregation**: The core trade-off we are exploring is between information preservation (using raw DCT coefficients) and robust feature engineering (using aggregated band energy). The high dimensionality of the raw spectral output can be challenging for simple linear models, which is why aggregation is a key experimental step.