# Project Plan: Spectral Probing of CLAP Audio Embeddings

**Objective**: Extend the baseline linear probing analysis to investigate *where* chord information is encoded in the temporal frequency spectrum of CLAP (HTSAT) representations.

---

## 1. Experimental Design

### 1.1 Methodology
Instead of using the standard global mean-pooled embedding, we will:
1.  **Extract Full Sequence**: Access the raw HTSAT output feature map `(Batch, Channels, Freq, Time)` and flatten it to a sequence `(T=64, D=768)`.
2.  **Spectral Decomposition**: Apply **Discrete Cosine Transform (DCT)** along the temporal axis to map embeddings into the frequency domain.
3.  **Band Filtering**: Partition the spectrum into 5 distinct frequency bands using logarithmic scaling adapted from **Tamkin et al. (2020)**.
4.  **Energy Extraction**: Compute the **RMS Energy** of the filtered signal for each band (instead of mean pooling, which discards non-DC info).
5.  **Probing**: Train a linear classifier (Logistic Regression) on the energy vector of each band to measure its semantic content.

### 1.2 Frequency Bands (N=64)
| Band | Indices ($k$) | Description |
| :--- | :--- | :--- |
| **Low** | `[0]` | DC Component (Static) |
| **Mid-Low** | `[1]` | Slow Drift (~20s period) |
| **Mid** | `[2-4]` | Phrase Level (5-10s period) |
| **Mid-High**| `[5-16]` | Chord Level (1.25-4s period) |
| **High** | `[17-63]` | Beat Level / Transients (0.3-1.2s) |

---

## 2. Implementation Plan

### 2.1 Code Structure (`src/spectral_analysis.py`)
*   **Sequence Extraction**: Handle HTSAT `2x32` output and flatten to `64` tokens.
*   **DCT/IDCT**: Implement PyTorch-based DCT utilities.
*   **Band Masking**: Implement `gen_tamkin_filters(seq_len=64)`.
*   **Feature Computation**: Implement RMS Energy calculation per band.
*   **Probing Loop**: Iterate through all bands + Full baseline, train LR, and report metrics.

### 2.2 Validation Steps
*   **Sanity Check 1**: Ensure "Full" band (all frequencies) matches baseline performance.
*   **Sanity Check 2**: Verify Low vs. High embeddings are distinct (Cosine Similarity < 0.99).
*   **Sanity Check 3**: Check Feature Norms (Low should be high energy, High should be low energy).

---

## 3. Execution Timeline

1.  **Phase 1: Setup & Baseline**
    *   Data Prep (GuitarSet BN/SS Comp).
    *   Baseline Mean Pooling Probe.
2.  **Phase 2: Spectral Implementation**
    *   Implement DCT and Filtering.
    *   Fix dimension flattening issues.
    *   Switch from Mean Pooling to Energy features (Critical Fix).
3.  **Phase 3: Analysis & Reporting**
    *   Run experiments.
    *   Analyze results (Mid-Low dominance).
    *   Generate Report (`reports/spectral_baseline.md`).

---

## 4. Artifacts
*   `src/spectral_analysis.py`: Main analysis script.
*   `results/spectral_baseline.md`: Detailed results and discussion.
*   `reports/spectral_baseline.md`: Formal summary report.
