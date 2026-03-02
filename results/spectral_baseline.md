# Spectral Probing Baseline: CLAP + GuitarSet (Corrected Analysis)

**Objective**: Investigate *where* chord information is encoded in the temporal frequency spectrum of CLAP representations. We decompose the sequence of hidden states into frequency bands using **Tamkin et al. (2020)** inspired intervals and probe the **energy** of each band.

---

## 1. Methodology

### 1.1 Band Definition (Tamkin Adapted)
We adapt the logarithmic band definitions from Tamkin et al. (2020) to our sequence length $N=64$. This preserves the relative spectral bandwidth distribution (exponentially increasing).

| Band | Indices ($k$) | Bandwidth | Description |
| :--- | :--- | :--- | :--- |
| **Low** | `[0]` | 1 | DC Component (Static Average) |
| **Mid-Low** | `[1]` | 1 | Lowest non-zero frequency (Very slow change) |
| **Mid** | `[2, 4]` | 3 | Slow changes |
| **Mid-High**| `[5, 16]` | 12 | Intermediate modulation |
| **High** | `[17, 63]` | 47 | Rapid changes / Transients |

### 1.2 Energy-Based Probing
1.  **Extract Sequence**: Flatten CLAP (HTSAT) output to `(T=64, D=768)`.
2.  **DCT**: Apply DCT over the temporal axis `T`.
3.  **Filter**: Apply the 5 binary masks defined above.
4.  **Compute Energy**: Calculate RMS Energy for each dimension in the band.
5.  **Probe**: Train Linear Probe (Logistic Regression) on the energy vector.

---

## 2. Results

### 2.1 Metrics & Definitions
*   **RMS Energy (Feature)**: Root Mean Square of the frequency coefficients in a band.
    *   **Why RMS over Band Dimension?**: We compute RMS along the *frequency/time* axis ($K$) for *each* hidden channel ($D=768$). This collapses the temporal dimension while preserving the 768 semantic features. If we pooled over the channel dimension, we would lose all semantic content (chord info) and only measure global loudness.
    *   **Formula**: $E_d = \sqrt{\frac{1}{N_{band}} \sum_{k \in Band} |X_{k,d}|^2}$, where $d$ is the channel index.
*   **DCT-II**: We utilize the Type-II Discrete Cosine Transform, implemented via the fast FFT-based algorithm (Makhoul, 1980). This standard variant is ideal for spectral probing as it efficiently compacts energy for correlated signals without introducing phase discontinuities at boundaries.
*   **Avg Norm**: The average Euclidean length (L2 norm) of the feature vectors. High norm = High signal energy/strength. Low norm = Weak signal.
*   **Accuracy**: Percentage of correct chord predictions.
*   **Macro F1**: Harmonic mean of Precision and Recall, averaged equally across all classes (treating rare chords as equally important).

### 2.2 Training Setup Comparison
The probing classifier (Logistic Regression) is identical to the baseline, but the **Input Features** differ:

| Setup | Input Shape | Feature Definition | What it sees |
| :--- | :--- | :--- | :--- |
| **Baseline** | `(Batch, 768)` | `Mean(Time, Freq)` | Only the static average (DC) of each channel. |
| **Spectral Probe** | `(Batch, 768)` | `RMS(Band_k)` | The **energy/intensity** of temporal modulation in a specific frequency band $k$. |

*   **Output**: Same for both (22 classes: Root:Quality).
*   **Expected Outcome**: If a band has high Accuracy, it means chord information is encoded in the *dynamics* of that specific speed (e.g., slow drift vs. fast transient).

### 2.3 Performance Table

| Band | Accuracy | Macro F1 | Avg Norm | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Full (Mean Pool)**| **25.27%** | **0.1955** | - | Baseline (Best performance). |
| **Low (DC)** | 21.43% | 0.1704 | 1963.8 | Pure static information. |
| **Mid-Low** | **23.63%** | **0.1869** | 635.0 | **Best spectral band**. Captures slow evolution. |
| **Mid** | 20.88% | 0.1728 | 246.4 | Consistent info. |
| **Mid-High** | 19.78% | 0.1348 | 167.7 | Slightly lower separability. |
| **High** | 21.43% | 0.1403 | 129.6 | Surprisingly robust info despite low energy. |

---

## 3. Analysis & Interpretation

1.  **Mid-Low Dominance**: The **Mid-Low** band (Freq index 1) outperforms the purely static **Low** (DC) band (23.6% vs 21.4%). This suggests that *slow temporal evolution* (roughly 0.5-1 cycle per context window) is more semantically relevant for chord recognition than the static average.
2.  **Broad Distribution**: Chord information is **broadly distributed** across the entire spectrum. Even the **High** frequency band (dominated by rapid transients) yields ~21.4% accuracy, comparable to the Low band. This supports the "active channel" hypothesis: if a feature dimension detects a chord, it tends to be active with high energy across *all* timescales (e.g., a specific texture pattern that is both sustained and has high-frequency content).
3.  **Norm vs. Info**: The feature norm drops dramatically with frequency (1963 -> 129), but probing accuracy remains stable (~20-23%). This implies that CLAP's representation maintains a high **Signal-to-Noise Ratio (SNR)** for semantic concepts even in low-energy high-frequency components.
4.  **Comparison to NLP**: Unlike BERT (where info is often concentrated in specific bands), CLAP audio embeddings appear to "smear" semantic information across the temporal spectrum, likely due to the nature of audio texture processing in HTSAT.

---

## 4. Conclusion
> *"Spectral probing reveals that chord information in CLAP is not confined to the DC component. While global mean pooling (Full) performs best (25.3%), the **Mid-Low frequency band** (capturing slow temporal drift) is the most informative individual spectral component (23.6%). Remarkably, even high-frequency components retain significant linear separability (~21%), indicating that harmonic information is encoded redundantly across temporal scales."*

---
---
# Appendix: Experimental History & Error Log

Below is a record of previous attempts, errors, and flawed analyses, preserved for transparency and method development tracking.

## ❌ Attempt 1: Flawed Shape Handling & Mean Pooling

**Error**: Incorrectly assumed `hidden_state` shape was `(T, D)` and performed DCT on the wrong axis. Also used Mean Pooling after IDCT, which collapsed all non-DC information.

**Flawed Results**:
*   Low: 25.82% (Identical to Full)
*   Mid: 2.75% (Random)
*   High: 2.75% (Random)

**Why it failed**:
1.  **Shape Mismatch**: CLAP HTSAT output is `(B, 768, 2, 32)`. Early code treated it as `(B, 768)` or handled dimensions incorrectly.
2.  **Mean Pooling Artifact**: After filtering out the DC component (Index 0) in Mid/High bands, applying Mean Pooling results in a vector near zero. This explains why Mid/High bands performed randomly.

## ❌ Attempt 2: Corrected Shape, Flawed Pooling (Method A)

**Correction**: Correctly flattened HTSAT output to `(T=64, D=768)`.
**Persisting Error**: Still used Mean Pooling after IDCT.

**Flawed Results**:
*   Low: 25.82%
*   Mid: 2.75%
*   High: 2.75%

**Why it failed**: Even with correct shapes, the fundamental flaw of Mean Pooling on zero-mean signals (band-passed signals without DC) persisted. The "Low" band performed well only because it included the DC component.

## ❌ Attempt 3: Energy-Based Probing (Initial)

**Correction**: Switched to **Energy (RMS)** features instead of Mean Pooling.
**Results**:
*   Full: 25.82%
*   Low: 20.88%
*   Mid: 20.33%
*   High: 20.33%

**Insight**: This was the breakthrough. It revealed that Mid/High bands *do* contain information (~20%), debunking the "Low-only" hypothesis. However, the band definitions were arbitrary (linear/equal split), which is not standard in spectral probing literature.

## Final Correction (Current): Tamkin et al. Adaptation

**Correction**: Adopted logarithmic band scaling from Tamkin et al. (2020) to properly separate "DC", "Slow Drift", and "Fast Transient" components.
**Outcome**: See Section 2 (Results) above. This provided the most granular and interpretable insight, highlighting the specific importance of the **Mid-Low** band.
