# Spectral Probing Pipeline Report

## 1. Data Preparation
**Source**: GuitarSet (Bossa Nova & Singer-Songwriter styles, Accompaniment mode only).
**Process**:
*   Scanned audio files matching `*_{BN,SS}*_*_comp_mix.wav`.
*   Parsed corresponding JAMS annotations to extract chord segments.
*   Simplified chord labels to `Root:Quality` (e.g., `C:maj`, `A:min`), mapping complex chords to `N` or nearest triad.
*   Generated a metadata registry `segments.csv` containing track IDs, timestamps, and labels.

**Output**: 864 labeled chord segments across 72 tracks.

## 2. Feature Extraction (Baseline)
**Model**: Frozen `laion/clap-htsat-unfused` (Audio Encoder).
**Process**:
*   Loaded audio segments at 48kHz.
*   Passed raw waveforms through the frozen CLAP audio encoder.
*   Extracted the global mean-pooled embedding (768-dim) representing the entire segment.
*   Cached embeddings as `embeddings.npy`.

## 3. Spectral Decomposition & Filtering
**Objective**: Isolate information at different temporal scales.

**Sequence Construction (Origin of N=64)**:
*   **Model Output (Fixed by CLAP)**: The `laion/clap-htsat-unfused` model uses an HTSAT backbone, which produces a 4D feature map: `(Batch, Channels=768, Freq=2, Time=32)`. This `2x32` grid is the native architectural output, typically corresponding to a fixed input duration (e.g., 10s) compressed by the model.
*   **Flattening (Our Operation)**: To apply 1D spectral analysis (DCT), we **flatten** the Frequency and Time dimensions into a single sequence: $N = \text{Freq} \times \text{Time} = 2 \times 32 = 64$.
*   **Implication (Coarse Resolution)**: Since CLAP typically processes ~10s of audio into 32 time steps, the effective temporal resolution is $\approx 312\text{ms/token}$. This is relatively coarse compared to frame-level audio features (e.g., 10ms), explaining why the model excels at capturing slower semantic evolutions (Mid-Low band) rather than rapid micro-transients.

**Process**:
*   **Sequence Recovery**: Extracted the full hidden state sequence `(T=64, D=768)` from the HTSAT encoder, flattening the frequency and time axes.
*   **DCT**: Applied Discrete Cosine Transform (DCT) along the temporal axis to convert embeddings to the frequency domain.
*   **Band Filtering**: Applied binary masks adapted from **Tamkin et al. (2020)** to isolate specific frequency bands. Given the input duration of ~10s and $N=64$, we map spectral indices $k$ to physical timescales (Period $\approx 20/k$ seconds):

    | Band | Indices ($k$) | Freq (Hz) | Period (Time Scale) | Description |
    | :--- | :--- | :--- | :--- | :--- |
    | **Low** | `[0]` | 0 | $\infty$ | **Static** (Global Average) |
    | **Mid-Low** | `[1]` | 0.05 | ~20s | **Slow Drift** (Trend over full clip) |
    | **Mid** | `[2-4]` | 0.1 - 0.2 | 5s - 10s | **Phrase Level** (Long evolution) |
    | **Mid-High**| `[5-16]` | 0.25 - 0.8 | 1.25s - 4s | **Chord/Bar Level** (Typical harmonic rate) |
    | **High** | `[17-63]` | 0.85 - 3.2 | 0.3s - 1.2s | **Beat Level** (Rapid changes/Transients) |

*   **Energy Feature**: Computed the RMS Energy of each feature dimension within each band, resulting in a 768-dim energy vector per band.

## 4. Linear Probing
**Task**: 22-way Chord Classification (Major/Minor + N).
**Method**:
*   Trained a Logistic Regression classifier on the extracted features (Baseline Mean-Pool vs. Spectral Energy Vectors).
*   **Split**: 80/20 Track-level split (ensuring no track leakage between train/test).
*   **Metric**: Accuracy and Macro F1-Score.

## 5. Key Findings
*   **Baseline Performance**: Global mean pooling yields the highest accuracy (~25.3%), indicating that static spectral statistics are the most robust predictor.
*   **Spectral Insight (Resolution Mismatch)**: Theoretically, the **Mid-High** band (1.25s - 4s) should align best with typical chord durations. However, our results show **Mid-Low** (~20s) outperforms it (23.6% vs 19.8%).
    *   **Reason 1 (Coarse Resolution)**: The ~312ms token resolution likely blurs the distinct transitions required for Mid-High features, whereas the ultra-slow Mid-Low features are unaffected by this bottleneck.
    *   **Reason 2 (Contextual Bias)**: CLAP likely relies on global tonal context (captured in Mid-Low) rather than precise local chord boundaries, prioritizing "mood/key" over "momentary harmony."
*   **Broad Encoding**: Significant chord information (~21%) is retained even in high-frequency bands, suggesting that semantic features are encoded redundantly across temporal scales.
