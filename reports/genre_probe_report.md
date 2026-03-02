
# Spectral Probing Analysis for Music Genre Classification

**Date**: 2026-02-17

**Objective**: To investigate how musical genre information is distributed across the temporal frequency spectrum of CLAP's hidden representations. This report summarizes the methodology, findings, and conclusions of the spectral probing analysis.

---

## 1. Dataset & Methodology

### 1.1. Dataset

This analysis was conducted on the **FMA Small** dataset, a subset of the full Free Music Archive (FMA). We specifically used the **8 balanced genre categories** provided in this subset: `Electronic`, `Experimental`, `Folk`, `Hip-Hop`, `Instrumental`, `International`, `Pop`, and `Rock`.

### 1.2. Methodology

Our analysis followed a systematic, multi-step approach to probe the spectral characteristics of CLAP embeddings for the task of music genre classification.

1.  **Feature Extraction**: We began by extracting the full, un-pooled hidden state sequence from the `laion/clap-htsat-unfused` model for each audio track. This resulted in a high-dimensional sequence representation capturing temporal evolution, with a final shape of `(64, 768)` for each track, representing `(time_steps, features)`.

2.  **Spectral Decomposition**: We applied the Discrete Cosine Transform (DCT-II) along the temporal axis (64 time steps) of the sequences. This transformed the representation from the time domain to the frequency domain, allowing us to analyze the importance of different frequency components.

3.  **Systematic Band Analysis (Forward Feature Selection)**: To understand the contribution of different frequency regions, we implemented a forward selection strategy:
    *   **Partitioning**: The 64 frequency coefficients were divided into 8 distinct, non-overlapping bands.
    *   **Independent Evaluation**: Each band was evaluated independently by training a linear probe (Logistic Regression) on its raw DCT coefficients to determine its individual predictive power for genre classification.
    *   **Greedy Accumulation**: Based on their individual performance, the bands were sorted from most to least informative. We then cumulatively added bands one by one, training a new probe at each step to observe how the accuracy evolved as more spectral information was included.

4.  **Spectral Weight Analysis**: To visualize which frequencies the model deemed most important, we trained a final linear probe on the complete set of raw DCT coefficients and extracted the learned model weights (`coef_`). By averaging these weights, we constructed a “Spectral Profile” that highlights the importance of each of the 64 frequency coefficients.

---

## 2. Core Findings

The analysis culminated in a comprehensive summary plot that visualizes our key discoveries.

### 2.1. Main Analysis: Spectral Summary

![Spectral Summary Plot](../results/spectral_summary_plot.png)

This plot is divided into two panels:

-   **Probe Performance (Left Panel)**: This bar chart compares the accuracy of different feature sets:
    -   **ORIG (Grey)**: The baseline performance using all raw DCT coefficients, achieving **59.1%** accuracy.
    -   **B0-B7 (Blue)**: The performance of the 8 individual frequency bands. A clear pattern emerges: **Band 0 (B0)**, representing the lowest frequencies, achieves an accuracy of **56.6%** on its own, dramatically higher than any other band.
    -   **AUTO (Purple)**: The peak performance achieved during our greedy accumulation process, reaching **58.8%**.

-   **Spectral Profile (Right Panel)**: This line graph visualizes the learned importance (weight) of each of the 64 frequency coefficients.
    -   **Extreme Low-Frequency Dominance**: The profile shows a massive spike at the very beginning of the spectrum, corresponding to the 0-th DCT coefficient.
    -   **Rapid Decay**: After the initial spike, the weights rapidly decay, indicating that the model assigns very little importance to mid and high-frequency information.

### 2.2. Preliminary Analysis: Simple Band Accumulation

As part of our initial exploration, we also analyzed a simple, non-greedy accumulation of frequency bands. The results, shown below, reinforce our final findings by demonstrating that accumulating bands from low to high frequency steadily improves performance, though not as efficiently as the performance-sorted greedy approach.

![Accuracy Comparison Plot](../results/accuracy_comparison.png)

---

## 3. Conclusion

Our spectral probing analysis has led to a clear and decisive conclusion:

**For the task of music genre classification, the vast majority of discriminative information within the CLAP embeddings is encoded in the lowest frequency components of the temporal sequence.**

The 0-th DCT coefficient, which represents the DC component or the mean of the sequence over time, is by far the most informative feature. While other frequency bands contain some signal, their contribution is minor compared to this dominant low-frequency component.

This finding suggests that for broad categorical tasks like genre, the global, averaged characteristics of the audio representation are more critical than the fine-grained temporal dynamics found in higher frequencies. This systematic analysis, moving from a high-level hypothesis to a detailed, evidence-backed conclusion, successfully fulfills the project's objective.
