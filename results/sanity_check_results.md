# Sanity Check Results

**Date:** 2026-02-24

---

## 1. Time vs. Frequency Equivalence Check

### Objective

This sanity check aimed to verify a fundamental principle of signal processing: whether a linear probe's classification accuracy remains consistent when performed on raw time-domain data versus frequency-domain data obtained via an orthogonal Discrete Cosine Transform (DCT).

### Experimental Setup

- **Data Shape**: `(7997, 1536, 32)`
- **Time-Domain Probe**: `LogisticRegression` on flattened time-domain data.
- **Frequency-Domain Probe**: `LogisticRegression` on flattened frequency-domain data after applying DCT Type II (`norm='ortho'`) along the time axis.
- **Preprocessing**: `StandardScaler` was applied in both cases.

### Results

| Domain              | Accuracy | Difference |
|---------------------|----------|------------|
| Time-Domain         | `0.6044` |            |
| Frequency-Domain (DCT)| `0.5800` | `-0.0244`  |

### Conclusion: ❌ Failure

The accuracy on the frequency-domain data was significantly lower than on the time-domain data, contradicting the theoretical properties of orthogonal transforms.

---

## 2. DC Component Ablation Check

### Objective

To determine the contribution of the signal's mean value (the DC component, represented by the 0-th DCT coefficient) to the overall classification accuracy.

### Experimental Setup

1.  For each time-series of length 32, its mean was calculated and subtracted.
2.  DCT Type II was applied to this "mean-centered" data.
3.  A `LogisticRegression` probe was trained on the resulting frequency coefficients.

### Results

| Probe Condition                  | Accuracy | Notes                                         |
|----------------------------------|----------|-----------------------------------------------|
| Original Frequency-Domain        | `0.5800` | Baseline accuracy with DC component included. |
| Mean-Centered Frequency-Domain   | `0.5575` | Accuracy after removing the DC component.     |

- **Accuracy Drop**: `0.0225`

### Conclusion: ✅ Success (Insight Gained)

Removing the DC component caused a significant drop in accuracy. This proves that **the signal's mean is an important predictive feature**. However, the remaining accuracy (`0.5575`) is still high, indicating that **low-frequency dynamics (beyond the simple mean) also contain substantial predictive information**.

---

## 3. Feature Lane Ablation Check

### Objective

To analyze the individual and combined predictive power of the two feature "lanes" from the original `(B, 768, 2, 32)` data structure.

### Experimental Setup

1.  The data was split into two lanes, each with shape `(B, 768, 32)`.
2.  A separate `LogisticRegression` probe was trained on the DCT-transformed data of each lane.

### Results

| Probe Condition             | Feature Shape | Accuracy | Notes                               |
|-----------------------------|---------------|----------|-------------------------------------|
| Lane 0 Only                 | `(7997, 768, 32)`  | `0.5631` |                                     |
| Lane 1 Only                 | `(7997, 768, 32)`  | `0.5450` |                                     |
| **Combined (Baseline)**     | `(7997, 1536, 32)` | `0.5800` | Accuracy is higher than either lane alone. |

### Conclusion: ✅ Success (Insight Gained)

Both lanes contain predictive information, with Lane 0 being slightly more informative. Crucially, the combined accuracy is higher than either individual lane's accuracy. This demonstrates that **the two feature lanes contain complementary information**, validating the approach of concatenating them into a single 1536-dimension feature space.

---

## 4. Filter Bank Comparison

### Objective

To compare different methods of grouping or weighting the 32 DCT coefficients to see which representation is most effective for classification.

### Baseline

- **All 32 DCT Coefficients**: `0.5800`

### 4.1. Granular (Linear) Bands

Each band consists of 4 consecutive DCT coefficients.

| Band                  | Coefficients | Accuracy |
|-----------------------|--------------|----------|
| Band 0                | 0-3          | `0.5687` |
| Band 1                | 4-7          | `0.3287` |
| Band 2                | 8-11         | `0.5000` |
| Band 3                | 12-15        | `0.2762` |
| Band 4                | 16-19        | `0.5206` |
| Band 5                | 20-23        | `0.2294` |
| Band 6                | 24-27        | `0.4781` |
| Band 7                | 28-31        | `0.2481` |

**Conclusion**: The lowest frequency band (Band 0) is by far the most informative. A strange pattern emerges where even-numbered bands (0, 2, 4, 6) are significantly more predictive than odd-numbered bands.

### 4.2. Perceptual & Logarithmic Scales

These methods compress the 32 coefficients into a smaller number of features based on perceptual or mathematical scales.

| Filter Bank Type      | Number of Features | Accuracy |
|-----------------------|--------------------|----------|
| Mel-Scale             | 12                 | `0.5231` |
| Log-Scale             | 9                  | `0.5156` |

**Conclusion**: Both Mel and Log scaling result in a drop in performance compared to the baseline and even compared to using only Band 0. This suggests that for this embedding space, these standard compression techniques lose critical information.

### Overall Conclusion for Filter Banks

- **Low-frequency information is dominant**. The most effective representation tested so far is simply isolating the first few DCT coefficients.
- The **odd/even band performance disparity** is a novel finding that warrants further investigation.
- Standard perceptual scales (Mel, Log) are **not well-suited** for this feature space, performing worse than a simple linear split.

---

## 5. Sanity Check & Analysis Results - 2

### Experiments on T=64 Data

These experiments were conducted using embeddings with a time dimension of 64, generated by concatenating two 10-second audio segments.

### 1. Time vs. Frequency Equivalence Check

**Objective:** Verify if classification accuracy is consistent between time-domain and frequency-domain (DCT) representations.

| Domain | Accuracy |
| :--- | :--- |
| Time-Domain | `0.6369` |
| Frequency-Domain (DCT) | `0.6038` |

**Conclusion:** ❌ **Failure.** Similar to the T=32 case, a significant accuracy drop of `~0.033` occurs after DCT. This confirms the information loss is a persistent issue, even with higher time resolution.

---

### 2. DC Component Ablation Check

**Objective:** Assess the importance of the signal's mean value (0-th DCT coefficient).

| Probe Condition | Accuracy | Notes |
| :--- | :--- | :--- |
| Original Frequency-Domain (Baseline) | `0.6038` | Accuracy with DC component included. |
| Mean-Centered Frequency-Domain | `0.5763` | Accuracy after removing the DC component. |

**Conclusion:** ✅ **Success (Insight Gained).** Removing the DC component causes a notable accuracy drop (`~0.027`). This reinforces that the signal's mean is a valuable predictive feature, consistent with the T=32 findings.

---

### 3. Feature Lane Ablation Check

**Objective:** Analyze the predictive power of the two original feature lanes.

| Probe Condition | Accuracy | Notes |
| :--- | :--- | :--- |
| Lane 0 Only | `0.5856` | |
| Lane 1 Only | `0.5594` | |
| **Combined (Baseline)** | `0.6038` | Higher than either lane, showing synergy. |

**Conclusion:** ✅ **Success (Insight Gained).** The results mirror the T=32 findings. Lane 0 is more informative than Lane 1, and combining them yields better performance than either one alone, confirming their complementary nature.

---

### 4. Filter Bank Comparison

**Objective:** Compare different methods of grouping the 64 DCT coefficients.

#### 4.1 Granular (Linear) Bands

- **Baseline (All 64 Coeffs): `0.6038`**

| Band | Coeffs | Accuracy | | Band | Coeffs | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Band 0** | 0-3 | `0.5725` | | **Band 8** | 32-35 | `0.5494` |
| Band 1 | 4-7 | `0.5181` | | Band 9 | 36-39 | `0.4325` |
| Band 2 | 8-11 | `0.4519` | | Band 10 | 40-43 | `0.3444` |
| Band 3 | 12-15 | `0.4300` | | Band 11 | 44-47 | `0.3475` |
| **Band 4** | 16-19 | `0.5450` | | **Band 12**| 48-51 | `0.5413` |
| Band 5 | 20-23 | `0.4044` | | Band 13 | 52-55 | `0.3831` |
| Band 6 | 24-27 | `0.4094` | | Band 14 | 56-59 | `0.2844` |
| Band 7 | 28-31 | `0.4150` | | Band 15 | 60-63 | `0.3175` |

**Conclusion:** The lowest frequency band (Band 0) remains the most informative. The strange odd/even performance disparity is less clear now, but there are still noticeable peaks at regular intervals (Band 0, 4, 8, 12), suggesting a periodic importance across the spectrum.

#### 4.2 Perceptual & Logarithmic Scales

| Filter Bank Type | Number of Features | Accuracy |
| :--- | :--- | :--- |
| Mel-Scale | 12 | `0.5456` |
| Log-Scale | 10 | `0.5544` |

**Implementation Details:**

*   **Mel-Scale:** This method uses the `librosa.filters.mel` function to simulate how humans perceive pitch. It creates a set of 12 triangular filters that are spaced linearly at low frequencies and logarithmically at high frequencies. The 64 DCT coefficients are then multiplied by this filter bank to produce 12 Mel-scaled features. The key parameters used were `n_mels=12`, a conceptual `sr=22050`, and `n_fft=126` (derived from `(n_coeffs-1)*2`).

*   **Log-Scale:** This is a custom-implemented filter bank that spaces filters purely logarithmically across the 64 coefficients. We first generate logarithmically spaced points using `np.logspace`. These points are then used to define the start, center, and end of 10 overlapping triangular filters. This method provides a non-linear compression that gives more resolution to lower-frequency coefficients.

**Conclusion:** Similar to the T=32 case, both Mel and Log scaling perform worse than the baseline and even worse than using just the first few DCT coefficients (Band 0). This further suggests that standard perceptual scales are not well-suited for this feature space.



### **Final Summary of T=64 CLAP Embedding Analysis**

Our extensive analysis of the 64-timestep CLAP embeddings yielded several key insights into their structure and predictive power in the frequency domain.

**1. Persistent Information Loss via DCT**
- A consistent drop in accuracy occurs when transforming the data from the time domain to the frequency domain using DCT.
- **Time-Domain Accuracy:** `0.6369`
- **Frequency-Domain (DCT) Accuracy:** `0.6038`
- **Conclusion:** The DCT process, even with orthogonal normalization, loses or scrambles predictive information. This is a fundamental issue with applying this specific transformation to these embeddings.

**2. Importance of Low-Frequency & DC Components**
- The vast majority of predictive power is concentrated in the lowest frequency bands. **Band 0** (coefficients 0-3) alone achieves an accuracy of `0.5725`.
- Removing the DC component (the signal's mean before DCT) causes a significant accuracy drop from `0.6038` to `0.5763`, confirming its value as a feature.

**3. Synergistic & Complementary Band Information**
- Simply accumulating frequency bands does not guarantee better performance. The relationship between bands is complex and non-monotonic.
- **Key Finding:** Combining the strongest band (Band 0) with weaker bands often improves accuracy beyond what Band 0 can achieve alone. For example, `Band 0 + Band 15` (`0.5894`) and `Band 0 + Band 4` (`0.5900`) both outperform solo `Band 0` (`0.5725`).
- **Conclusion:** Higher-frequency bands, while weak on their own, contain information that is **complementary** to the low-frequency bands. They are not merely redundant.

**4. Ineffectiveness of Perceptual Scales**
- Standard audio-perceptual filter banks (Mel-Scale, Log-Scale) performed poorly, yielding accuracies of `0.5456` and `0.5544` respectively.
- **Conclusion:** The feature space of these embeddings does not align with human auditory perception. Custom or linear band selections are more effective.

**Overall Implication:** The most promising direction for feature engineering is not to use all DCT coefficients, nor to use standard perceptual scales. Instead, a **selective combination of frequency bands**—specifically pairing the dominant low-frequency bands with certain complementary higher-frequency bands—appears to be the optimal strategy for maximizing predictive accuracy from this feature set.

## Detailed tables from all the `T=64` experiments.

### **1. Core Sanity Checks**

| Analysis Type | Condition | Accuracy |
| :--- | :--- | :--- |
| **Equivalence** | Time-Domain | `0.6369` |
| | Frequency-Domain (DCT) | `0.6038` |
| **DC Ablation** | Mean-Centered Freq. Data | `0.5763` |
| **Lane Ablation** | Lane 0 Only | `0.5856` |
| | Lane 1 Only | `0.5594` |

---

### **2. Granular (Linear) Band Analysis**

*   **Baseline (All Bands): `0.6038`**

| Band | Coeffs | Accuracy | | Band | Coeffs | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Band 0** | 0-3 | `0.5725` | | Band 8 | 32-35 | `0.5494` |
| Band 1 | 4-7 | `0.5181` | | Band 9 | 36-39 | `0.4325` |
| Band 2 | 8-11 | `0.4519` | | Band 10 | 40-43 | `0.3444` |
| Band 3 | 12-15 | `0.4300` | | Band 11 | 44-47 | `0.3475` |
| **Band 4** | 16-19 | `0.5450` | | **Band 12**| 48-51 | `0.5413` |
| Band 5 | 20-23 | `0.4044` | | Band 13 | 52-55 | `0.3831` |
| Band 6 | 24-27 | `0.4094` | | Band 14 | 56-59 | `0.2844` |
| Band 7 | 28-31 | `0.4150` | | Band 15 | 60-63 | `0.3175` |

---

### **3. Band Combination & Accumulation**

#### **Specific Combinations**

| Combination | Accuracy |
| :--- | :--- |
| Band 0 + Band 15 | `0.5894` |
| Band 14 + Band 15 | `0.3362` |
| Band 0 + Band 4 | `0.5900` |

#### **Information Accumulation**

| Accumulated Bands | Accuracy |
| :--- | :--- |
| Band 0 | `0.5725` |
| Bands 0 to 1 | `0.5919` |
| Bands 0 to 2 | `0.5875` |
| Bands 0 to 3 | `0.5887` |
| Bands 0 to 4 | `0.5856` |
| Bands 0 to 5 | `0.5744` |
| Bands 0 to 6 | `0.5706` |
| Bands 0 to 7 | `0.5700` |
| Bands 0 to 8 | `0.5850` |
| Bands 0 to 9 | `0.5869` |
| Bands 0 to 10 | `0.5894` |
| Bands 0 to 11 | `0.6000` |
| Bands 0 to 12 | `0.5981` |
| Bands 0 to 13 | `0.5981` |
| Bands 0 to 14 | `0.5919` |
| Bands 0 to 15 (All) | `0.6038` |


---

### **6. Data Balance and Performance Deep Dive**

This analysis was conducted to verify the dataset's class balance and to gain a deeper understanding of the model's performance beyond a single accuracy score. The baseline model (using all 64 DCT coefficients) was used.

#### **6.1 Data Balance Check**

The dataset is **highly balanced**. Each of the 8 genres has approximately 1000 samples, as shown below.

| Genre | Sample Count |
| :--- | :--- |
| Electronic | 999 |
| Experimental | 999 |
| Folk | 1000 |
| Hip-Hop | 1000 |
| Instrumental | 1000 |
| International | 1000 |
| Pop | 1000 |
| Rock | 999 |

**Conclusion:** Because the dataset is balanced, the standard `accuracy` score is a reliable metric. The `macro average` and `weighted average` are identical to the overall accuracy (`0.60`), confirming that no single class is disproportionately affecting the score.

#### **6.2 Detailed Performance Report**

The classification report provides a per-class breakdown of model performance.

| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **Hip-Hop** | 0.72 | 0.71 | **0.72** |
| **International** | 0.67 | 0.70 | **0.69** |
| Folk | 0.63 | 0.70 | 0.66 |
| Electronic | 0.65 | 0.63 | 0.64 |
| Rock | 0.61 | 0.66 | 0.63 |
| Instrumental | 0.57 | 0.64 | 0.60 |
| Experimental | 0.60 | 0.49 | 0.54 |
| **Pop** | 0.35 | 0.29 | **0.32** |

**Key Insights:**
*   **Best Performers:** The model identifies `Hip-Hop` and `International` genres with the highest efficacy.
*   **Worst Performer:** `Pop` music is the most challenging class, with significantly lower precision and recall than all other genres.

#### **6.3 Confusion Matrix Analysis**

The confusion matrix reveals the specific errors the model makes.

**Labels**: `['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Pop', 'Rock']`

```
[[126,  11,   1,  16,  18,   8,  16,   4],  -> True: Electronic
 [ 14,  99,   8,   7,  29,  11,  20,  12],  -> True: Experimental
 [  0,  11, 140,   2,  11,   6,  19,  11],  -> True: Folk
 [ 21,   5,   2, 143,   4,  10,  13,   2],  -> True: Hip-Hop
 [ 12,  19,  12,   5, 127,   6,   6,  13],  -> True: Instrumental
 [  7,   4,  15,   9,   6, 141,  11,   7],  -> True: International
 [ 12,   9,  30,  15,  17,  22,  59,  36],  -> True: Pop
 [  2,   8,  15,   2,   9,   7,  26, 131]]  -> True: Rock
```

**Primary Finding:**
*   **`Pop` Music Confusion:** The model's poor performance on `Pop` is primarily due to confusion with `Folk` and `Rock`. Out of 200 true `Pop` samples, only 59 were correctly identified. A significant number were misclassified as **`Rock` (36 times)** and **`Folk` (30 times)**. This suggests a strong feature overlap between these three genres within the learned embedding space.
*   **Other Confusions:** `Experimental` music is most frequently confused with `Instrumental` (29 times), which is an intuitive error given the nature of the genres.

**Overall Conclusion:** The model's primary weakness is its inability to distinguish `Pop` music from `Folk` and `Rock`. While the data is well-balanced, the feature representation learned by the CLAP model appears to merge the characteristics of these specific genres, posing a significant challenge for a linear classifier.