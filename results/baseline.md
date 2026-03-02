# Baseline Results: Linear Probing of CLAP Embeddings

**Objective**: Assess the chord classification capability of frozen CLAP embeddings using a simple linear probe (Logistic Regression).
*   **What is Linear Probing?**: We freeze the CLAP model and only train a simple linear layer on top. The performance of this classifier acts as a **diagnostic score**: it tells us how easily accessible (linearly separable) the chord information is within the pre-trained representations. We are not trying to build the best possible chord classifier, but rather to *probe* what the model already knows.

---

## 1. Experimental Setup

*   **Model**: Frozen `laion/clap-htsat-unfused` (Audio Encoder).
*   **Feature**: Global Mean-Pooled Embedding (768-dim).
*   **Task**: 22-way Chord Classification (Root:Quality).
*   **Dataset Split**: Track-level splitting (80/20).
    *   **Why Track-level?**: To prevent "data leakage". If we split randomly by segment, adjacent segments from the same song (which sound very similar) could end up in both train and test sets, artificially inflating the score. By splitting by Track ID, we ensure the model is tested on *unseen songs*.
    *   **Train**: 57 tracks (682 samples) - Used to train the Logistic Regression classifier.
    *   **Test**: 15 tracks (182 samples) - Used to evaluate performance (the results below).

---

## 2. Metrics Definition

*   **Key Terms**:
    *   **TP (True Positive)**: Correctly predicted positive. (e.g., Model says C:maj, and it IS C:maj)
    *   **FP (False Positive)**: Incorrectly predicted positive. (e.g., Model says C:maj, but it is actually G:maj)
    *   **FN (False Negative)**: Missed positive. (e.g., Model says G:maj, but it is actually C:maj)
    *   **TN (True Negative)**: Correctly rejected negative. (e.g., Model says NOT C:maj, and it is NOT C:maj)

*   **Precision**: The accuracy of positive predictions. "Of all the chords predicted as C:maj, how many were actually C:maj?"
    *   Formula: $TP / (TP + FP)$
*   **Recall**: The ability to find all positive instances. "Of all the actual C:maj chords in the dataset, how many did we correctly find?"
    *   Formula: $TP / (TP + FN)$
*   **F1-Score**: The harmonic mean of Precision and Recall. Useful when you need a balance between the two.
    *   Formula: $2 * (Precision * Recall) / (Precision + Recall)$
*   **Support**: The number of actual occurrences of the class in the specified dataset (Test set in this case).

---

## 3. Overall Performance

| Metric | Score | Random Guess (Chance) |
| :--- | :--- | :--- |
| **Accuracy** | **25.82%** | ~4.5% |
| **Macro F1** | **0.1970** | ~4.5% |

---

## 3. Class-wise Breakdown

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **A:maj** | 0.38 | **0.60** | 0.46 | 10 |
| **C:maj** | **0.62** | 0.31 | 0.42 | 16 |
| **D#:maj** | 0.32 | 0.50 | 0.39 | 18 |
| **F#:maj** | 0.27 | **0.75** | 0.40 | 8 |
| **G:min** | **1.00** | 0.25 | 0.40 | 4 |
| **F:min** | **1.00** | 0.25 | 0.40 | 4 |
| *Others* | < 0.30 | < 0.30 | < 0.30 | - |

**Observations**:
*   **Major Chords**: Generally better recognized than Minor chords.
*   **High Precision**: Some minor classes (G:min, F:min) have perfect precision but very low recall, indicating the model is conservative about predicting them.
*   **Zero-Shot Gap**: Many classes (A#:maj, G#:maj) have 0.00 scores, highlighting the difficulty of the task for a frozen model without fine-tuning.

---

## 4. Raw Output Log
```text
Data loaded: X shape (864, 768), df shape (864, 6)
Classes: ['A#:maj' 'A#:min' 'A:maj' 'A:min' 'B:maj' 'B:min' 'C#:maj' 'C#:min'
 'C:maj' 'D#:maj' 'D:maj' 'D:min' 'E:maj' 'E:min' 'F#:maj' 'F:maj' 'F:min'
 'G#:maj' 'G#:min' 'G:maj' 'G:min' 'N']

Classification Report:
              precision    recall  f1-score   support

      A#:maj       0.00      0.00      0.00        10
      A#:min       0.00      0.00      0.00         4
       A:maj       0.38      0.60      0.46        10
       A:min       0.00      0.00      0.00         2
       B:maj       0.11      0.40      0.17         5
       B:min       0.22      0.25      0.24         8
      C#:maj       0.25      0.10      0.14        10
      C#:min       0.00      0.00      0.00         2
       C:maj       0.62      0.31      0.42        16
      D#:maj       0.32      0.50      0.39        18
       D:maj       0.33      0.17      0.22        12
       D:min       0.00      0.00      0.00         4
       E:maj       0.17      0.43      0.24         7
       E:min       0.40      0.20      0.27        10
      F#:maj       0.27      0.75      0.40         8
       F:maj       0.22      0.17      0.19        12
       F:min       1.00      0.25      0.40         4
      G#:maj       0.00      0.00      0.00        10
      G#:min       0.00      0.00      0.00         2
       G:maj       0.27      0.25      0.26        16
       G:min       1.00      0.25      0.40         4
           N       0.17      0.12      0.14         8

    accuracy                           0.26       182
   macro avg       0.26      0.22      0.20       182
weighted avg       0.29      0.26      0.24       182
```
