
# Project Insight & Future Work: Probing for Fundamental Musical Attributes

**Objective**: To move beyond a single, high-level task (genre classification) and conduct a systematic, multi-task probing analysis to create a detailed "spectral profile" of the CLAP model's understanding of fundamental musical concepts.

This plan outlines a series of controlled experiments designed to answer the question: **Which frequency bands does the model rely on to perform specific, granular musical listening tasks?**

---

## 1. The Probing Tasks (The "What")

Based on our discussion, we have defined a suite of 10 distinct probing tasks. Each task is designed to test the model's ability to perceive a specific musical attribute, and is paired with a hypothesis about which frequency range is most relevant.

1.  **Pitch Accuracy**: Discriminating subtle pitch variations (e.g., in-tune vs. slightly off-tune). (Hypothesis: **Mid-Frequency**)
2.  **Chord Recognition**: Identifying changes in chord progressions. (Hypothesis: **Mid-Frequency**)
3.  **Rhythm Stability**: Detecting tempo delays or swing variations. (Hypothesis: **Low-Frequency**)
4.  **Timbre Variation**: Judging changes in instrumentation. (Hypothesis: **Low to Mid-Frequency**)
5.  **Fine Articulation**: Detecting delicate musical articulations like vibrato or ornaments. (Hypothesis: **High-Frequency**)
6.  **Rhythm Synchronization**: Distinguishing between synchronous and asynchronous rhythmic sections. (Hypothesis: **Low-Frequency**)
7.  **Dynamics Detection**: Judging overall volume changes. (Hypothesis: **Low-Frequency**)
8.  **Texture & Detail**: Assessing changes in timbral texture. (Hypothesis: **Mid-Frequency**)
9.  **Harmonics Recognition**: Identifying high-frequency overtone details. (Hypothesis: **High-Frequency**)
10. **Structural Complexity**: Judging changes in musical structure or texture complexity. (Hypothesis: **Cross-Band**)

---

## 2. The Unified Methodology (The "How")

As you clarified, the analysis is **not based on specific songs, but on the distributional performance of a trained probe**. For *each* of the 10 tasks listed above, we will follow this standardized procedure:

1.  **Synthetic Dataset Generation**: For each task, we must first programmatically create a labeled dataset. For example, for "Pitch Accuracy," we would take clean audio clips and use tools like `librosa` to generate two versions: the original (label: "in-tune") and a slightly pitch-shifted version (label: "off-tune").

2.  **Feature Extraction**: We will use our established pipeline to process every clip in the synthetic dataset. This involves loading the audio, extracting the full `(64, 768)` hidden state sequence from the CLAP model, and applying the Discrete Cosine Transform (DCT) to get the final spectral representation.

3.  **Linear Probing**: For each task, we will train a simple linear probe (e.g., Logistic Regression) on the corresponding synthetic dataset. The probe's goal is to learn to classify the audio based on the task's labels (e.g., predict "in-tune" vs. "off-tune").

4.  **Performance & Weight Analysis**: This is the core of the analysis. After training each probe, we will extract and analyze two key outputs:
    *   **Task Performance**: The accuracy of the probe on its test set. This tells us *if* the model's representation contains enough information to solve the task at all.
    *   **Spectral Profile (Learned Weights)**: We will extract the learned coefficients (`model.coef_`) from the trained probe. By processing these weights (as we did in our genre analysis), we can generate a plot showing which of the 64 frequency coefficients were most important for the probe to make its decision. This profile reveals the spectral fingerprint of each task.

---

## 3. Expected Outcome

The final deliverable of this advanced analysis will be a comparative visualization, similar to Figure 3 in the reference paper. It will be a multi-panel plot where each row corresponds to one of the 10 probing tasks and displays its unique spectral profile.

This will allow us to directly compare, for example, the spectral profile of "Rhythm Stability" against that of "Harmonics Recognition," providing a deep and nuanced understanding of how and where the CLAP model encodes a wide range of fundamental musical knowledge.
