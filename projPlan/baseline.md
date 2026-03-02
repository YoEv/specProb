# 📋 Executable Plan: CLAP + GuitarSet Chord Probing (BN/SS Comp)

**Objective**: Evaluate if a frozen CLAP audio encoder can linearly separate chord classes from specific GuitarSet styles (Bossa Nova, Swing) in accompaniment mode.

---

## 1. ⚙️ Configuration & Constraints

### 1.1 Data Selection
We strictly filter the dataset to the following subset:
*   **Styles**: `BN` (Bossa Nova), `SS` (Singer-Songwriter)
*   **Playing Style**: `comp` (Accompaniment) only
*   **Exclusions**: No `Rock`, `Funk`, `Jazz` (unless SS counts as Jazz, but we filter by "SS" string). No `solo` files.

### 1.2 File Paths
*   **Audio Source**: `/home/evev/specProb/guitarSet/audio_mono-pickup_mix`
    *   *Pattern*: `*_{BN,SS}*_*_comp_mix.wav`
*   **Annotation Source**: `/home/evev/specProb/guitarSet/annotation`
    *   *Pattern*: `*_{BN,SS}*_*_comp.jams`
*   **Output Directory**: `/home/evev/specProb/data_artifacts`

### 1.3 Model
*   **Model**: `laion/clap-htsat-unfused` (HuggingFace Transformers)
*   **Mode**: Audio Encoder only, Frozen (No Fine-tuning).

---

## 2. 🏗️ Pipeline & Code Modules

We will implement the following Python modules in `src/`.

### Module 1: Data Preparation
**File**: `src/prep_data.py`

**Functions**:
1.  `scan_files(audio_dir, annot_dir)`:
    *   Glob `*_{BN,SS}*_*_comp_mix.wav`.
    *   Find corresponding `.jams` file (remove `_mix` suffix from audio basename).
    *   Return pairs of `(audio_path, jams_path)`.
2.  `parse_jams_chords(jams_path)`:
    *   Use `jams` library.
    *   Extract chord observations: `time`, `duration`, `value`.
    *   **Mapping**: Simplify chords to `Root:Quality` (e.g., `C:maj`, `A:min`). Map complex chords to `N` or closest triad if necessary.
3.  `create_segments(pairs)`:
    *   Generate a metadata table.
    *   **Columns**: `track_id`, `audio_path`, `t_start`, `t_end`, `chord_label`.
    *   Save to `data_artifacts/segments.csv`.

### Module 2: Feature Extraction
**File**: `src/extract_features.py`

**Functions**:
1.  `load_model()`:
    *   `ClapModel.from_pretrained("laion/clap-htsat-unfused").audio_model`
    *   `ClapProcessor.from_pretrained("laion/clap-htsat-unfused")`
    *   Move to GPU (`cuda`).
2.  `extract_embeddings(segments_csv)`:
    *   Iterate rows in CSV.
    *   Load Audio: `librosa.load(path, sr=48000)` (CLAP expects 48k usually, check processor).
    *   Crop: `audio[int(t_start*sr) : int(t_end*sr)]`.
    *   Preprocess: `processor(audios, sampling_rate=48000, return_tensors="pt")`.
    *   Inference: `model(**inputs).pooler_output`.
    *   **Output**: Save `embeddings.npy` (Shape: `[N, 768]`) and `labels.npy`.

### Module 3: Probing & Evaluation
**File**: `src/train_probe.py`

**Functions**:
1.  `load_data()`:
    *   Load `.npy` files.
    *   **Track Split**: Ensure segments from the same *Track ID* are strictly in Train OR Test. Do NOT split randomly by segment.
    *   Split: 80% Tracks Train, 20% Tracks Test.
2.  `train_classifier(X_train, y_train)`:
    *   Use `sklearn.linear_model.LogisticRegression(max_iter=1000)` or `torch.nn.Linear`.
    *   Baseline: Simple Softmax Classifier.
3.  `evaluate(model, X_test, y_test)`:
    *   Compute **Accuracy**.
    *   Compute **Macro F1-Score**.
    *   Confusion Matrix (Optional).

---

## 3. 📅 Execution Steps (Immediate)

1.  **Setup**: Create `src/` and `data_artifacts/` directories.
2.  **Step 1**: Run `python src/prep_data.py`. Verify CSV.
3.  **Step 2**: Run `python src/extract_features.py`. Verify `embeddings.npy` shape.
4.  **Step 3**: Run `python src/train_probe.py`. Report metrics.

---

## 4. 📝 Dependencies

```txt
torch
transformers
librosa
jams
pandas
numpy
scikit-learn
tqdm
soundfile
```
