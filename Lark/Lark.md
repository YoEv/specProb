SPECTRAL PROBING OF CLAP AUDIO EMBEDDINGS

1. PROJECT OVERVIEW

Objective: Investigate where chord information is encoded in the temporal frequency spectrum of CLAP (HTSAT) representations.
Dataset: GuitarSet (Bossa Nova & Singer-Songwriter, Accompaniment only).
Task: 22-way Chord Classification (Root:Quality).

2. METHODOLOGY

Sequence Construction:
Source: CLAP HTSAT final layer output (Batch, 768, 2, 32).
Flattening: Flatten Freq(2) and Time(32) axes to create a sequence of length N=64.
Resolution: Input ~10s -> 32 time steps = ~312ms/token resolution.

Spectral Analysis:
Transform: Discrete Cosine Transform (DCT-II) along the flattened temporal axis.
Band Filtering: 5 bands adapted from Tamkin et al. (2020) with logarithmic scaling.
Feature Extraction: RMS Energy per band.

Frequency Bands Definition:
Low (Index 0): Static (DC)
Mid-Low (Index 1): Slow Drift (~20s period)
Mid (Indices 2-4): Phrase Level (5-10s period)
Mid-High (Indices 5-16): Chord/Bar Level (1.25-4s period)
High (Indices 17-63): Beat/Transient (0.3-1.2s period)

3. KEY RESULTS

Baseline (Mean Pool): 25.27% Accuracy.

Spectral Probing Insights:
Mid-Low (23.63%): Best single band. Captures slow tonal context.
Low (21.43%): Static average is strong but less informative than slow drift.
High (21.43%): Robust info despite low energy; redundant encoding.
Mid-High (19.78%): Lowest performance despite aligning with chord duration.

Conclusion: CLAP prioritizes global contextual information (Mid-Low) over precise local features (Mid-High), likely due to its coarse temporal resolution (312ms).

--------------------------------------------------

4. Q&A SUMMARY

Task Finding & Specifics
Time Resolution: CLAP's resolution is coarse (~312ms/token), favoring global semantic tasks over micro-rhythm.
Onset vs Rhythm: Resolution is too coarse for precise onset but sufficient for macro-rhythm.
Chords vs Key: Results suggest CLAP encodes "Key/Tonal Center" (long-term) more strongly than specific "Chord" changes.
Synthesized Rhythm Dataset: Future work. Synthesize 4th/8th/16th note pulses to determine encoder cutoff frequency.

Technical Issues
DCT Type: Used DCT-II (implemented via fast FFT, Makhoul 1980). Ideal for energy compaction.
Amplitude/Phase: Used RMS Energy to discard phase and focus on activity level. Applied along the *time/frequency* axis for each of the 768 channels individually, preserving semantic feature dimensions while collapsing temporal ones.
FT Complexity: Avoided Fourier Transform to stay in real domain.

CLAP Time Resolution
Data Points: ~3.2 tokens/second (32 tokens / 10s).
Implication: Low frame rate explains why slow features (Mid-Low) outperform chord-rate features (Mid-High).

Original Setup
DCT Setup: Global DCT over the entire sequence (N=64). No sliding window due to short sequence length.
