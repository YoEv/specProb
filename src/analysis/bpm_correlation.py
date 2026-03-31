import os
from typing import List, Tuple

import librosa
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from src.data_processing.loader import load_data
from src.analysis.spectral import apply_transform


EMBEDDINGS_PATH = "data_artifacts/clap_embeddings_t64.npz"
METADATA_PATH = "data_artifacts/fma_metadata.csv"
RESULTS_DIR = "results/peak_artifact_investigation"
SAMPLE_RATE = 48000
N_COEFFS = 33  # rfft(64)


def _ensure_results_dir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


def estimate_bpm_for_all_tracks(
    metadata_path: str = METADATA_PATH,
    max_tracks: int = 2000,
) -> pd.DataFrame:
    """Estimate BPM for tracks in metadata using librosa.beat.tempo."""
    _ensure_results_dir()

    df = pd.read_csv(metadata_path)
    if max_tracks is not None and max_tracks > 0:
        df = df.iloc[:max_tracks].copy()

    bpms: List[float] = []
    track_ids: List[int] = []
    genres: List[str] = []

    for _, row in df.iterrows():
        track_id = int(row["track_id"])
        genre = str(row["genre"])
        audio_path = row["audio_path"]
        try:
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=30)
            tempo = librosa.beat.tempo(y=y, sr=sr)
            bpm = float(tempo[0]) if tempo.size > 0 else np.nan
        except Exception as e:
            print(f"[bpm_correlation] Error processing {audio_path}: {e}")
            bpm = np.nan

        track_ids.append(track_id)
        genres.append(genre)
        bpms.append(bpm)

    bpm_df = pd.DataFrame(
        {"track_id": track_ids, "genre": genres, "bpm": bpms},
    )

    out_csv = os.path.join(RESULTS_DIR, "bpm_estimates.csv")
    bpm_df.to_csv(out_csv, index=False)
    print(f"[bpm_correlation] Saved BPM estimates to {out_csv}")
    return bpm_df


def _compute_peak_indices_from_data(X: np.ndarray, k: int = 3) -> np.ndarray:
    """Compute top-k peak indices from global mean FFT magnitude."""
    coeffs = apply_transform(X, transform_type="fft", axis=2, window_type=None)
    mean_spec = np.mean(coeffs, axis=(0, 1))  # (N_COEFFS,)
    peak_indices = np.argsort(mean_spec)[-k:]
    peak_indices = np.sort(peak_indices)
    print(f"[bpm_correlation] Peak indices (global): {peak_indices.tolist()}")
    return peak_indices


def compute_bpm_energy_correlation(
    embeddings_path: str = EMBEDDINGS_PATH,
    bpm_df: pd.DataFrame | None = None,
) -> None:
    """Compute correlation between BPM and energy at the 3 peak frequency indices."""
    _ensure_results_dir()

    if bpm_df is None:
        if not os.path.exists(os.path.join(RESULTS_DIR, "bpm_estimates.csv")):
            bpm_df = estimate_bpm_for_all_tracks()
        else:
            bpm_df = pd.read_csv(os.path.join(RESULTS_DIR, "bpm_estimates.csv"))

    # Load embeddings + track_ids
    with np.load(embeddings_path) as data:
        X_raw = data["embeddings"]  # (B, 768, 2, 64)
        genres_np = data["genres"]
        track_ids_np = data["track_ids"].astype(int)

    # Use loader to get (B, 1536, 64)
    X, _ = load_data(embeddings_path)

    # Align bpm_df with embedding order via track_id
    bpm_map = dict(zip(bpm_df["track_id"].astype(int), bpm_df["bpm"]))
    bpms: List[float] = []
    for tid in track_ids_np:
        bpms.append(float(bpm_map.get(int(tid), np.nan)))
    bpms = np.array(bpms, dtype=float)

    # Compute peak indices from data
    peak_indices = _compute_peak_indices_from_data(X, k=3)

    coeffs = apply_transform(X, transform_type="fft", axis=2, window_type=None)  # (B, F, N_COEFFS)
    # Energy at peaks per track: mean over features
    peak_energies = []
    for pi in peak_indices:
        energy_pi = np.mean(coeffs[:, :, pi], axis=1)  # (B,)
        peak_energies.append(energy_pi)
    peak_energies = np.stack(peak_energies, axis=1)  # (B, 3)

    # Global and per-genre correlations
    out_lines: List[str] = []
    out_lines.append("BPM vs peak energy correlation (global and per genre)\n")
    out_lines.append(f"Peak indices: {peak_indices.tolist()}\n")

    def _corr(vec_bpm: np.ndarray, vec_e: np.ndarray) -> Tuple[float, float]:
        mask = np.isfinite(vec_bpm) & np.isfinite(vec_e)
        if np.sum(mask) < 10:
            return np.nan, np.nan
        r, p = pearsonr(vec_bpm[mask], vec_e[mask])
        return float(r), float(p)

    # Global
    out_lines.append("Global:\n")
    for j, pi in enumerate(peak_indices):
        r, p = _corr(bpms, peak_energies[:, j])
        out_lines.append(f"  Peak {pi}: r={r:.4f}, p={p:.4e}\n")

    # Per-genre
    out_lines.append("\nPer-genre:\n")
    for g in np.unique(genres_np):
        mask_g = genres_np == g
        if np.sum(mask_g) < 20:
            continue
        out_lines.append(f"Genre: {g}\n")
        for j, pi in enumerate(peak_indices):
            r, p = _corr(bpms[mask_g], peak_energies[mask_g, j])
            out_lines.append(f"  Peak {pi}: r={r:.4f}, p={p:.4e}\n")
        out_lines.append("\n")

    out_path = os.path.join(RESULTS_DIR, "bpm_correlation.txt")
    with open(out_path, "w") as f:
        f.writelines(out_lines)
    print(f"[bpm_correlation] Saved BPM correlation report to {out_path}")


if __name__ == "__main__":
    bpm_df = estimate_bpm_for_all_tracks()
    compute_bpm_energy_correlation(embeddings_path=EMBEDDINGS_PATH, bpm_df=bpm_df)

