"""
Extract PaSST embeddings for the FMA / LMD dataset, using the same metadata
CSV as the CLAP pipeline, and save them as an NPZ file consumable by the
multi-model periodicity analysis.

Output NPZ keys:
    - 'embeddings': np.ndarray, shape (B, F_passt, T_passt)
    - 'genres':     np.ndarray[str], shape (B,)
    - 'file_paths': np.ndarray[str], shape (B,)

The target path should match the PaSST spec in src/config/multi_model_sanity.py,
currently: data_artifacts/passt_embeddings_t64.npz (name kept for consistency,
even if T_passt != 64).
"""

import os
from typing import List, Tuple, Optional

import librosa
import numpy as np
import pandas as pd
import torch
from hear21passt.base import get_timestamp_embeddings, load_model
from tqdm.auto import tqdm


METADATA_PATH = "data_artifacts/fma_metadata.csv"
OUTPUT_DIR = "data_artifacts"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "passt_embeddings_t64.npz")
SAMPLE_RATE = 32000  # PaSST default in hear21passt examples


def _load_passt_model(device: str) -> torch.nn.Module:
    """Load PaSST model via hear21passt."""
    print("[extract_passt_fma] Loading PaSST model (hear21passt.base)...")
    model = load_model()
    model.to(device)
    model.eval()
    return model


def _extract_passt_for_file(
    audio_path: str,
    model: torch.nn.Module,
    device: str,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Extract PaSST timestamp embeddings for a single audio file.

    Returns:
        emb_np: np.ndarray of shape (F_passt, T_passt)
    """
    waveform, _ = librosa.load(audio_path, sr=sr, mono=True)
    # hear21passt expects (B, T) tensor
    audio_tensor = torch.from_numpy(waveform).float().unsqueeze(0).to(device)

    with torch.no_grad():
        # get_timestamp_embeddings returns (B, T_frames, D)
        emb, _ = get_timestamp_embeddings(audio_tensor, model)

    emb = emb.squeeze(0).cpu().numpy()  # (T_frames, D)
    emb_np = emb.T  # (D, T_frames)
    return emb_np


def extract_passt_fma_embeddings(
    metadata_df: pd.DataFrame,
    device: str,
    max_tracks: Optional[int] = 800,
    max_per_genre: int = 100,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Iterate over FMA metadata, extract PaSST embeddings per track, and stack.

    If max_tracks / max_per_genre are set, we perform a simple stratified
    subsample: up to `max_per_genre` tracks per genre, capped by max_tracks
    in total（例如 8 个 genre × 100 ≈ 800 条）.

    Returns:
        embeddings: np.ndarray, shape (B, F_passt, T_passt)
        genres:     list[str]
        file_paths: list[str]
    """
    if "genre" not in metadata_df.columns:
        raise ValueError(
            f"Metadata CSV must contain 'genre' column for stratified sampling, "
            f"got columns: {list(metadata_df.columns)}"
        )

    # Stratified sampling: up to max_per_genre per genre.
    sampled_indices: List[int] = []
    for genre, group in metadata_df.groupby("genre"):
        idx = group.index.to_list()
        if len(idx) > max_per_genre:
            idx = idx[:max_per_genre]
        sampled_indices.extend(idx)
    sampled_indices = sorted(sampled_indices)
    if max_tracks is not None and len(sampled_indices) > max_tracks:
        sampled_indices = sampled_indices[:max_tracks]

    sampled_df = metadata_df.loc[sampled_indices].reset_index(drop=True)

    model = _load_passt_model(device)

    embeddings_list: List[np.ndarray] = []
    genres: List[str] = []
    file_paths: List[str] = []

    print(f"[extract_passt_fma] Starting extraction for {len(sampled_df)} tracks...")
    for _, row in tqdm(
        sampled_df.iterrows(),
        total=sampled_df.shape[0],
        desc="PaSST FMA embeddings",
    ):
        audio_path = row.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            continue
        try:
            emb = _extract_passt_for_file(audio_path, model=model, device=device)
            embeddings_list.append(emb)
            genres.append(str(row.get("genre", "")))
            file_paths.append(audio_path)
        except Exception as e:
            print(f"[extract_passt_fma] Error processing {audio_path}: {e}")
            continue

    if not embeddings_list:
        raise RuntimeError("[extract_passt_fma] No embeddings were extracted.")

    # Handle variable T_passt across files, if any, by right-padding to max length.
    max_T = max(emb.shape[1] for emb in embeddings_list)
    F = embeddings_list[0].shape[0]

    padded = np.zeros((len(embeddings_list), F, max_T), dtype=np.float32)
    for i, emb in enumerate(embeddings_list):
        T_i = emb.shape[1]
        padded[i, :, :T_i] = emb

    return padded, genres, file_paths


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[extract_passt_fma] Using device: {device}")

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata CSV not found at {METADATA_PATH}. "
            "Please run the FMA data preparation first."
        )

    print(f"[extract_passt_fma] Loading metadata from {METADATA_PATH}...")
    metadata = pd.read_csv(METADATA_PATH)
    if "audio_path" not in metadata.columns or "genre" not in metadata.columns:
        raise ValueError(
            f"Metadata CSV must contain 'audio_path' and 'genre' columns, "
            f"got columns: {list(metadata.columns)}"
        )

    embeddings, genres, file_paths = extract_passt_fma_embeddings(
        metadata_df=metadata,
        device=device,
        max_tracks=800,
        max_per_genre=100,
    )

    print(
        f"[extract_passt_fma] Final embeddings shape: {embeddings.shape} "
        f"(B={embeddings.shape[0]}, F={embeddings.shape[1]}, T={embeddings.shape[2]})"
    )

    print(f"[extract_passt_fma] Saving NPZ to {OUTPUT_FILE}...")
    np.savez_compressed(
        OUTPUT_FILE,
        embeddings=embeddings,
        genres=np.asarray(genres),
        file_paths=np.asarray(file_paths),
    )
    print("[extract_passt_fma] Done.")


if __name__ == "__main__":
    main()

