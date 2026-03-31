"""
Extract AudioMAE embeddings for the FMA dataset and save NPZ outputs
compatible with the existing probing pipeline.

Output NPZ keys:
    - embeddings: np.ndarray, shape (B, F_audiomae, T=64)
    - genres: np.ndarray[str], shape (B,)
    - file_paths: np.ndarray[str], shape (B,)
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel


MODEL_ID = "hance-ai/audiomae"
METADATA_PATH = "data_artifacts/fma_metadata.csv"
OUTPUT_DIR = "data_artifacts"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "audiomae_embeddings_t64.npz")


def _load_model(device: str):
    print(f"[extract_audiomae_fma] Loading model: {MODEL_ID}")
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.to(device)
    model.eval()
    return model


def _extract_single(audio_path: str, model, device: str) -> np.ndarray:
    """
    Returns embedding in (F, T) format to match existing NPZ convention.
    AudioMAE output from this checkpoint is typically (768, 8, 64), where:
        - 768: channel/hidden dimension
        - 8: latent frequency bins
        - 64: latent time bins

    We reshape (768, 8, 64) -> (768*8, 64) = (6144, 64) to keep the time axis
    as the last dimension, because downstream FFT probing expects (B, F, T) and
    applies transform on axis=2.
    """
    with torch.no_grad():
        emb = model(audio_path)

    if isinstance(emb, torch.Tensor):
        emb = emb.detach().to(device)
    else:
        raise TypeError(f"Unexpected embedding type: {type(emb)}")

    emb_np = emb.cpu().numpy().astype(np.float32)
    if emb_np.ndim != 3:
        raise ValueError(f"Expected AudioMAE output ndim=3, got shape={emb_np.shape}")

    # Flatten channel and latent-frequency axes; preserve latent-time axis.
    # This is the project-standard representation for generic FFT probing.
    c, h, t = emb_np.shape
    return emb_np.reshape(c * h, t)


def extract_embeddings(
    metadata_df: pd.DataFrame,
    device: str,
    max_tracks: Optional[int] = 800,
    max_per_genre: int = 100,
) -> Tuple[np.ndarray, List[str], List[str]]:
    if "audio_path" not in metadata_df.columns or "genre" not in metadata_df.columns:
        raise ValueError(
            "Metadata CSV must contain 'audio_path' and 'genre' columns."
        )

    # Stratified subsample for parity with existing runs.
    sampled_indices: List[int] = []
    for _, group in metadata_df.groupby("genre"):
        idx = group.index.to_list()
        if len(idx) > max_per_genre:
            idx = idx[:max_per_genre]
        sampled_indices.extend(idx)
    sampled_indices = sorted(sampled_indices)
    if max_tracks is not None and len(sampled_indices) > max_tracks:
        sampled_indices = sampled_indices[:max_tracks]
    sampled_df = metadata_df.loc[sampled_indices].reset_index(drop=True)

    model = _load_model(device)

    embeddings_list: List[np.ndarray] = []
    genres: List[str] = []
    file_paths: List[str] = []

    print(f"[extract_audiomae_fma] Extracting {len(sampled_df)} tracks...")
    for _, row in tqdm(sampled_df.iterrows(), total=sampled_df.shape[0], desc="AudioMAE FMA"):
        audio_path = row.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            continue
        try:
            emb = _extract_single(audio_path=audio_path, model=model, device=device)
            embeddings_list.append(emb)
            genres.append(str(row.get("genre", "")))
            file_paths.append(audio_path)
        except Exception as e:
            print(f"[extract_audiomae_fma] Error processing {audio_path}: {e}")
            continue

    if not embeddings_list:
        raise RuntimeError("No embeddings extracted for FMA.")

    # T should already be fixed at 64; keep pad logic for robustness.
    max_t = max(x.shape[1] for x in embeddings_list)
    f_dim = embeddings_list[0].shape[0]
    padded = np.zeros((len(embeddings_list), f_dim, max_t), dtype=np.float32)
    for i, emb in enumerate(embeddings_list):
        t_i = emb.shape[1]
        padded[i, :, :t_i] = emb

    return padded, genres, file_paths


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[extract_audiomae_fma] Using device: {device}")

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_PATH}")

    metadata = pd.read_csv(METADATA_PATH)
    embeddings, genres, file_paths = extract_embeddings(
        metadata_df=metadata,
        device=device,
        max_tracks=800,
        max_per_genre=100,
    )

    print(
        f"[extract_audiomae_fma] Final embeddings shape: {embeddings.shape} "
        f"(B={embeddings.shape[0]}, F={embeddings.shape[1]}, T={embeddings.shape[2]})"
    )
    print(f"[extract_audiomae_fma] Saving NPZ to {OUTPUT_FILE}")
    np.savez_compressed(
        OUTPUT_FILE,
        embeddings=embeddings,
        genres=np.asarray(genres),
        file_paths=np.asarray(file_paths),
    )
    print("[extract_audiomae_fma] Done.")


if __name__ == "__main__":
    main()
