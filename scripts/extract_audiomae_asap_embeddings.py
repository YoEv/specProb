"""
Extract AudioMAE embeddings for ASAP composer dataset and save NPZ outputs
compatible with the existing probing pipeline.

Output NPZ keys:
    - embeddings: np.ndarray, shape (B, F_audiomae, T=64)
    - composers: np.ndarray[str], shape (B,)
    - file_paths: np.ndarray[str], shape (B,)
"""

import glob
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModel


MODEL_ID = "hance-ai/audiomae"
AUDIO_DIR = "/home/evev/noiseloss/datasets/D_asap_100"
OUTPUT_DIR = "data_artifacts"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "audiomae_embeddings_asap_t32.npz")


def _load_model(device: str):
    print(f"[extract_audiomae_asap] Loading model: {MODEL_ID}")
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.to(device)
    model.eval()
    return model


def _extract_single(audio_path: str, model, device: str) -> np.ndarray:
    with torch.no_grad():
        emb = model(audio_path)

    if isinstance(emb, torch.Tensor):
        emb = emb.detach().to(device)
    else:
        raise TypeError(f"Unexpected embedding type: {type(emb)}")

    emb_np = emb.cpu().numpy().astype(np.float32)
    if emb_np.ndim != 3:
        raise ValueError(f"Expected AudioMAE output ndim=3, got shape={emb_np.shape}")

    # AudioMAE tensor is typically (768, 8, 64): (channel, freq-bin, time-bin).
    # Flatten channel and freq-bin into feature axis; keep time as the last axis
    # for downstream FFT probing on axis=2.
    c, h, t = emb_np.shape
    return emb_np.reshape(c * h, t)


def extract_embeddings(
    audio_dir: str,
    device: str,
    max_files: Optional[int] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    wav_paths = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    if not wav_paths:
        raise FileNotFoundError(f"No WAV files found under {audio_dir}")
    if max_files is not None and len(wav_paths) > max_files:
        wav_paths = wav_paths[:max_files]

    model = _load_model(device)

    embeddings_list: List[np.ndarray] = []
    composers: List[str] = []
    file_paths: List[str] = []

    print(f"[extract_audiomae_asap] Found {len(wav_paths)} WAV files.")
    for audio_path in tqdm(wav_paths, desc="AudioMAE ASAP"):
        try:
            emb = _extract_single(audio_path=audio_path, model=model, device=device)
            embeddings_list.append(emb)
            composers.append(os.path.basename(audio_path).split("_", 1)[0])
            file_paths.append(audio_path)
        except Exception as e:
            print(f"[extract_audiomae_asap] Error processing {audio_path}: {e}")
            continue

    if not embeddings_list:
        raise RuntimeError("No embeddings extracted for ASAP.")

    max_t = max(x.shape[1] for x in embeddings_list)
    f_dim = embeddings_list[0].shape[0]
    padded = np.zeros((len(embeddings_list), f_dim, max_t), dtype=np.float32)
    for i, emb in enumerate(embeddings_list):
        t_i = emb.shape[1]
        padded[i, :, :t_i] = emb

    return padded, composers, file_paths


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[extract_audiomae_asap] Using device: {device}")

    embeddings, composers, file_paths = extract_embeddings(
        audio_dir=AUDIO_DIR,
        device=device,
        max_files=None,
    )

    print(
        f"[extract_audiomae_asap] Final embeddings shape: {embeddings.shape} "
        f"(B={embeddings.shape[0]}, F={embeddings.shape[1]}, T={embeddings.shape[2]})"
    )
    print(f"[extract_audiomae_asap] Saving NPZ to {OUTPUT_FILE}")
    np.savez_compressed(
        OUTPUT_FILE,
        embeddings=embeddings,
        composers=np.asarray(composers),
        file_paths=np.asarray(file_paths),
    )
    print("[extract_audiomae_asap] Done.")


if __name__ == "__main__":
    main()
