"""
Extract BEATs embeddings for the D_asap_100 classical composer dataset,
与 CLAP / PaSST 的 ASAP 抽取保持一致。

输出 NPZ:
    - 'embeddings': np.ndarray, shape (B, F_beats, T_beats)
    - 'composers':  np.ndarray[str], shape (B,)
    - 'file_paths': np.ndarray[str], shape (B,)
"""

import glob
import os
from typing import List, Tuple, Optional

import librosa
import numpy as np
import torch
from tqdm.auto import tqdm

from third_party.beats_loader import load_beats_model


AUDIO_DIR = "/home/evev/noiseloss/datasets/D_asap_100"
OUTPUT_DIR = "data_artifacts"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "beats_embeddings_asap_t32.npz")
SAMPLE_RATE = 16000
BEATS_CKPT_PATH = os.path.join(OUTPUT_DIR, "BEATs_iter3_finetuned_on_AS2M_cpt1.pt")

def _load_beats_model(device: str):
    """
    Load BEATs model from an official checkpoint using the local loader.

    Make sure BEATS_CKPT_PATH points to a valid *.pt file downloaded from
    the official BEATs repository.
    """
    print(f"[extract_beats_asap] Loading BEATs model from {BEATS_CKPT_PATH} ...")
    return load_beats_model(BEATS_CKPT_PATH, device=device)


def _extract_beats_for_file(
    audio_path: str,
    model,
    device: str,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """对单条 ASAP WAV 抽取 BEATs 时间序列 embedding, 统一为 (D, T_frames)."""
    waveform, _ = librosa.load(audio_path, sr=sr, mono=True)
    # 10 秒中心裁剪，保持与 CLAP/PaSST 一致
    target_len = sr * 10
    if len(waveform) > target_len:
        start = (len(waveform) - target_len) // 2
        waveform = waveform[start : start + target_len]
    elif len(waveform) < target_len:
        pad = target_len - len(waveform)
        waveform = np.pad(waveform, (0, pad), mode="constant")

    wav_tensor = torch.from_numpy(waveform).float().unsqueeze(0).to(device)  # (1, T)
    padding_mask = torch.zeros(wav_tensor.shape, dtype=torch.bool, device=device)

    with torch.no_grad():
        # Official BEATs interface: returns (features, padding_mask)
        feats, _ = model.extract_features(wav_tensor, padding_mask=padding_mask)

    # 通常 feats: (1, T_frames, D)，但为安全起见做一次规范化
    feats = feats.squeeze(0)  # 期望 (T_frames, D)
    if feats.ndim == 1:
        # 极端退化情况：只有时间维，当作 D=1 处理
        feats = feats.unsqueeze(-1)  # (T_frames, 1)
    elif feats.ndim != 2:
        raise RuntimeError(f"Unexpected BEATs feature shape {feats.shape} for {audio_path}")

    emb = feats.cpu().numpy()  # (T_frames, D)
    emb_np = emb.T  # (D, T_frames)
    return emb_np


def extract_beats_asap_embeddings(
    audio_dir: str,
    device: str,
    max_files: Optional[int] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    pattern = os.path.join(audio_dir, "*.wav")
    wav_paths = sorted(glob.glob(pattern))
    if not wav_paths:
        raise FileNotFoundError(f"No WAV files found under {audio_dir}")
    if max_files is not None and len(wav_paths) > max_files:
        wav_paths = wav_paths[:max_files]

    model = _load_beats_model(device)

    embeddings_list: List[np.ndarray] = []
    composers: List[str] = []
    file_paths: List[str] = []

    print(f"[extract_beats_asap] Found {len(wav_paths)} WAV files in {audio_dir}")

    for audio_path in tqdm(wav_paths, desc="BEATs ASAP embeddings"):
        try:
            emb = _extract_beats_for_file(audio_path, model=model, device=device)
            if emb.ndim != 2:
                print(f"[extract_beats_asap] Skipping {audio_path}: unexpected emb shape {emb.shape}")
                continue
            embeddings_list.append(emb)

            basename = os.path.basename(audio_path)
            composer = basename.split("_", 1)[0]
            composers.append(composer)
            file_paths.append(audio_path)
        except Exception as e:
            print(f"[extract_beats_asap] Error processing {audio_path}: {e}")
            continue

    if not embeddings_list:
        raise RuntimeError("[extract_beats_asap] No embeddings were extracted.")

    max_T = max(emb.shape[1] for emb in embeddings_list)
    F = embeddings_list[0].shape[0]
    padded = np.zeros((len(embeddings_list), F, max_T), dtype=np.float32)
    for i, emb in enumerate(embeddings_list):
        T_i = emb.shape[1]
        padded[i, :, :T_i] = emb

    return padded, composers, file_paths


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[extract_beats_asap] Using device: {device}")

    embeddings, composers, file_paths = extract_beats_asap_embeddings(
        AUDIO_DIR,
        device=device,
        max_files=None,
    )

    print(
        f"[extract_beats_asap] Final embeddings shape: {embeddings.shape} "
        f"(B={embeddings.shape[0]}, F={embeddings.shape[1]}, T={embeddings.shape[2]})"
    )

    print(f"[extract_beats_asap] Saving NPZ to {OUTPUT_FILE}...")
    np.savez_compressed(
        OUTPUT_FILE,
        embeddings=embeddings,
        composers=np.asarray(composers),
        file_paths=np.asarray(file_paths),
    )
    print("[extract_beats_asap] Done.")


if __name__ == "__main__":
    main()

