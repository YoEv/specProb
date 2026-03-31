"""
Extract BEATs embeddings for the FMA / LMD dataset, using the same metadata
CSV as the CLAP pipeline,并保存为 NPZ 供多模型对比使用。

输出 NPZ:
    - 'embeddings': np.ndarray, shape (B, F_beats, T_beats)
    - 'genres':     np.ndarray[str], shape (B,)
    - 'file_paths': np.ndarray[str], shape (B,)
"""

import os
from typing import List, Tuple, Optional

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from third_party.beats_loader import load_beats_model


METADATA_PATH = "data_artifacts/fma_metadata.csv"
OUTPUT_DIR = "data_artifacts"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "beats_embeddings_t64.npz")
SAMPLE_RATE = 16000  # BEATs 默认 16k
BEATS_CKPT_PATH = os.path.join(OUTPUT_DIR, "BEATs_iter3_finetuned_on_AS2M_cpt1.pt")

def _load_beats_model(device: str):
    """
    加载 BEATs 模型（使用官方 BEATs checkpoint + 本地 loader）。

    请确保 BEATS_CKPT_PATH 指向你下载的 *.pt 文件。
    """
    print(f"[extract_beats_fma] Loading BEATs model from {BEATS_CKPT_PATH} ...")
    return load_beats_model(BEATS_CKPT_PATH, device=device)


def _extract_beats_for_file(
    audio_path: str,
    model,
    device: str,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    对单条音频抽取 BEATs 时间序列 embedding, 统一为 (F_beats, T_beats).

    返回:
        emb_np: np.ndarray, shape (F_beats, T_beats)
    """
    waveform, _ = librosa.load(audio_path, sr=sr, mono=True)
    wav_tensor = torch.from_numpy(waveform).float().unsqueeze(0).to(device)  # (1, T)
    padding_mask = torch.zeros(wav_tensor.shape, dtype=torch.bool, device=device)

    with torch.no_grad():
        # 官方 BEATs 接口：extract_features -> (features, padding_mask)
        feats, _ = model.extract_features(wav_tensor, padding_mask=padding_mask)

    feats = feats.squeeze(0)  # 期望 (T_frames, D)
    if feats.ndim == 1:
        feats = feats.unsqueeze(-1)  # (T_frames, 1)
    elif feats.ndim != 2:
        raise RuntimeError(f"Unexpected BEATs feature shape {feats.shape} for {audio_path}")

    emb = feats.cpu().numpy()  # (T_frames, D)
    emb_np = emb.T  # (D, T_frames)
    return emb_np


def extract_beats_fma_embeddings(
    metadata_df: pd.DataFrame,
    device: str,
    max_tracks: Optional[int] = 800,
    max_per_genre: int = 100,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    在 FMA 上按 genre 分层抽样，最多每个 genre 100 条，总数 ~800 条，
    为每条 track 抽取 BEATs embedding 并堆叠。
    """
    if "genre" not in metadata_df.columns:
        raise ValueError(
            f"Metadata CSV must contain 'genre' column for stratified sampling, "
            f"got columns: {list(metadata_df.columns)}"
        )

    # 简单按 genre 分层采样
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

    model = _load_beats_model(device)

    embeddings_list: List[np.ndarray] = []
    genres: List[str] = []
    file_paths: List[str] = []

    print(f"[extract_beats_fma] Starting extraction for {len(sampled_df)} tracks...")
    for _, row in tqdm(
        sampled_df.iterrows(),
        total=sampled_df.shape[0],
        desc="BEATs FMA embeddings",
    ):
        audio_path = row.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            continue
        try:
            emb = _extract_beats_for_file(audio_path, model=model, device=device)
            if emb.ndim != 2:
                print(f"[extract_beats_fma] Skipping {audio_path}: unexpected emb shape {emb.shape}")
                continue
            embeddings_list.append(emb)
            genres.append(str(row.get("genre", "")))
            file_paths.append(audio_path)
        except Exception as e:
            print(f"[extract_beats_fma] Error processing {audio_path}: {e}")
            continue

    if not embeddings_list:
        raise RuntimeError("[extract_beats_fma] No embeddings were extracted.")

    # 右侧 padding 对齐时间长度
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
    print(f"[extract_beats_fma] Using device: {device}")

    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata CSV not found at {METADATA_PATH}. "
            "Please run the FMA data preparation first."
        )

    print(f"[extract_beats_fma] Loading metadata from {METADATA_PATH}...")
    metadata = pd.read_csv(METADATA_PATH)
    if "audio_path" not in metadata.columns or "genre" not in metadata.columns:
        raise ValueError(
            f"Metadata CSV must contain 'audio_path' and 'genre' columns, "
            f"got columns: {list(metadata.columns)}"
        )

    embeddings, genres, file_paths = extract_beats_fma_embeddings(
        metadata_df=metadata,
        device=device,
        max_tracks=800,
        max_per_genre=100,
    )

    print(
        f"[extract_beats_fma] Final embeddings shape: {embeddings.shape} "
        f"(B={embeddings.shape[0]}, F={embeddings.shape[1]}, T={embeddings.shape[2]})"
    )

    print(f"[extract_beats_fma] Saving NPZ to {OUTPUT_FILE}...")
    np.savez_compressed(
        OUTPUT_FILE,
        embeddings=embeddings,
        genres=np.asarray(genres),
        file_paths=np.asarray(file_paths),
    )
    print("[extract_beats_fma] Done.")


if __name__ == "__main__":
    main()

