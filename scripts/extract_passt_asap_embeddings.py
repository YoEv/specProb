"""
Extract PaSST embeddings for the D_asap_100 classical composer dataset,
mirroring the CLAP ASAP extraction, and save them as an NPZ file.

Output NPZ keys:
    - 'embeddings': np.ndarray, shape (B, F_passt, T_passt)
    - 'composers':  np.ndarray[str], shape (B,)
    - 'file_paths': np.ndarray[str], shape (B,)
"""

import argparse
import glob
import os
from typing import List, Tuple, Optional

import librosa
import numpy as np
import torch
from hear21passt.base import get_timestamp_embeddings, load_model
from tqdm.auto import tqdm


AUDIO_DIR = "/home/evev/noiseloss/datasets/D_asap_100"
OUTPUT_DIR = "data_artifacts"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "passt_embeddings_asap_t32.npz")
SAMPLE_RATE = 32000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract PaSST embeddings for the ASAP dataset."
    )
    parser.add_argument(
        "--audio_dir",
        default=AUDIO_DIR,
        help=f"Directory containing ASAP WAV files (default: {AUDIO_DIR}).",
    )
    parser.add_argument(
        "--output_file",
        default=OUTPUT_FILE,
        help=f"Output NPZ path (default: {OUTPUT_FILE}).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to use, e.g. 'cuda', 'cuda:0', or 'cpu'. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Optional cap on the number of ASAP WAV files to process.",
    )
    return parser.parse_args()


def _load_passt_model(device: str) -> torch.nn.Module:
    print("[extract_passt_asap] Loading PaSST model (hear21passt.base)...")
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
    """Extract PaSST timestamp embeddings for a single ASAP WAV file."""
    waveform, _ = librosa.load(audio_path, sr=sr, mono=True)
    # Centre crop / pad to 10 seconds for comparability with CLAP ASAP setup.
    target_len = sr * 10
    if len(waveform) > target_len:
        start = (len(waveform) - target_len) // 2
        waveform = waveform[start : start + target_len]
    elif len(waveform) < target_len:
        pad = target_len - len(waveform)
        waveform = np.pad(waveform, (0, pad), mode="constant")

    audio_tensor = torch.from_numpy(waveform).float().unsqueeze(0).to(device)
    with torch.no_grad():
        emb, _ = get_timestamp_embeddings(audio_tensor, model)  # (1, T_frames, D)

    emb = emb.squeeze(0).cpu().numpy()  # (T_frames, D)
    emb_np = emb.T  # (D, T_frames)
    return emb_np


def extract_passt_asap_embeddings(
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

    model = _load_passt_model(device)

    embeddings_list: List[np.ndarray] = []
    composers: List[str] = []
    file_paths: List[str] = []

    print(f"[extract_passt_asap] Found {len(wav_paths)} WAV files in {audio_dir}")

    for audio_path in tqdm(wav_paths, desc="PaSST ASAP embeddings"):
        try:
            emb = _extract_passt_for_file(audio_path, model=model, device=device)
            embeddings_list.append(emb)

            basename = os.path.basename(audio_path)
            composer = basename.split("_", 1)[0]
            composers.append(composer)
            file_paths.append(audio_path)
        except Exception as e:
            print(f"[extract_passt_asap] Error processing {audio_path}: {e}")
            continue

    if not embeddings_list:
        raise RuntimeError("[extract_passt_asap] No embeddings were extracted.")

    max_T = max(emb.shape[1] for emb in embeddings_list)
    F = embeddings_list[0].shape[0]

    padded = np.zeros((len(embeddings_list), F, max_T), dtype=np.float32)
    for i, emb in enumerate(embeddings_list):
        T_i = emb.shape[1]
        padded[i, :, :T_i] = emb

    return padded, composers, file_paths


def main() -> None:
    args = parse_args()
    output_dir = os.path.dirname(args.output_file) or "."
    os.makedirs(output_dir, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[extract_passt_asap] Using device: {device}")

    embeddings, composers, file_paths = extract_passt_asap_embeddings(
        args.audio_dir,
        device=device,
        max_files=args.max_files,
    )

    print(
        f"[extract_passt_asap] Final embeddings shape: {embeddings.shape} "
        f"(B={embeddings.shape[0]}, F={embeddings.shape[1]}, T={embeddings.shape[2]})"
    )

    print(f"[extract_passt_asap] Saving NPZ to {args.output_file}...")
    np.savez_compressed(
        args.output_file,
        embeddings=embeddings,
        composers=np.asarray(composers),
        file_paths=np.asarray(file_paths),
    )
    print("[extract_passt_asap] Done.")


if __name__ == "__main__":
    main()

