"""
Batch-convert audio files listed in an FMA-style metadata CSV to 32 kHz mono WAV.

Typical use after extracting FMA zip with nested dirs, e.g.:

  python scripts/convert_fma_metadata_audio_to_wav.py \\
    --metadata_path data_artifacts/fma_medium_metadata.csv \\
    --strip_prefix data/fma_medium/fma_medium \\
    --output_root data/fma_medium_wav/fma_medium \\
    --num_workers 8

Then run PaSST extraction with:

  --wav_mirror_root data/fma_medium_wav/fma_medium \\
  --wav_strip_prefix data/fma_medium/fma_medium
"""

from __future__ import annotations

import argparse
import os
from multiprocessing import Pool
from typing import List, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm

SAMPLE_RATE = 32000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert metadata audio paths to WAV.")
    p.add_argument("--metadata_path", required=True)
    p.add_argument(
        "--strip_prefix",
        required=True,
        help="Absolute path prefix removed from each audio_path to get a relative key.",
    )
    p.add_argument(
        "--output_root",
        required=True,
        help="Root directory for mirrored .wav files (same relative layout as strip_prefix).",
    )
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-convert even if destination WAV already exists.",
    )
    return p.parse_args()


def _one(job: Tuple[str, str, bool]) -> str:
    src, dst, overwrite = job
    try:
        if os.path.exists(dst) and not overwrite:
            return "skip"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        y, _ = librosa.load(src, sr=SAMPLE_RATE, mono=True)
        sf.write(dst, y.astype(np.float32), SAMPLE_RATE, subtype="PCM_16")
        return "ok"
    except Exception as e:
        return f"err:{src}:{e}"


def build_jobs(
    df: pd.DataFrame,
    strip_prefix: str,
    output_root: str,
    overwrite: bool,
) -> List[Tuple[str, str, bool]]:
    strip_abs = os.path.abspath(strip_prefix)
    out_abs = os.path.abspath(output_root)
    jobs: List[Tuple[str, str, bool]] = []
    seen = set()
    for _, row in df.iterrows():
        src = row.get("audio_path")
        if not src or not isinstance(src, str):
            continue
        src_abs = os.path.abspath(src)
        if not src_abs.startswith(strip_abs):
            continue
        rel = os.path.relpath(src_abs, strip_abs)
        base, _ = os.path.splitext(rel)
        dst = os.path.join(out_abs, base + ".wav")
        key = (src_abs, dst)
        if key in seen:
            continue
        seen.add(key)
        if os.path.isfile(src_abs):
            jobs.append((src_abs, dst, overwrite))
    return jobs


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.metadata_path)
    if "audio_path" not in df.columns:
        raise ValueError("metadata CSV must contain audio_path")

    jobs = build_jobs(
        df,
        strip_prefix=args.strip_prefix,
        output_root=args.output_root,
        overwrite=args.overwrite,
    )
    print(f"[convert_wav] {len(jobs)} files to convert (or skip)")

    if args.num_workers <= 1:
        results = [_one(j) for j in tqdm(jobs, desc="convert wav")]
    else:
        with Pool(args.num_workers) as pool:
            results = list(
                tqdm(pool.imap_unordered(_one, jobs), total=len(jobs), desc="convert wav")
            )

    ok = sum(1 for r in results if r == "ok")
    skip = sum(1 for r in results if r == "skip")
    err = [r for r in results if r.startswith("err:")]
    print(f"[convert_wav] ok={ok} skip={skip} err={len(err)}")
    for r in err[:20]:
        print(r)
    if len(err) > 20:
        print(f"... and {len(err) - 20} more errors")


if __name__ == "__main__":
    main()
