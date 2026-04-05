"""
Merge multiple NPZ shards produced by extract_passt_fma_embeddings.py (or similar).

Expects each file to contain keys: embeddings, genres, file_paths.
Right-pads embeddings along the time axis to a common T before stacking.

Usage:
  python scripts/merge_fma_embedding_npz_shards.py \\
    --inputs data_artifacts/passt_fma_medium_shard0.npz \\
             data_artifacts/passt_fma_medium_shard1.npz \\
    --output data_artifacts/passt_embeddings_fma_medium_t64.npz
"""

from __future__ import annotations

import argparse
from typing import List

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge FMA-style embedding NPZ shards.")
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Shard NPZ paths in order (e.g. shard0, shard1, ...).",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Merged output NPZ path.",
    )
    return p.parse_args()


def _load_shard(path: str):
    d = np.load(path, allow_pickle=True)
    for k in ("embeddings", "genres", "file_paths"):
        if k not in d:
            raise KeyError(f"{path} missing key {k!r}")
    return d["embeddings"], d["genres"], d["file_paths"]


def merge_shards(paths: List[str]):
    if len(paths) < 1:
        raise ValueError("Need at least one input NPZ.")

    chunks_emb = []
    chunks_genre = []
    chunks_fp = []
    F = None
    max_T = 0

    for p in paths:
        emb, genres, fps = _load_shard(p)
        if emb.ndim != 3:
            raise ValueError(f"{p}: expected embeddings ndim=3, got {emb.shape}")
        if F is None:
            F = emb.shape[1]
        elif emb.shape[1] != F:
            raise ValueError(
                f"{p}: feature dim {emb.shape[1]} != first shard F={F}"
            )
        max_T = max(max_T, emb.shape[2])
        chunks_emb.append(emb)
        chunks_genre.append(np.asarray(genres))
        chunks_fp.append(np.asarray(fps))

    B = sum(e.shape[0] for e in chunks_emb)
    out = np.zeros((B, F, max_T), dtype=np.float32)
    genres_out = []
    fps_out = []
    row = 0
    for emb, g, fp in zip(chunks_emb, chunks_genre, chunks_fp):
        n = emb.shape[0]
        T_i = emb.shape[2]
        out[row : row + n, :, :T_i] = emb
        genres_out.append(g)
        fps_out.append(fp)
        row += n

    return (
        out,
        np.concatenate(genres_out, axis=0),
        np.concatenate(fps_out, axis=0),
    )


def main() -> None:
    args = parse_args()
    emb, genres, fps = merge_shards(args.inputs)
    np.savez_compressed(
        args.output,
        embeddings=emb,
        genres=genres,
        file_paths=fps,
    )
    print(
        f"[merge_fma_npz] Wrote {args.output} shape={emb.shape} "
        f"genres={len(genres)}"
    )


if __name__ == "__main__":
    main()
