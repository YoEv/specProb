"""
Extract CLAP multi-layer embeddings and save to a multi-layer NPZ.

This uses `src.features.extraction.extract_embeddings_multi_layer` and writes:

    data_artifacts/clap_embeddings_t64_multilayer.npz

with keys:
    - embeddings: (B, L, 768, 2, 32)
    - layers: (L,)
    - genres: (B,)
    - track_ids: (B,)

After运行本脚本，你就可以用:

    python scripts/run_multi_layer_spectral.py

来做多层 spectral profile 分析。
"""
import os

import numpy as np
import pandas as pd
import torch

from src.features.extraction import (
    METADATA_PATH,
    EMBEDDINGS_MULTI_FILE,
    extract_embeddings_multi_layer,
    load_model,
)


def main():
    # 1) Load CLAP model & processor
    model, processor = load_model()
    if model is None or processor is None:
        print("[extract_clap_multilayer] Failed to load CLAP model; aborting.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[extract_clap_multilayer] Using device: {device}")

    # 2) Load metadata
    if not os.path.exists(METADATA_PATH):
        print(f"[extract_clap_multilayer] Metadata not found at {METADATA_PATH}")
        return

    metadata = pd.read_csv(METADATA_PATH)
    print(f"[extract_clap_multilayer] Loaded metadata with {len(metadata)} tracks.")

    # 3) Extract multi-layer embeddings
    # layer_indices=None → 默认取第一层和最后一层；如需其它层可改成 [0, 6, 11] 等。
    embeddings, layers, genres, track_ids = extract_embeddings_multi_layer(
        metadata_df=metadata,
        model=model,
        processor=processor,
        device=device,
        layer_indices=None,
    )

    # 4) Save to NPZ
    os.makedirs(os.path.dirname(EMBEDDINGS_MULTI_FILE), exist_ok=True)
    np.savez_compressed(
        EMBEDDINGS_MULTI_FILE,
        embeddings=embeddings,
        layers=layers,
        genres=np.array(genres),
        track_ids=np.array(track_ids),
    )
    print(
        f"[extract_clap_multilayer] Saved multi-layer embeddings to {EMBEDDINGS_MULTI_FILE} "
        f"with shape {embeddings.shape} and layers={layers}."
    )


if __name__ == "__main__":
    main()

