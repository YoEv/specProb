# Scripts

GPU allocation, PaSST FMA sharding, resume/checkpoints, and merging shards are documented in:

**[projPlan/setup_server.md §8](../projPlan/setup_server.md)** — *GPU jobs and PaSST FMA embedding extraction (PACE)*

Related entry points:

| Script | Role |
|--------|------|
| `extract_passt_fma_embeddings.py` | PaSST embeddings for FMA-style metadata CSV; `--num_shards` / `--shard_index`, checkpoints, resume |
| `merge_fma_embedding_npz_shards.py` | Merge shard NPZ files after all shards complete |
| `convert_fma_metadata_audio_to_wav.py` | Optional WAV mirror for faster I/O (see extraction script help) |
