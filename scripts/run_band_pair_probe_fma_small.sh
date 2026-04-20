#!/usr/bin/env bash
# M1 run for band-pair probing on PaSST FMA-small (800 clips).
#
# Stage 1 (this script): solo + 1+1 pair + cumulative prefix + band ranking.
#   Writes pair_matrix / synergy / redundancy heatmaps + ranking JSONs
#   *as soon as the pair phase finishes*, so a later time-out / OOM in the
#   triplet phase does not wipe out the main deliverables.
#
# This script ALSO runs the 1+2 triplet phase by default with a reduced
# epoch count (200) to keep total wall-clock tractable. If the triplet
# phase gets killed you'll still have all pair-level outputs.
#
# If you want an even faster pair-only run, pass --skip_triplets:
#   bash scripts/run_band_pair_probe_fma_small.sh --skip_triplets
#
# Usage (inside `specprob` conda env on a GPU node):
#   bash scripts/run_band_pair_probe_fma_small.sh [extra args passed through]

set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p results/3rd_exp

LOG=results/3rd_exp/passt_fma_small_band_pairs.log
: > "$LOG"

python -u scripts/run_band_pair_probe.py \
    --npz_path data_artifacts/passt_embeddings_fma_small_t64.npz \
    --label_key genres \
    --transform_type fft \
    --band_width 4 \
    --backend gpu \
    --gpu_device cuda \
    --gpu_epochs 400 \
    --gpu_epochs_triplet 200 \
    --gpu_lr 0.05 \
    --gpu_weight_decay 1e-4 \
    --random_state 42 \
    --pair_chunk_size 16 \
    --results_dir results/3rd_exp/passt_fma_small_band_pairs \
    --prefix passt_fma_small \
    "$@" \
    2>&1 | tee "$LOG"

echo "[done] log: $LOG"
echo "[done] results: results/3rd_exp/passt_fma_small_band_pairs/"
ls -la results/3rd_exp/passt_fma_small_band_pairs/
