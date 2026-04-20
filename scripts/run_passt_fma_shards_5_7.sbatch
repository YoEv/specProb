#!/bin/bash
#SBATCH --job-name=passt_fma_s5_s7
#SBATCH --account=musi
#SBATCH --partition=pace-gpu
#SBATCH --qos=pace-ice
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate specprob
cd "$HOME/Desktop/specProb"

CUDA_VISIBLE_DEVICES=0 python scripts/extract_passt_fma_embeddings.py \
  --metadata_path data_artifacts/fma_medium_metadata.csv \
  --output_file data_artifacts/passt_fma_medium_shard5.npz \
  --max_tracks -1 \
  --max_per_genre 1000000 \
  --num_shards 8 \
  --shard_index 5 \
  --device cuda:0 \
  --wav_mirror_root data/fma_medium_wav/fma_medium \
  --wav_strip_prefix data/fma_medium/fma_medium \
  --batch_size 8 \
  --prefetch 8 \
  --checkpoint_every 200 \
  --auto_resume

CUDA_VISIBLE_DEVICES=0 python scripts/extract_passt_fma_embeddings.py \
  --metadata_path data_artifacts/fma_medium_metadata.csv \
  --output_file data_artifacts/passt_fma_medium_shard6.npz \
  --max_tracks -1 \
  --max_per_genre 1000000 \
  --num_shards 8 \
  --shard_index 6 \
  --device cuda:0 \
  --wav_mirror_root data/fma_medium_wav/fma_medium \
  --wav_strip_prefix data/fma_medium/fma_medium \
  --batch_size 8 \
  --prefetch 8 \
  --checkpoint_every 200 \
  --auto_resume

CUDA_VISIBLE_DEVICES=0 python scripts/extract_passt_fma_embeddings.py \
  --metadata_path data_artifacts/fma_medium_metadata.csv \
  --output_file data_artifacts/passt_fma_medium_shard7.npz \
  --max_tracks -1 \
  --max_per_genre 1000000 \
  --num_shards 8 \
  --shard_index 7 \
  --device cuda:0 \
  --wav_mirror_root data/fma_medium_wav/fma_medium \
  --wav_strip_prefix data/fma_medium/fma_medium \
  --batch_size 8 \
  --prefetch 8 \
  --checkpoint_every 200 \
  --auto_resume
