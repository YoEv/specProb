# Setup on Gatech Server

This guide follows the recommended split:

- Code -> GitHub
- Dataset -> direct transfer (`rsync` or `scp`)
- Checkpoints (if any) -> direct transfer (`rsync` or `scp`)

## 1) Push code to GitHub from local server (`192.168.1.14`)

In `/home/evev/specProb`:

```bash
git status
git add .
git commit -m "Prepare repo for gatech server setup"
git remote -v
# If remote is missing, add one:
# git remote add origin <your-github-repo-url>
git push -u origin master
```

## 2) Clone code on Gatech server

```bash
git clone <your-github-repo-url>
cd specProb
```

## 3) Transfer dataset (and checkpoints if available)

Use `rsync` if possible (recommended: resumable and faster for large files).

### Option A: rsync (recommended)

Run these on `192.168.1.14` (target: `xli3252@login-ice.pace.gatech.edu:~/specProb`):

```bash
rsync -avhP /home/evev/specProb/data/ xli3252@login-ice.pace.gatech.edu:~/specProb/data/
rsync -avhP /home/evev/specProb/data_artifacts/ xli3252@login-ice.pace.gatech.edu:~/specProb/data_artifacts/
rsync -avhP /home/evev/specProb/dataset/ xli3252@login-ice.pace.gatech.edu:~/specProb/dataset/
rsync -avhP /home/evev/specProb/guitarSet/ xli3252@login-ice.pace.gatech.edu:~/specProb/guitarSet/
# Optional: only if checkpoints directory exists
# rsync -avhP /home/evev/specProb/checkpoints/ xli3252@login-ice.pace.gatech.edu:~/specProb/checkpoints/
```

### Option B: scp

If you prefer `scp`:

```bash
scp -r /home/evev/specProb/data xli3252@login-ice.pace.gatech.edu:~/specProb/
scp -r /home/evev/specProb/data_artifacts xli3252@login-ice.pace.gatech.edu:~/specProb/
scp -r /home/evev/specProb/dataset xli3252@login-ice.pace.gatech.edu:~/specProb/
scp -r /home/evev/specProb/guitarSet xli3252@login-ice.pace.gatech.edu:~/specProb/
# Optional: only if checkpoints directory exists
# scp -r /home/evev/specProb/checkpoints xli3252@login-ice.pace.gatech.edu:~/specProb/
```

### Option C: local relay (when direct SSH is blocked)

Use your laptop as a bridge with local folder `/Users/evev/Documents/highFreq/specProb_transfer`.

Run these on your laptop:

```bash
mkdir -p /Users/evev/Documents/highFreq/specProb_transfer

# Step 1: pull from your own server (192.168.1.14) to laptop
scp -r evev@192.168.1.14:/home/evev/specProb/data /Users/evev/Documents/highFreq/specProb_transfer/
scp -r evev@192.168.1.14:/home/evev/specProb/data_artifacts /Users/evev/Documents/highFreq/specProb_transfer/
scp -r evev@192.168.1.14:/home/evev/specProb/dataset /Users/evev/Documents/highFreq/specProb_transfer/
scp -r evev@192.168.1.14:/home/evev/specProb/guitarSet /Users/evev/Documents/highFreq/specProb_transfer/
# Optional: only if checkpoints directory exists
# scp -r evev@192.168.1.14:/home/evev/specProb/checkpoints /Users/evev/Documents/highFreq/specProb_transfer/

# Step 2: push from laptop to Gatech
ssh xli3252@login-ice.pace.gatech.edu "mkdir -p ~/specProb"
scp -r /Users/evev/Documents/highFreq/specProb_transfer/data xli3252@login-ice.pace.gatech.edu:~/specProb/
scp -r /Users/evev/Documents/highFreq/specProb_transfer/data_artifacts xli3252@login-ice.pace.gatech.edu:~/specProb/
scp -r /Users/evev/Documents/highFreq/specProb_transfer/dataset xli3252@login-ice.pace.gatech.edu:~/specProb/
scp -r /Users/evev/Documents/highFreq/specProb_transfer/guitarSet xli3252@login-ice.pace.gatech.edu:~/specProb/
# Optional: only if checkpoints directory exists
# scp -r /Users/evev/Documents/highFreq/specProb_transfer/checkpoints xli3252@login-ice.pace.gatech.edu:~/specProb/
```

## 4) Verify on Gatech server

```bash
cd /home/<gatech_user>/specProb
ls
du -sh data data_artifacts dataset guitarSet
# Optional: run only if checkpoints exists
# du -sh checkpoints
```

## 5) Export current base environment (on `192.168.1.14`)

Because your project currently runs in base/system Python, first export a reproducible dependency snapshot.

Run in `/home/evev/specProb`:

```bash
python -V > py_version.txt
python -m pip --version > pip_version.txt
python -m pip freeze > requirements.lock.txt
```

Optional but useful for debugging package mismatch:

```bash
python -m pip list --format=freeze > requirements.list.txt
python -m pip check > pip_check.txt || true
```

Then commit these text files to the repo (or copy them to Gatech manually).

## 6) Recreate environment with conda on Gatech server

Since you want conda on Gatech, create a dedicated conda env and install from lock file:

```bash
cd /home/<gatech_user>/specProb
conda create -n specprob python=3.10 -y
conda activate specprob
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.lock.txt
```

If your code depends on `spectral-probing/requirements.txt`, install it too:

```bash
python -m pip install -r spectral-probing/requirements.txt
```

Optional: after install succeeds, export a conda env file for future reuse:

```bash
conda env export -n specprob > environment.gatech.yml
```

## 7) Validate the conda environment

```bash
cd /home/<gatech_user>/specProb
conda activate specprob
python -V
python -m pip check
python -m pip freeze | wc -l
conda list | wc -l
```

For GPU projects, also verify Torch/CUDA:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
PY
```

## 8) GPU jobs and PaSST FMA embedding extraction (PACE)

This section records how to request GPU time on PACE, run `scripts/extract_passt_fma_embeddings.py` with **sharding**, and avoid common pitfalls (shared filesystem contention, wrong shard index, corrupt checkpoints). Adjust account, partition, QoS, and paths to match your project layout (home vs `/storage/ice1/...` on compute nodes).

### 8.1 Request an interactive GPU allocation (`srun`)

From a PACE login node, start an interactive shell on a GPU node (example flags; yours may differ):

```bash
srun --account=musi --partition=pace-gpu --qos=pace-ice \
  --gres=gpu:1 --time=8:00:00 --mem=32G --cpus-per-task=16 --pty bash
```

- Use `**--gres=gpu:1**` when you plan to run **one** Python extraction process per allocation (recommended for two shards in parallel: submit **two** such jobs so they can land on **different** nodes).
- Using `**--gres=gpu:2`** and then starting **two** Python processes **in the same shell** keeps both on the **same node**. They then compete for the same link to shared storage and the same CPUs; throughput often drops sharply compared to running one job at a time.

To see your jobs: `squeue -u $USER`. To cancel from another session: `scancel <JOBID>`. To release the interactive allocation, run `exit` or press `Ctrl+D` in that shell.

### 8.2 Environment and repo directory

On the GPU node:

```bash
module load anaconda3
conda activate specprob
cd Desktop/specProb
```

Ensure there is **no trailing space after `\`** when continuing long shell lines; otherwise the next line is treated as a new command.

### 8.3 Sharded extraction (two disjoint halves of the dataset)

Sharding is **deterministic** in code: after the same metadata and sampling options, the track list is split with stride `num_shards`. Shard `k` takes rows `k, k+num_shards, k+2*num_shards, ...`.

- `**--num_shards 2 --shard_index 0`**: even indices in that list.
- `**--num_shards 2 --shard_index 1**`: odd indices.

**Critical:** For two parallel jobs, use the **same** `--metadata_path`, `--max_tracks`, `--max_per_genre`, and other sampling-related flags, and the **same** `--num_shards`. Only `**--shard_index`**, `**--output_file**`, and (per job) `**CUDA_VISIBLE_DEVICES**` / `--device` should differ. Otherwise the two jobs are not guaranteed to partition the same logical list correctly.

**Do not** point two jobs at the **same** `--output_file` or the same `*.checkpoint.npz`; they will corrupt each other’s NPZ.

Example: **shard 0** (single GPU visible as `cuda:0`):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/extract_passt_fma_embeddings.py \
  --metadata_path data_artifacts/fma_medium_metadata.csv \
  --output_file data_artifacts/passt_fma_medium_shard0.npz \
  --max_tracks -1 \
  --max_per_genre 1000000 \
  --num_shards 2 \
  --shard_index 0 \
  --device cuda:0 \
  --wav_mirror_root data/fma_medium_wav/fma_medium \
  --wav_strip_prefix data/fma_medium/fma_medium \
  --batch_size 8 \
  --prefetch 8 \
  --checkpoint_every 200 \
  --auto_resume
```

Example: **shard 1** (separate `srun` / separate node recommended):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/extract_passt_fma_embeddings.py \
  --metadata_path data_artifacts/fma_medium_metadata.csv \
  --output_file data_artifacts/passt_fma_medium_shard1.npz \
  --max_tracks -1 \
  --max_per_genre 1000000 \
  --num_shards 2 \
  --shard_index 1 \
  --device cuda:0 \
  --wav_mirror_root data/fma_medium_wav/fma_medium \
  --wav_strip_prefix data/fma_medium/fma_medium \
  --batch_size 8 \
  --prefetch 8 \
  --checkpoint_every 200 \
  --auto_resume
```

If you ever saved `**passt_fma_medium_shard1.npz**` while still using `**--shard_index 0**`, that file can contain the **wrong** tracks (shard 0’s slice, not shard 1’s). Remove the mistaken outputs and rerun shard 1 with `--shard_index 1`:

```bash
rm -f data_artifacts/passt_fma_medium_shard1.npz \
      data_artifacts/passt_fma_medium_shard1.npz.checkpoint.npz
```

### 8.4 Resume and checkpoints

- `**--auto_resume**`: if `--output_file` already exists, it is used like `--resume_from_npz` to skip paths already present in that NPZ.
- **Checkpoint**: with `--checkpoint_every N`, progress is also written to `<output_file>.checkpoint.npz` (or `--checkpoint_path`) every **N newly extracted** tracks, so a killed job loses at most about **N** tracks of work. On startup, a readable checkpoint is merged into the resume map.
- **Path keys (important)**: resume/checkpoint matching uses a stable key for each `audio_path`. Relative paths are resolved against `**--audio_path_base`**, which defaults to the **repository root** when metadata lives under `data_artifacts/` (parent of that folder). That way checkpoint resume does **not** depend on your shell’s current working directory. Override with `--audio_path_base /path/to/specProb` if your layout differs.
- If a checkpoint is **empty, truncated, or corrupt** (e.g. interrupted write, disk quota full, or two processes writing the same file), the script may skip it and continue (you may re-extract some tracks).
- **Leftover `data_artifacts/tmp*.npz.tmp.npz` files** are failed atomic checkpoint writes; safe to remove to free space:  
`rm -f data_artifacts/tmp*.npz.tmp.npz`

### 8.5 Merge shards into one NPZ

After **all shards** finish successfully (example for `num_shards=8`):

```bash
python scripts/merge_fma_embedding_npz_shards.py \
  --inputs data_artifacts/passt_fma_medium_shard0.npz \
           data_artifacts/passt_fma_medium_shard1.npz \
           data_artifacts/passt_fma_medium_shard2.npz \
           data_artifacts/passt_fma_medium_shard3.npz \
           data_artifacts/passt_fma_medium_shard4.npz \
           data_artifacts/passt_fma_medium_shard5.npz \
           data_artifacts/passt_fma_medium_shard6.npz \
           data_artifacts/passt_fma_medium_shard7.npz \
  --output data_artifacts/passt_embeddings_fma_medium_t64.npz
```

Order of `--inputs` should match shard indices if you care about a stable ordering (shard0 ... shard7).

If merge is OOM-killed, you can run spectral probing directly on shard files (no merged NPZ needed):

```bash
python scripts/run_passt_spectral_profiles.py \
  --npz_paths data_artifacts/passt_fma_medium_shard0.npz \
              data_artifacts/passt_fma_medium_shard1.npz \
              data_artifacts/passt_fma_medium_shard2.npz \
              data_artifacts/passt_fma_medium_shard3.npz \
              data_artifacts/passt_fma_medium_shard4.npz \
              data_artifacts/passt_fma_medium_shard5.npz \
              data_artifacts/passt_fma_medium_shard6.npz \
              data_artifacts/passt_fma_medium_shard7.npz \
  --label_key genres \
  --results_dir results/passt_spectral_medium \
  --prefix passt_fma_medium
```

### 8.6 If two parallel jobs are still slow

Raising `**--cpus-per-task**` helps CPU-side decode/prefetch but often does **not** fix **two jobs on the same node** both reading from **shared network storage**. Mitigations: use **two separate `srun` jobs** with `**gpu:1`** (different nodes when the scheduler allows), **stage audio to node-local disk** first, or run shards **sequentially** on one GPU if wall-clock time is lower due to less I/O contention.

## Notes

- Keep large data/checkpoints out of GitHub.
- If a checkpoint must be versioned with code, consider Git LFS for specific files only.
- Keep secrets in `.env` and never commit them.

