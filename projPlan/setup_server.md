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

## Notes

- Keep large data/checkpoints out of GitHub.
- If a checkpoint must be versioned with code, consider Git LFS for specific files only.
- Keep secrets in `.env` and never commit them.
