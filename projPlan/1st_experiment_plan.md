# 1st Experiment Plan: FMA Embedding Naming + PaSST Medium Extraction

## 0) 当前目标

本轮先做三件事：

1. 统一 `data_artifacts` 中 FMA small embedding 的命名。
2. 明确 `fma_medium` 的 metadata 生成方式。
3. 在 server 端运行 PaSST，对 `fma_medium` 抽取 embedding，并使用新的规范命名。

---

## 1) 命名规范

### 1.1 FMA small 现有文件重命名

以下三个文件统一显式带上 `fma_small`：

- `data_artifacts/beats_embeddings_t64.npz` -> `data_artifacts/beats_embeddings_fma_small_t64.npz`
- `data_artifacts/clap_embeddings_t64.npz` -> `data_artifacts/clap_embeddings_fma_small_t64.npz`
- `data_artifacts/passt_embeddings_t64.npz` -> `data_artifacts/passt_embeddings_fma_small_t64.npz`

对应命令：

```bash
mv data_artifacts/beats_embeddings_t64.npz data_artifacts/beats_embeddings_fma_small_t64.npz
mv data_artifacts/clap_embeddings_t64.npz data_artifacts/clap_embeddings_fma_small_t64.npz
mv data_artifacts/passt_embeddings_t64.npz data_artifacts/passt_embeddings_fma_small_t64.npz
```

### 1.2 后续命名约定

后续统一使用：

- `data_artifacts/<model>_embeddings_fma_small_t64.npz`
- `data_artifacts/<model>_embeddings_fma_medium_t64.npz`
- `data_artifacts/<model>_embeddings_asap_t32.npz`

示例：

- `data_artifacts/passt_embeddings_fma_small_t64.npz`
- `data_artifacts/passt_embeddings_fma_medium_t64.npz`
- `data_artifacts/clap_embeddings_fma_small_t64.npz`
- `data_artifacts/beats_embeddings_fma_small_t64.npz`

---

## 2) 关键前提

- 当前 `data_artifacts/fma_metadata.csv` 实际对应的是 `fma_small`，其 `audio_path` 指向 `data/fma_small/...`。
- 不能直接拿当前 `fma_metadata.csv` 去跑 `fma_medium`，否则会仍然读到 small 数据。
- 在跑 `fma_medium` 的 PaSST 抽取前，必须先生成一个 `fma_medium` 专用 metadata 文件。

---

## 3) Server 端操作

### Step A — 登录并进入环境

```bash
module load anaconda3/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate specprob
cd /storage/ice1/2/2/xli3252/specProb
```

如需申请交互式 GPU：

```bash
srun --account=musi --partition=pace-gpu --qos=pace-ice --gres=gpu:2 --time=4:00:00 --mem=32G --cpus-per-task=4 --pty bash
module load anaconda3/2023.03
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate specprob
cd /storage/ice1/2/2/xli3252/specProb
```

### Step B — 先做 small 文件规范化重命名

```bash
mv data_artifacts/beats_embeddings_t64.npz data_artifacts/beats_embeddings_fma_small_t64.npz
mv data_artifacts/clap_embeddings_t64.npz data_artifacts/clap_embeddings_fma_small_t64.npz
mv data_artifacts/passt_embeddings_t64.npz data_artifacts/passt_embeddings_fma_small_t64.npz
```

### Step C — 生成 `fma_medium` 专用 metadata

当前仓库里的 `src/data_processing/fma_preparation.py` 默认只为 `fma_small` 生成 metadata，所以这里先临时在 server 端生成 `medium` 版本：

```bash
python - <<'PY'
import os
import pandas as pd

tracks_path = "data/fma_metadata/fma_metadata/tracks.csv"
audio_dir = "data/fma_medium"
output_path = "data_artifacts/fma_medium_metadata.csv"

tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])
medium_subset = tracks[tracks[("set", "subset")] == "medium"]

rows = []
for track_id, row in medium_subset.iterrows():
    tid_str = f"{track_id:06d}"
    audio_path = os.path.join(audio_dir, tid_str[:3], f"{tid_str}.mp3")
    if os.path.exists(audio_path):
        genre = row[("track", "genre_top")]
        if pd.notna(genre):
            rows.append(
                {
                    "track_id": track_id,
                    "genre": genre,
                    "audio_path": audio_path,
                }
            )

pd.DataFrame(rows).to_csv(output_path, index=False)
print(f"saved {len(rows)} rows to {output_path}")
PY
```

### Step D — 用 PaSST 抽取 `fma_medium`

现有 `scripts/extract_passt_fma_embeddings.py` 默认写出 `data_artifacts/passt_embeddings_t64.npz`，而且默认读取 `data_artifacts/fma_metadata.csv`。  
因此在不改脚本的前提下，server 端建议用一次性命令运行，并显式指定 `fma_medium` metadata 与输出文件名。

推荐命令：

```bash
python - <<'PY'
import os
import numpy as np
import pandas as pd
import torch

from scripts.extract_passt_fma_embeddings import extract_passt_fma_embeddings

metadata_path = "data_artifacts/fma_medium_metadata.csv"
output_path = "data_artifacts/passt_embeddings_fma_medium_t64.npz"

metadata = pd.read_csv(metadata_path)
device = "cuda" if torch.cuda.is_available() else "cpu"

embeddings, genres, file_paths = extract_passt_fma_embeddings(
    metadata_df=metadata,
    device=device,
    max_tracks=None,
    max_per_genre=1000000,
)

np.savez_compressed(
    output_path,
    embeddings=embeddings,
    genres=np.asarray(genres),
    file_paths=np.asarray(file_paths),
)

print(f"saved embeddings to {output_path}")
print(f"shape = {embeddings.shape}")
PY
```

说明：

- `max_tracks=None` 表示不再限制总条数。
- `max_per_genre=1000000` 的目的只是关闭原脚本里的每类 100 条上限。
- 最终输出名必须使用：`data_artifacts/passt_embeddings_fma_medium_t64.npz`。

### Step E — 结果核验

```bash
ls -lh data_artifacts/passt_embeddings_fma_medium_t64.npz
python - <<'PY'
import numpy as np
d = np.load("data_artifacts/passt_embeddings_fma_medium_t64.npz", allow_pickle=True)
print(d["embeddings"].shape)
print(len(d["genres"]), len(d["file_paths"]))
print(d["file_paths"][0])
PY
```

---

## 4) 本轮交付物

本轮完成后，应至少具备：

1. `data_artifacts/beats_embeddings_fma_small_t64.npz`
2. `data_artifacts/clap_embeddings_fma_small_t64.npz`
3. `data_artifacts/passt_embeddings_fma_small_t64.npz`
4. `data_artifacts/fma_medium_metadata.csv`
5. `data_artifacts/passt_embeddings_fma_medium_t64.npz`

---

## 5) 验收标准

通过标准：

1. small 的三份 embedding 已按新命名规范重命名。
2. `fma_medium_metadata.csv` 生成成功，且 `audio_path` 指向 `data/fma_medium/...`。
3. `passt_embeddings_fma_medium_t64.npz` 成功生成并可被 `np.load(...)` 正常读取。
4. 后续所有 command 一律使用带数据集名的文件名，不再使用歧义命名。
