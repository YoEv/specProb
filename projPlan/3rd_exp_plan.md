# 3rd Experiment Plan (3rd_exp_plan)

## Goal

在现有 spectral probing 基础上，从「per-band solo probe」扩展到两条主线：

1. **Band combination (feature selection) 分析**：不止看单 band 的表现，而是系统性地枚举 1+1、1+2 组合，量化「band 之间谁冗余、谁互补」，并给出一个基于 solo + cumulative 边际增益的 band ranking。
2. **补一个高频敏感的 downstream**：在现有 PaSST × FMA-small (genre) 之外，加 **NSynth 乐器家族 (11 类)** —— 这是 audio representation learning 社区最常用的线性探针 benchmark 之一，类间高频内容差异大（mallet 瞬态 / reed 高次谐波 / percussive 宽带噪声），是对 "高频 band 到底管不管用" 的明确压力测试。

所有实验 encoder 固定为 **PaSST**，probe 固定为多分类 GPU 路径（`run_generic_fft_band_probe.py --label_mode multiclass` 与新写的 `run_band_pair_probe.py` 共享）。

本轮**不**做：ASAP composer、BEATs 抽取、frame-level beat onset detection（与当前 FFT-on-T 的 spectral probing 框架不兼容，作为 future work；具体原因见 §6）、FMA-medium sharded streaming（延后）。

---

## 1) Band combination analysis: 1+1, 1+2, ranking, redundancy

### 约定符号

- K = `n_coeffs // band_width`，即 band 数量。fma_small t=64 + band_width=4 → K=8，后面以此为参照。
- 每个 band `B_i` 对应 FFT 系数切片 `[i*band_width : (i+1)*band_width]`；展平后每个样本对该 band 的特征维度是 `F * band_width`。

### 1.1 所有 "ordered" 1+1 pair probes

- 枚举所有 `{i, j}` with `i < j`，共 `C(K, 2) = K*(K-1)/2` 对（K=8 时 = 28 对），顺序 `(0,1), (0,2), …, (K-2, K-1)` 确定。
- 每对的 probe 输入：`[B_i ∪ B_j]`（沿频率轴 concat），多分类线性 head，train/test split 与 seed 与现有 multiclass 流程一致。
- GPU batched 执行：feature 维度统一 `F * 2 * band_width`，直接复用 `_run_probe_gpu_batched_same_dim_multiclass`。

### 1.2 所有 "ordered" 1+2 probes

- 枚举 `(i, [j, j+1])` 组合：
  - `i ∈ {0..K-1}`，`j ∈ {0..K-2}`，且 `j != i` 且 `j+1 != i`（避免重复 band）；
  - 顺序固定：外层 `i` 升序，内层 `j` 升序。
- 每个组合的 probe 输入：`[B_i ∪ B_j ∪ B_{j+1}]`，feature 维度 `F * 3 * band_width`。
- K=8 时组合数约 `8 × (K-2) − (去掉 j、j+1 覆盖 i 的情况) ≈ 40` 级别，依然一把 batched 可完成。

### 1.3 Band ranking

只做最轻量的两种 ranking，都不引入新训练：

- `solo_acc[i]`：`band_specific` 单 band 的 overall accuracy（现成）。
- `prefix_delta[i] = cumulative_acc[i] − cumulative_acc[i-1]`（`cumulative_acc[-1] ≡ 0`）：cumulative prefix 在加入 band `i` 时的边际收益（现成，重用 `cumulative_per_band`）。

最终：

- `ranking_by_solo`：按 `solo_acc` 降序排 band。
- `ranking_by_prefix_delta`：按 `prefix_delta` 降序排。
- 报告两者的 top-3 / bottom-3 和 rank-correlation（Spearman）。

> 注：这不是严格 Shapley，只是 "solo + 边际" 两个视角。若两种 ranking 相关性很高，说明单 band 的 solo 已经足够代理；若差别很大，说明某些 band 本身弱但在合集里关键。

### 1.4 Redundancy / synergy (中文定义，便于报告)

> **Redundancy 指两个 band 提供的判别信息有多少是重复的。** 举例：某 task 里 band A 单独训 probe 拿 0.4 overall acc，band B 单独训拿 0.42；把两者合并训一个 probe（输入维度 2×band_width×F），如果合并版拿 0.43，几乎等于两个里较好的那一个——说明 B 的信息已经被 A 表达过了，B 对 A 来说是 **冗余 (redundant)** 的。反之如果合并版拿 0.55，远高于任一单独 band，说明它们在不同方向上互补（**synergy**）。

形式化（都从 §1.1 的 pair probe 的 `acc(i+j)` 得到）：

- `synergy(i, j) = acc(i + j) − max(acc(i), acc(j))`
  - 正数：合起来比任何一个单独都好，有协同。
  - 接近 0：合起来没帮助，两者之一是冗余。
  - 负数（罕见，通常是噪声）：合起来反而更差。
- `redundancy(i, j) = acc(i) + acc(j) − acc(i + j)`
  - 正数越大越冗余（两个 band 信息重合越多）。
  - 接近 0 表示近似独立。
  - 负数表示互补到超线性（罕见）。

### 1.5 Deliverables (per encoder × task)

- 脚本：`scripts/run_band_pair_probe.py`
  - 新脚本；复用 `scripts/run_generic_fft_band_probe.py` 里的 `_run_probe_gpu_multiclass` / `_run_probe_gpu_batched_same_dim_multiclass`。
  - 参数与 `run_generic_fft_band_probe.py` 保持风格一致：`--npz_path --label_key --band_width --backend gpu --gpu_device --gpu_epochs --gpu_lr --gpu_weight_decay --random_state --results_dir --prefix`。
  - 强制 `--label_mode multiclass`（GPU-only，与 4th_sanity_check 决策一致）。
  - seed 固定 `--random_state 42`，单次运行（与现有 multiclass probe 保持一致）。
- 结果目录：`results/3rd_exp/<encoder>_<task>_band_pairs/`
  - `pair_1p1_matrix.json`：K×K `acc(i+j)` 矩阵（对角线填 `solo_acc[i]`）。
  - `synergy_1p1_matrix.png` + `redundancy_1p1_matrix.png`：heatmap。
  - `pair_1p2_summary.json`：所有 `(i, [j, j+1])` 的 acc + Δ over anchor + Δ over 2-block。
  - `band_ranking.json`：`solo` 和 `prefix_delta` 两个 ranking + Spearman。

---

## 2) 新 task (PaSST): NSynth 11-class instrument family classification

### 为什么选这个

- Audio representation learning 社区最常引用的线性探针 benchmark 之一（AudioMAE、BEATs、CLAP 论文都用）。
- **类别分布决定高频内容有明显区分度**：mallet 的瞬态尖峰、reed 的高次谐波、percussive 的宽带噪声、string 的泛音结构 —— 如果 "高频 band 不管用" 的直觉是错的，NSynth 上应该立刻看到高频 band 的 accuracy 凸起，比 FMA genre 的 "高频几乎没贡献" 结论反差更大，是明确的压力测试。
- 11 类单标签多分类，直接套当前 multiclass probe，不用改 head。
- 官方已分 train / valid / test；每 clip 4s、16 kHz、pitched + unpitched 皆有，信号类型覆盖全。

### Execution

- **数据**：NSynth (官方 magenta NSynth release; `nsynth-train / -valid / -test` jsonl + wav)。instrument_family 字段即 11 类标签。
- **Embedding 抽取**：新脚本 `scripts/extract_passt_nsynth.py`
  - 仿 `scripts/extract_passt_fma_embeddings.py` 结构（load wav → PaSST timestamp embedding → NPZ）。
  - 输入：官方 NSynth `wav/` 目录 + metadata jsonl；输出 `data_artifacts/passt_embeddings_nsynth_t<T>.npz`，包含 `embeddings (N, F, T)`、`instrument_family (N,)` (str)、`split (N,)` (str, 值 ∈ {train, valid, test})、`note_id (N,)` 可选。
  - 如果官方 train split 太大（~289K clip），本轮先用 `nsynth-valid`（12K）+ `nsynth-test`（4K）合并做 probe 训练集，保留 `nsynth-train` 作为后续可扩展。JSON 里记 `used_splits`。
- **Multiclass probe**：
  - `python scripts/run_generic_fft_band_probe.py --npz_path data_artifacts/passt_embeddings_nsynth_t<T>.npz --label_key instrument_family --label_mode multiclass --transform_type fft --band_width 4 --backend gpu --gpu_batch_bands --gpu_batch_cumulative --results_dir results/passt_nsynth_multiclass_fft --prefix passt_nsynth`
- **Band pair 分析**：跑 `scripts/run_band_pair_probe.py`（§1），输出到 `results/3rd_exp/passt_nsynth_band_pairs/`。

### Deliverables

- `data_artifacts/passt_embeddings_nsynth_t<T>.npz`
- `results/passt_nsynth_multiclass_fft/`
- `results/3rd_exp/passt_nsynth_band_pairs/`

### 验收条件

- `full_embedding` overall acc ≥ 0.6（11 类 chance ≈ 0.09；文献里线性探针 PaSST/AudioMAE/BEATs 在 NSynth-instrument-family 上都在 0.7~0.85 区间，0.6 是松阈值）。低于 0.6 要检查 embedding 是否抽对（例如是否误用了 PaSST 的 mel-scale feature 前段）。
- 单 band solo_acc 在至少一个高频 band（`band_idx ≥ K/2`）上**显著高于** chance（≥ 0.2），否则 "高频在 NSynth 上也不起作用"，是有意思的负面结论，需要额外截图保留。

---

## 3) Cross-task aggregation

两个 (encoder, task) 组合：PaSST × FMA-small、PaSST × NSynth。

- **Per-task ranking 对比**：两个 task 各自的 `ranking_by_solo` top-3 / bottom-3，以及 `ranking_by_prefix_delta` top-3 / bottom-3。
- **Per-task redundancy pattern**：两张 K×K redundancy heatmap 并排画，观察：
  - 是否存在 "universal redundant pair"（在两个 task 都冗余，说明 band 结构本身的重合）。
  - 是否存在 "task-specific synergy"（某对 band 在 NSynth 上互补，在 FMA-small 上冗余）—— 这是本轮**核心论据**：band 偏好随 task 性质（是不是高频敏感）变化。
- **AUTO 对比**：每 task 的 `cumulative_auto` / `trained_auto` 与 full_embedding 的差距，看 "只用最有用的前几个 band" 对不同 task 的压缩率。

### Deliverables

- `results/3rd_exp/cross_task_band_comparison.md`：文字 + 表格。
- `results/3rd_exp/cross_task_redundancy_grid.png`：两张 K×K heatmap 合成一张。
- `results/3rd_exp/cross_task_ranking_table.csv`：`task, band_idx, solo_acc, prefix_delta, rank_by_solo, rank_by_prefix_delta`。

---

## 4) Execution milestones

- **M1 — pair probe 脚手架**（先不碰新数据集）
  - 写 `scripts/run_band_pair_probe.py`。
  - 先在 `data_artifacts/passt_embeddings_fma_small_t64.npz`（现有 800 条那份）上跑 1+1 + 1+2（`random_state=42`），产出 pair matrix / heatmap / ranking。
  - 数值验证：对角线 `acc(i+i)` 不存在（跳过），但 off-diagonal `acc(i+j)` 应该始终 ≥ `max(solo_acc[i], solo_acc[j]) − noise`。若大量 pair 比 solo 还差，说明训练 epoch/lr 设置不稳，需要调。

- **M2 — NSynth**
  - 写 `scripts/extract_passt_nsynth.py` → 抽 embedding（先用 valid+test subset ~16K clip，后续按需扩到 train）。
  - `run_generic_fft_band_probe.py --label_mode multiclass` 全跑一次 → 存 `results/passt_nsynth_multiclass_fft/`。
  - `run_band_pair_probe.py` 跑 §1 → 存 `results/3rd_exp/passt_nsynth_band_pairs/`。
  - 验收：`full_acc ≥ 0.6`，且至少一个高频 band solo_acc ≥ 0.2。

- **M3 — cross-task aggregation**
  - 写 `scripts/aggregate_band_pair_results.py`（新）吃两个 `results/3rd_exp/*_band_pairs/` 目录，产出 §3 的 deliverables。

顺序可并行的是 M1（只需现有 NPZ）和 M2 前半（NSynth embedding 抽取在另一个 GPU 进程上跑），后半 M2 依赖 M1 的 `run_band_pair_probe.py`。

---

## 5) 风险 & 回退

- **NSynth 下载体量大**（原始 wav 数百 GB）：本轮只下 `nsynth-valid` + `nsynth-test`（≈16K clip，几 GB），若嫌少再扩 `nsynth-train` 或其子集。
- **PaSST 抽 NSynth 4s clip 时的 window 选择**：NSynth clip 是 4s，PaSST 默认输入 10s。沿用 `extract_passt_fma_embeddings.py` 的 padding/truncation 逻辑（pad_mode='repeat' or 'zero'，和 FMA 统一），脚本里显式记录决策到 NPZ metadata。
- **1+2 组合爆炸**：若后续任务 K 变大（比如 t=128 → K=16 → 1+2 组合 >200），给 `run_band_pair_probe.py` 加 `--max_pairs_1p2 N` 限制，默认随机 anchor 子采样。
- **multiclass probe 在小样本任务上方差大**：本轮按约定只跑 `random_state=42` 单次；若某 task 结果明显受单次 split 扰动（例如 heatmap 出现 acc > 1 或明显非物理的 cell），再补跑其它 seed。

---

## 6) 本轮**不**做的任务（与理由）

- **Beat onset detection**：
  - 当前 pipeline 在时间轴 T 上做 FFT，`n_coeffs` 表达的是 "整段 clip 里某个频率的 feature 波动强度"，已经和具体时间点脱钩。
  - Beat onset 是 frame-level 二分类（每 10 ms 一帧是/否 onset），需要保留时间轴对齐，和 FFT-on-T 的特征表示本质冲突。
  - 折衷方案（Option B1 frame-level probe、Option B2 tempo 桶分类）都需要独立另起一套 pipeline，且无法纳入 §1 的 pair 分析，本轮放入 future work。
- **ASAP composer**：已在现有 results 中有结果，本轮不重跑、不改目录。
- **BEATs encoder**：抽取与 multiclass probe 路径未在 4th_sanity_check 中落地，避免同时变动 encoder 和 task 两个维度。
- **FMA-medium sharded streaming multiclass**：延续到后续 plan；本轮 multiclass GPU 路径的 800 条 fma_small 验证已经足够证明 batched 路径数值一致。

---

## 7) 与既有计划的关系

- 复用 `4th_sanity_check_plan.md` 里已落地的 multiclass probe GPU 路径（`src/visualization/spectral_profile.py` 的 `learned_weight_profile_multiclass` / `learned_weight_profile_stats_multiclass`；`scripts/run_generic_fft_band_probe.py` 的 `_run_probe_gpu_multiclass` / `_run_probe_gpu_batched_same_dim_multiclass` / `_run_probe_gpu_batched_prefix_multiclass`）。
- 不与 `2nd_exp_plan.md` 冲突：slope 聚合 / feature-std / attention-head 那些任务依旧是 future work；§1 的 ranking / redundancy 分析可看作是 2nd_exp_plan §1 slope 分析的补充视角（slope 看全局趋势，pair 看局部交互）。

---

## 8) Immediate next actions

1. 写 `scripts/run_band_pair_probe.py`（M1）。
2. 在现有 `data_artifacts/passt_embeddings_fma_small_t64.npz` 上跑 M1 smoke + full（`random_state=42`），确认 pair / 1+2 两个矩阵数值合理。
3. 写 `scripts/extract_passt_nsynth.py`（M2 前半）。
4. 等 NSynth NPZ 就位后按 M2 后半 / M3 顺序推进。
