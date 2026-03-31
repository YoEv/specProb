## 二次频谱分析与实验迭代计划 (2nd_analysis_plan)

本计划围绕当前 DCT/FFT 频谱分析中出现的 **“三峰结构”异常**，以及频谱可视化粒度不足等问题，设计新一轮的诊断与扩展实验。目标是：

- **确认 3 个 peak 的真实成因（算法 / 数据 / embedding / 可视化方式）。**
- **在高维空间上更细致地理解 768 维 embedding 的频谱结构，而不是只看平均后的 1 条曲线。**
- **消除可能的 FFT 实现与 windowing artifact。**
- **在 genre 之外再引入一个音乐分类任务，检验频域 probe 方法的泛化性。**

---

## 一、现状检测 (Current Status)

### 1.1 代码库快照

| 模块 | 路径 | 职责 | 备注 |
|------|------|------|------|
| 数据加载 | `src/data_processing/loader.py` | `load_data(path)` → (X, y_str)，X 为 (B, F, T) | 与部分 script 内联的 load 逻辑不一致 |
| 频谱变换 | `src/analysis/spectral.py` | `apply_transform(..., transform_type, axis)`，DCT/FFT；`get_raw_band_features` 按 band 取系数 | **无 window**；已删除 get_spectral_bands / get_band_energy（不做 energy、不调 band） |
| 探针 | `src/training/probes.py` | `run_probe`, `run_probe_with_predictions` | 二分类/多分类通用 |
| 探针（重复） | `src/utils.py` | `run_probe(X_flat, y)` | 仅返回 accuracy，无 model/le |
| 探针（重复） | `src/analysis/deeper_analysis.py` | `run_probe`, `load_and_prepare_data` | 与 utils 逻辑重复 |
| 指标 | `src/analysis/metrics.py` | `compute_confusion` 等 | 被 confusion 脚本使用 |
| 可视化 | `src/visualization/plotting.py` | `create_and_save_plot`（Probe Performance + Spectral Profile） | 单 genre 双面板图 |
| 可视化 | `src/visualization/confusion.py` | `plot_confusion_matrix`, `plot_confusion_grid` | 混淆矩阵热力图 |
| 脚本 | `scripts/run_single_genre_analysis.py` | 单 genre 全流程：load → transform → probe → 存 JSON + 画图 | 自带 `load_data`/`run_probe`，与 src 不完全一致 |
| 脚本 | `scripts/run_fft_analysis.py` | 遍历所有 genre，写 `results/fft_genre_specific_analysis/` | **从 run_single_genre_analysis 导入**，未用 src 统一入口 |
| 脚本 | `scripts/run_probe_confusion_analysis.py` | 多分类 + 分 band 混淆矩阵 | 已用 `src.data_processing.loader`, `src.analysis.spectral`, `src.training.probes` |

### 1.2 数据与形状约定（统一、不可改）

- **NPZ 源**: `data_artifacts/clap_embeddings_t64.npz` → `embeddings`, `genres`, `track_ids`。
- **embedding 原始形状**: (B, 768, 2, 64)。两段 10s 的帧在时间维上 **concat** 成 64，即 2×32→64。
- **唯一合法的 reshape**（全项目一致）:
  - `X` 从 (B, 768, 2, 64) 变为 **(B, 1536, 64)**：即 `X.reshape(B, 768*2, 64)`，F=1536，T=64。
  - `loader.load_data` 必须按此方式返回 (B, 1536, 64)；所有脚本、分析都基于此形状，不再出现 (B, 1536, 32) 或其它 T。
- **FFT 轴**: `axis=2`，长度 **T=64** → `np.fft.rfft(..., axis=2)` 得到 **n_coeffs=33**（rfft(64)=33）。与 `N_COEFFS=33` 一致。

### 1.2.1 Embedding 层与 AUTO 含义（文档约定）

- **Embedding 来自哪一层**  
  当前使用的表征来自 **CLAP 音频编码器（HTSAT）的最后一层**：`model.get_audio_features(**inputs).last_hidden_state`。即对每段音频取 **last hidden state**（顶层 Transformer 输出），无池化；两段 10s 在时间维 concat 后得到 (B, 768, 2, 64)，reshape 为 (B, 1536, 64)。见 `src/features/extraction.py`。

- **AUTO 是什么**  
  图中的 **AUTO** 不是“由 probe 学出来的最优 band 组合”，而是：  
  对 **按固定顺序拼接的累积 band**（B0 → B0+B1 → B0+B1+B2 → … → 全部）分别训练 probe，取这些准确率中的 **最大值**。  
  即 AUTO = max(cumulative_per_band)，其中 cumulative_per_band[k] = 用前 k+1 个 band 的系数拼在一起训练的 probe 的准确率。  
  因此 AUTO 表示“按低频到高频依次加入 band 时，能达到的最好准确率”，**不是**“模型自动选出的 band 子集或权重”。

### 1.3 已知问题与缺口

- **spectral.py**: **删除** `get_spectral_bands` 与 `get_band_energy`，不做 energy、不对 band 做任何调整；保留 `get_raw_band_features`（仅按 band 取系数，供 selection 等使用）。
- **run_probe 多处实现**: `utils.run_probe`、`probes.run_probe`、`run_single_genre_analysis.run_probe`、`deeper_analysis.run_probe` 等，接口不一（有的返回 model/le，有的仅 accuracy），**执行计划以 `src/training/probes.py` 为唯一探针入口**。
- **load_data 不统一**: 脚本层存在与 loader 不一致的 reshape；**执行前统一由 `src/data_processing/loader.py` 提供唯一 load_data，且必须输出 (B, 1536, 64)，与 1.2 约定一致**。
- **无 FFT window**: `apply_transform` 中 FFT 分支未加窗，需在本轮增加可选 window 参数并做 ablation。

### 1.4 已有结果目录（供复用与扩展）

- `results/fft_genre_specific_analysis/` — 各 genre 的 `spectral_summary_<Genre>_fft_metrics.json`（full/band/cumulative accuracy + spectral_profile 权重）。
- `results/confusion_analysis/` — 分 band 混淆矩阵图。
- `results/*.json` / `*.md` — 各类 baseline、band_selection、critical_band 等。

### 1.5 图示约定：论文 spectral probing vs 我们的两种 “spectral profile”

**参考论文里的三张图（spectral-probing/plot/）：**

| 脚本 | 画的是什么 | 对应我们 |
|------|------------|----------|
| **accuracies.py** | 每个 frequency band 上的 **accuracy**（条形图，x=band，y=accuracy） | 我们 genre 分析里的 **左图 Probe Performance**（ORIG, B0…B7, AUTO） |
| **decomposition.py** | 按 band 滤波后，**单条序列上各 token 的 embedding 激活**（多条线，x=token） | 我们目前没有等价图（是“单句分解”的可视化） |
| **filter.py** | **Weight per frequency**：每个频率 bin 的权重（条形图，y 轴标 "Weight"） | 对应我们 **learned weight** 的 spectral profile（见下） |

论文 Figure 3 的 “Spectral Profiles… weight per frequency” + “lower and upper bounds across languages” 就是 **weight**：**探针/模型学到的“每个频率有多重要”**，不是数据的平均幅度。

**我们有两类不同的 “spectral profile”：**

1. **Learned weight（和论文一致）**  
   - **出处**：`results/fft_genre_specific_analysis/spectral_summary_<Genre>_fft.png` 的**右图**，以及 JSON 里的 `spectral_profile`。  
   - **算法**：用**全谱**训练 logistic regression probe，取 `model.coef_`，按 (n_features, n_coeffs) reshape 后对 feature 维取平均得到每条频率的权重，再归一化到 [0,1]。  
   - **含义**：**哪个频率系数被 probe 用得最多** → 与论文的 “weight per frequency” 一致，是 **probe 的 spectral profile**。

2. **Mean magnitude（和论文不同）**  
   - **出处**：`results/fft_window_ablation/spectral_profile_by_window_<Genre>.png` 的**左图**。  
   - **算法**：对 FFT 后的幅度 `X_t` 在 sample 和 feature 维做 `np.mean(X_t, axis=(0,1))`，得到一条 33 维的**平均幅度曲线**。  
   - **含义**：**数据在频域上的平均能量分布**（随 window 类型变化），**不是** probe 的权重。  
   - **同一张图内**：该 PNG 现已包含两栏——左栏为 Mean FFT magnitude by window，右栏为 **Spectral profile (learned weight) by window**（每种 window 训练 probe 后的 coef_ 归一化曲线），便于与论文的 weight 对比。

---

## 二、代码管理原则 (Code Management)

### 2.1 单一数据源与调用链

- **数据加载**: 所有脚本统一使用 `src.data_processing.loader.load_data(EMBEDDINGS_PATH)`；如需不同 reshape（如保留 768 维做逐维 FFT），在 loader 中增加可选参数或新函数（如 `load_data_for_per_dim_analysis`），避免在脚本里重复 reshape。
- **变换**: 仅使用 `src.analysis.spectral.apply_transform`；本轮在 spectral 内增加 `window_type: str | None` 及 FFT 前加窗逻辑，不新增分散的 FFT 实现。
- **探针**: 二分类/多分类、仅要 accuracy 或要 model/预测，统一使用 `src.training.probes.run_probe` 或 `run_probe_with_predictions`；逐步弃用 `utils.run_probe` 及脚本内联的 `run_probe`，改为从 probes 导入。
- **指标与画图**: 继续使用 `src.analysis.metrics`、`src.visualization.plotting`、`src.visualization.confusion`；新增高维可视化（heatmap、per-sample 曲线）放在 `src.visualization/` 下新文件（见下）。

### 2.2 新增代码的归属

- **与“三峰排查”相关的数值/对照实验**（随机向量、单样本手算、归一化检查）→ 放在 `src/analysis/` 下，例如 `src/analysis/peak_artifact_checks.py`，仅负责生成中间数据与简单打印/保存，不替代现有 spectral 管线。
- **高维频谱可视化**（768×33 heatmap、cluster 平均谱、per-sample 曲线）→ `src/visualization/spectral_highdim.py`（或类似命名），由脚本调用并写入指定 results 子目录。
- **FFT window 与 n_coeffs 实验**→ 在 `src/analysis/spectral.py` 内扩展 `apply_transform`；ablation 脚本只改参数，不复制 FFT 代码。
- **新任务（如 Vocal vs Instrumental、古典作曲家分类）**→ 标签构造可放在 `src/data_processing/`（如从 metadata 生成标签）；同一套 probe + 频带逻辑复用，仅换 `y` 与输出子目录。

### 2.4 配置与可扩展性（config 约定）

- **统一配置源**：为所有“有选项”的组件（如 CLAP 抽取模式、FFT window 类型、`n_coeffs`、是否使用多层 embedding 等）提取出一个可靠的配置：
  - 推荐在 `configs/` 或 `src/config/` 下维护一个 Python/JSON/YAML 配置（例如 `spectral_experiment_config.py`），由脚本统一读取；
  - 常用字段包括：
    - `embedding_mode`：`'fma_main'` 或 `'asap_composer'`（决定是否使用 `(768, 2, 64)` 还是 `(768, 2, 32)`）；
    - `fft_window_type`：`None | 'hann' | 'hamming' | 'blackman'`；
    - `n_fft` / `n_coeffs`：如 64→33 或 32→17；
    - 其它如是否启用多层、是否保存中间谱等。
- **extraction / dataloader 读取 config**：
  - `src/features/extraction.py`：通过 config 决定：
    - 调用 CLAP 时的段数、帧数，以及最终返回 `(768, 2, T)` 的 T（64 或 32）；
    - 是否额外返回多层 / 只返回 last layer。
  - `src/data_processing/loader.py`：通过相同 config 决定：
    - 从 NPZ / 其它缓存中读取何种形状的 embedding；
    - 如何 reshape 到 `(F, T)` 与对应的 `n_coeffs`，避免脚本层面散落硬编码。
- **脚本只关心“实验配置名”**：
  - 各类 `scripts/run_*.py` 通过传入一个简单的 experiment 名称 / config key（如 `--exp_config=fma_main_fft` 或 `--exp_config=asap_composer_fft`）来加载完整配置；
  - 提高代码的延展性，后续新增 Task 只需在 config 中增加一个条目，而不必修改大量脚本内的常量。

### 2.3 结果目录约定（第二轮迭代）

建议在 `results/` 下按“实验目的”分子目录，便于复用与复现：

```
results/
  fft_genre_specific_analysis/     # 已有：genre 维度的 FFT 指标与曲线
  confusion_analysis/              # 已有：分 band 混淆矩阵
  peak_artifact_investigation/     # 新增：Task 1 输出（随机对照、per-sample 分布、BPM 等）
  highdim_spectral/                # 新增：Task 2 输出（heatmap、cluster 谱、切片）
  fft_window_ablation/             # 新增：Task 3 输出（no window / Hann / Hamming 对比）
  vocal_vs_instrumental/           # 新增：Task 4 输出（与 genre 同结构的 JSON + 图）
```

所有新脚本默认从 `src` 导入，写入上述对应目录；如需复用已有 JSON（如 fft_genre_specific_analysis），只读不写，避免覆盖。

---

## Task 1：追踪 DCT/FFT 中“三个 Peaks”的成因

当前在 DCT 与 FFT 版本的谱图中，不同 genre 的平均频谱都出现了 **位置几乎相同的 3 个尖峰**，形状类似 “高–低–高–低” 的 pattern，令人怀疑可能是：

- 实现/管线中的 **artifact 或 bug**（例如重复拼接、某个维度被 double-count）。
- 来自 **数据分布本身**（例如某种固定的节奏或结构）。
- 来自 **embedding 的结构**（某类维度对特定频率高度敏感）。

### 1.1 实现与数值层面的排查

- **[ ] 单样本、单维度手算对比**
  - 选取 1 条样本的 1 个 embedding 维度：
    - 用 NumPy 在 notebook 中手写 DCT/FFT，并打印完整 33 个系数。
    - 与当前管线输出的对应谱向量对比，检查：
      - 是否做了额外的拼接（如实部/虚部拼成 2×N）。
      - 是否只截取前 N/2 但又在别处镜像了一次。
  - 目的：排除 **“实现导致三峰”** 的可能。

- **[ ] 对照随机向量 / 白噪声**
  - 构造与真实 embedding 形状相同的随机高斯向量，走一遍完全相同的 DCT/FFT+聚合流程。
  - 如果随机数据也出现类似 3 peaks：
    - 很可能是 **算法或后处理方式本身的偏置**（例如对某几个系数做了缩放/归一化）。
  - 如果随机数据是平滑的而真实数据有三峰：
    - 更可能是 **数据或 embedding 结构** 的原因。

- **[ ] 检查归一化与拼接逻辑**
  - 确认在频域上是否做了：
    - `abs` / `power` / `log(·+eps)` 等非线性，而这些是否只对部分系数有效。
    - 是否对前若干系数和后若干系数分别做了不同的缩放（例如手动构造 band 的时候）。
  - 重点排查：
    - 是否存在 **重复使用同一批系数造出多个 band** 的情况。
    - 是否在 reshape / concat 时，把某个轴（如 batch or feature）错当成频率轴。

### 1.2 数据与 embedding 结构层面：三峰是不是“真实模式”

- **[ ] 对比不同层 / 不同模型的 embedding**
  - 在同一批音频上：
    - 理想目标：取同一模型的不同层（或不同 checkpoint）的 embedding 做 FFT/DCT。  
    - **当前发现 / 限制**：CLAP 的 `get_audio_features` 返回的 `last_hidden_state` 已经是我们用的 `(B, 768, 2, 32)` 这种「token×时间」形式，但 `output_hidden_states=True` 得到的 `hidden_states[0]` 等中间层输出是 **4 维的卷积特征图 (B, C, H, W)**，而不是与 `last_hidden_state` 同形状的 Transformer 层输出。  
      - 这意味着不能简单地把 `hidden_states` 当作 “不同 layer 的 `(768, 2, 32)`” 来 reshape/拼接，否则会得到形状错误或物理含义不清的结果（当前 `extract_embeddings_multi_layer` 正是因为这一点会在所有样本上报 `cannot reshape array of size ... into shape (64,2,32)`）。  
      - 简单总结：**CLAP 的 audio encoder 是 HTSAT 结构（卷积 + Transformer 混合），`last_hidden_state` 是已经过一系列处理后的 top layer 表征；`hidden_states` 暴露的是更底层的 conv/patch map，而不是同布局的多层 token 表征。**
    - 因此，多层分析这条路目前暂时 **暂停在实验阶段**：
      - `extract_embeddings_multi_layer.py / multi_layer_spectral.py / run_multi_layer_spectral.py` 标记为 **experimental / 不用于正式实验与论文结果**；
      - 如果未来真的需要做「多层三峰对比」，需要专门阅读 HTSAT 源码，设计一条针对 HTSAT 的多层抽取接口（在 encoder 内部拿到真正的多层 token 输出），而不是简单依赖 `hidden_states` + reshape。
    - 概念上仍然成立的推断是：
      - 如果未来能拿到多层 token 表征，并在所有层/模型上都观察到三峰，可能是 **positional / architecture 级别的频率偏置**；
      - 如果只在某几层特别明显，则说明是这几层在学习 genre/task 时引入的结构。

- **[ ] 分 genre 的平均谱 vs per‑sample 分布**
  - 目前图是对一个 genre 的 **平均谱**，可能把很多细节平均掉。
  - 计划：
    - 把同一 genre 的谱向量按样本画成 **多条透明曲线**（或 violin/箱线图），叠加在平均谱之上。
    - 观察这三个 peak：
      - 是每首歌都有，还是只是个别样本极端大被平均出来。
  - 如果 peak 只来自少数样本：
    - 说明平均可视化存在误导，需要重新设计 summary 方式。

- **[ ] BPM / 节奏相关性的 sanity check**
  - 尽管当前 FFT 的频率轴是对 embedding 维度或 token 序列做变换，不一定直接对应物理时间，但可以：
    - 对每首歌估计 BPM 区间（来自原始音频/元数据）。
    - 计算在三峰对应的频率 index 上的能量，并与 BPM 做相关性检验。
  - 如果不同 BPM 的歌曲在这几个系数上能量分布几乎一样：
    - 基本可以排除“这些 peak 直接对应某个恒定节奏频率”的解释。

> 这一 Task 的输出：  
> **一小节总结“实现 artifact 是否被排除、三峰是否在 per‑sample 上也稳定存在、和节奏/数据有什么关系”。**

#### 1.3 结论：三峰是 embedding 的时间轴结构，不是我们计算方式导致的

对比 **composer（n_coeffs=17，T=32）** 与 **FMA / highdim（n_coeffs=33，T=64）** 的 mean magnitude 与 learned weight 曲线可知：

- **Peak 的相对位置一致**  
  - Composer 17 系数：峰大致在 index **4, 8, 12**（若把 0 当作 DC，则约在 1/4、2/4、3/4 处）。  
  - FMA 33 系数：三峰大致在 **8, 16, 24** 附近，即同样在 **1/4、2/4、3/4** 的归一化频率位置。  
  - 即：**与分辨率无关，都是“每隔约 1/4 长度一个峰”的同一模式。**

- **FFT 做在什么上**  
  - 我们是在 **embedding 的时间轴**（最后一维，长度 T=32 或 64）上做 `rfft`，不是在 feature 维、也不是在音频波形上。  
  - 因此这里的“频率”是 **沿 token/帧序列的离散频率**（k/N，N=32 或 64），不是 Hz。

- **含义**  
  - 若三峰来自我们的实现错误（例如轴搞错、重复拼接、band 划分），换 T/n_coeffs 后峰位应会变或消失；但 **17 与 33 下相对峰位一致** → 更可能是 **embedding 沿时间轴的真实谱结构**。  
  - 即：**CLAP 编码器在时间维上的输出，在“沿帧序列的频谱”上存在稳定模式**——能量在 k/N ≈ 1/4, 2/4, 3/4 处相对偏高，可能来自：  
    - **架构**：HTSAT 的 patch/stride、位置编码或层结构使时间序列呈现固定周期分量；  
    - **数据**：训练数据在 embedding 时间维上的共性；或二者共同作用。  
  - 建议在文中表述为：**“三峰是 CLAP 表征在时间维上的固有谱结构，与具体实验的 FFT 分辨率（17 或 33 系数）无关。”**

#### 执行步骤（Task 1）

1. **统一入口与清理 spectral**
   - 在 `src/data_processing/loader.py` 中：**唯一** reshape 为 (B, 1536, 64)，并在 docstring 中写明“原始 (B, 768, 2, 64) → (B, 1536, 64)；T=64 → FFT 得 n_coeffs=33”。所有脚本只从 loader 导入 load_data，不再在脚本内做其它 reshape。
   - 在 `src/analysis/spectral.py` 中：**删除** `get_spectral_bands` 与 `get_band_energy`（不实现 energy、不对 band 做任何调整）；保留 `get_raw_band_features`。
2. **新建 `src/analysis/peak_artifact_checks.py`**
   - `run_single_sample_fft_check(embeddings_path, sample_idx=0, dim_idx=0)`：取单样本单维度，用 `np.fft.rfft` 手算 33 系数（T=64 → rfft 得 33），与 `apply_transform(..., 'fft', axis=2)` 逐元素对比，打印 diff；写入 `results/peak_artifact_investigation/single_sample_fft_check.txt`。
   - `run_random_vector_fft_pipeline(shape=(1, 1536, 64), n_trials=5)`：生成与真实数据同形状的随机高斯 (B=1, F=1536, T=64)，经 `apply_transform(..., 'fft')` 后对“频率轴”取平均得到一条 33 维曲线，保存并画图；若多次都出现类似三峰 → 倾向算法/后处理 artifact。
   - `check_band_concat_usage()`：在 `scripts/run_single_genre_analysis.py` 与 `run_fft_analysis.py` 中 grep 所有对系数的 reshape/concatenate，确认没有重复使用同一段系数造多个 band，文档化“band 划分方式”到 plan 或 README。
3. **per-sample 分布与 BPM**
   - 在 `peak_artifact_checks.py` 中增加：对某一 genre 所有样本保留 `X_freq`（不平均），按样本画 33 维曲线（透明或 violin），与平均谱叠加，保存到 `results/peak_artifact_investigation/per_sample_spectrum_<genre>.png`。
   - BPM：若元数据或已有 pipeline 有 BPM/节奏标签，在 `peak_artifact_checks.py` 中增加“三峰对应 index 能量 vs BPM”的相关系数计算，结果写入 `results/peak_artifact_investigation/bpm_correlation.txt`；若无 BPM，步骤可标为可选，留空文件说明。
4. **脚本**
   - 新增 `scripts/run_peak_artifact_checks.py`：调用上述三个检查，写结果到 `results/peak_artifact_investigation/`。

**复用**：`load_data` → `src.data_processing.loader`；`apply_transform` → `src.analysis.spectral`；不新增 run_probe 调用。

#### 1.4 后续关键补充实验：顺序对照（Permutation vs Layout）

> 噪声对照实验（random vector FFT）已经说明：**FFT 本身不会「自动造三峰」**。下一步更关键的是确认：三峰究竟来自 **token 顺序 / flatten layout**，还是来自 **embedding 数值内容本身的统计结构**。

**补充实验 A：真实 embedding 的 token permutation（顺序打乱）**  

- **目的**：测试三峰是否依赖于**原始时间顺序**。  
- **设计**：
  - 在真实 FMA embedding 上，保持每个样本的频率维 / feature 维不变，只对**时间轴上的 token index 做随机置换**：  
    - 对每个样本生成一个随机 permutation `perm`，将 `X[b, f, t]` 重排为 `X_perm[b, f, perm[t]]`。  
    - 然后在 permuted 时序上做与原实验完全相同的 FFT（axis=2）、mean magnitude 和 learned weight 计算。  
  - 比较：
    - 原始顺序 vs permuted 顺序下的 mean magnitude 曲线（是否仍在 1/4、2/4、3/4 有稳定峰）。  
    - 原始 vs permuted 下的 learned weight spectral profile。  
- **判读**：
  - 若 **三峰在 permutation 后显著减弱或消失**：  
    → 峰**高度依赖原始 token 顺序**，说明更多是由 **layout / periodic ordering** 导致（例如固定 patch/stride + 位置编码模式），而非仅仅由单点数值分布决定。  
  - 若 **三峰在 permutation 后仍稳定存在**：  
    → 峰更可能来自 embedding 本身在时间维的**更深层统计结构**（例如每个位置的局部模式整体上可交换），而不只是简单“顺序编码”的 footprint。

**补充实验 B：改变 flatten / FFT layout（time-major vs freq-major）**  

- **目的**：测试三峰是否由我们选择的 flatten 方式 / FFT 轴引入的“布局周期性”决定。  
- **设计示例**（都以真实 embedding 为输入）：  
  1. **Time-major baseline**（当前做法）：  
     - 先在频率/feature 维聚合或展平，再沿**时间轴**做 FFT：`X ∈ R^{B×F×T} → FFT_t(X)`。  
  2. **Freq-major variant**：  
     - 人为将 flatten 顺序改为对**频率轴**做 FFT（如 `X ∈ R^{B×T×F}`，在 F 上做 rfft），或先对时间轴做某种聚合后，再在另一个轴上做 FFT。  
  3. **仅取 time axis，不混 freq**：  
     - 例如先对频率维做平均或 pooling 得到 `X_time_only ∈ R^{B×1×T}`，再沿时间轴 FFT；对比与“全 feature 展平 + FFT”的谱形差异。  
- **判读**：
  - 若 **改变 flatten / FFT 轴后，三峰的位置或形状发生系统性变化**，尤其是出现“跟着 layout 走”的新峰位：  
    → 可较强地判断为 **representation layout induced periodicity**——即我们如何展开与在哪个轴上做 FFT，会直接决定谱中的规律；当前三峰是“时间轴 layout”下的偏置。  
  - 若 **无论如何改 layout，三峰总在相同归一化频率处出现**：  
    → 更支持“架构 + 数据”共同塑造的内在谱结构，而非特定 flatten 方案的 artifact。

> 实现建议：  
> - 在 `src/analysis/peak_artifact_checks.py` 中新增 permutation 与 layout 变体的实验函数，结果写入 `results/peak_artifact_investigation/`（例如 `permuted_time_spectrum_*.png`, `layout_variants_spectrum_*.png`）。  
> - 文档层面，将“噪声对照 + 顺序对照 + layout 对照”作为三步逻辑链：  
>   1）不是 FFT 自己造峰；2）峰对真实顺序/布局的敏感性；3）综合架构与数据假说，给出更精细的成因定位。  

---

## Task 2：高维频谱结构可视化（3D / Heatmap 视角）

当前谱图是对整层 768 维 embedding、所有样本按 genre 做了平均，因此很可能：

- 大量维度其实是 **近似噪声或弱结构**；
- 少数维度有非常强的结构，被平均后表现为全局的 3 peaks。

为了弄清“哪些维度在贡献这些 peaks、哪些维度是没啥用的”，需要更高维、更细致的可视化。

### 2.1 数据形状与基本变换

假设对某一层的 embedding，有：

- 原始：`X` 形状约为 `[n_samples, 768]`（或 `[n_samples, seq_len, 768]`，这里只关注被 FFT 的那一维）。
- 经过 DCT/FFT 后，对每个 embedding 维度得到 `n_coeffs=33` 个系数：
  - 形状可记为：`X_freq ∈ R^{n_samples × 768 × 33}`。

我们之前画的是对 `n_samples` 和 768 都平均后得到的 `mean_spectrum ∈ R^{33}`。

### 2.2 维度×频率的 Heatmap / 3D 视图

- **[ ] Step A：按维度聚合样本**
  - 对每个 embedding 维度 `d`：
    - 取所有样本在该维度上的谱 `X_freq[:, d, :]`，沿样本轴取均值和标准差：
      - `mean_spectrum_d ∈ R^{33}`，`std_spectrum_d ∈ R^{33}`。
  - 把所有维度的 `mean_spectrum_d` 堆成矩阵：
    - `M ∈ R^{768 × 33}`，行是维度、列是频率系数。

- **[ ] Step B：画维度×频率的 Heatmap**
  - 直接把 `M` 画成 heatmap（横轴：频率系数，纵轴：embedding 维度）。
  - 为了更易读：
    - 可以先对每个维度做归一化（例如除以自身 L2 范数），避免少数极大值 dominate。
    - 也可以按 **整体能量或峰值大小排序维度**，把“有强结构的维度”排在顶部。

- **[ ] Step C：聚类维度并画 cluster 平均谱**
  - 在 `M` 上做简单的 k‑means / spectral clustering（例如 k=5~10）：
    - 得到若干“谱形态相似”的维度簇。
  - 对每个簇：
    - 画一条 **cluster mean spectrum**（33 维）。
    - 比较各簇是否都在同样 3 个位置有峰，还是只有某几个簇有。
  - 若发现：
    - 只有少数簇在那 3 个 index 有大峰，其它簇比较平坦；
    - 那么当前平均图里的三峰，实际上是**那几个簇的 footprint**，而不是全层的“平均行为”。

### 2.3 样本轴上的变化：15k × 33 的“切片”检查

- **[ ] Step D：样本×频率的分布**
  - 固定若干典型的 embedding 维度（例如：
    - 一个来自“强三峰簇”的维度；
    - 一个来自“近似平坦簇”的维度），
  - 对这些维度画 `n_samples × 33` 的 heatmap 或 PCA 投影：
    - 看 15k 首歌在这些系数上的分布是否有结构（e.g. genre‑specific cluster）。

- **[ ] Step E：跨 genre 的对比**
  - 对比同一维度在不同 genre 子集上的 `mean_spectrum_d`：
    - 如果峰的位置固定但幅度在不同 genre 之间变化明显：
      - 说明位置可能来自 architecture / embedding，**区分信息主要体现在幅度差异**。

> 这一 Task 的目标：  
> **把“平均谱上的 3 个 peak”拆解成“哪一部分维度 / 哪些 cluster 贡献了这些 peak”，顺便识别哪些维度几乎是无结构噪声。**

#### 执行步骤（Task 2）

1. **数据形状与 loader**
   - 若当前 FFT 在 (B, 1536, T) 上做，得到 (B, 1536, n_coeffs)；若需“按 768 维”做 FFT，需在 loader 中提供 (B, 768, T) 的视图（例如只取第一段 32 帧，或合并两段为 64 再在 axis=2 做 FFT）。在计划中明确：**高维 heatmap 的“维度”是 768 还是 1536**，与现有 spectral_profile 的 33 系数对应关系（若 1536，则 33 为时间维 rfft 系数，每个 embedding 维度共享同一套 33）。
2. **新建 `src/visualization/spectral_highdim.py`**
   - `compute_dimension_spectrum_matrix(X_freq)`：输入 (n_samples, n_dims, n_coeffs)，对每个 dim 沿 sample 轴求 mean 和 std，返回 `M_mean (n_dims, n_coeffs)`, `M_std (n_dims, n_coeffs)`。
   - `plot_heatmap_dim_x_freq(M, out_path, title, normalize_per_dim=True)`：画 heatmap，可选按行 L2 归一化或按行排序（如按 max 或 L2 排序）。
   - `cluster_dimensions_kmeans(M, n_clusters=8)`：对 M 做 k-means，返回每个维度的 label；对每个 cluster 求平均谱并画线（`cluster_mean_spectra.png`）。
   - `plot_per_sample_curves_for_dims(X_freq, dim_indices, genre_mask, out_path)`：固定若干 dim（如“强峰簇”的代表、平坦簇的代表），画这些 dim 上所有样本的 33 维曲线（半透明）并叠加平均，按 genre 分子图或颜色。
   - `plot_3d_sample_freq_surface(X_freq, out_path, title, max_samples=200)`：对样本轴和频率轴画 3D surface（sample×freq×mean-over-dims），用于看“不同 sample 在低/中/高频能量分布的整体形状”。
   - `plot_3d_dim_freq_surface(S, out_path, title, dim_step=4)`：对单个 sample 的 `(n_dims, n_coeffs)` 画 3D surface（dim×freq×magnitude），支持对 dim 轴下采样，便于观察“哪些维度在三峰附近特别活跃”。
3. **脚本**
   - 新增 `scripts/run_highdim_spectral.py`：load_data → apply_transform → 得到 X_freq；若当前是 (B,1536,33)，则“维度”为 1536；
     - 默认 `--genres` 为 8 个 genre：`Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock`；
     - `--plot_sample_surface`：画样本×频率的 3D surface（最多 200 首），输出 `surface_sample_x_freq_mean_over_dims.png`；
     - `--per_sample_heatmap --per_sample_n 3`：对每个 genre 随机抽取 3 首歌，画单个 sample 的 dim×freq heatmap，输出 `single_sample_dim_freq_<Genre>_idx*.png`；
     - `--per_sample_3d --per_sample_n 3`：对每个 genre 随机抽取 3 首歌，画单个 sample 的 dim×freq 3D surface，输出 `single_sample_dim_freq_3d_<Genre>_idx*.png`；
     - 以及 `cluster_mean_spectra.png`、`per_sample_curves_dims_*.png` 等，用于分析“哪些维度簇贡献了三峰”以及 per‑sample 在代表维度上的分布。

**复用**：`load_data` → `src.data_processing.loader`；`apply_transform` → `src.analysis.spectral`；不依赖 probe。

---

## Task 3：FFT Window 与参数的系统性调试

目前的 FFT 实现没有使用 window function，等价于对“截断信号”做矩形窗 FFT，理论上容易引入 **谱泄漏 (spectral leakage)**，在有限长度的嵌入/序列上可能造成假峰。

### 3.1 引入不同 window function 并做对比

- **[ ] 在进行 FFT 之前，对被变换的轴应用 window**
  - 典型选择：
    - Hann（推荐作为默认）。
    - Hamming。
    - Blackman。
  - 对每一种 window：
    - 重新计算频谱与 genre‑specific 平均谱。
    - 比较：
      - 三个 peak 的位置是否变化。
      - 峰值和底噪之间的对比度是否明显减弱。

- **[ ] Ablation：No window vs Hann vs Hamming**
  - 统一画在同一张图中（或者并排）：
    - 如果峰在有 window 时明显变钝甚至消失：
      - 说明这部分结构很可能是 **windowing artifact**。
    - 如果峰在各种 window 下都稳定存在：
      - 更可能是 embedding 结构 / 数据本身。

### 3.2 FFT 参数与轴选择的 sanity check

- **[ ] 明确 FFT 所在的轴与长度**
  - 在文档与代码注释中写清楚：
    - 我们是在 **沿哪一个维度** 做 FFT（时间帧 / token 位置 / embedding 维度索引）。
    - `n_fft` / `n_coeffs=33` 的具体来源（是截断、下采样还是对更长 FFT 的前 33 项）。

- **[ ] 不同 `n_coeffs` / zero‑padding 的对比**
  - 例如：
    - 取 `n_coeffs = 33, 64, 128` 等不同长度，比较三峰是否总出现在“比例相同的位置”，还是在绝对 index 上固定。
  - 如果三峰只在特定 `n_coeffs` 出现：
    - 说明可能与我们对系数的截断/重采样方式有关。

> 这一 Task 的目标：  
> **把“是否是 FFT/window 相关的数值 artifact”与“embedding/数据的真实谱结构”区分开。**

#### 执行步骤（Task 3）

1. **扩展 `src/analysis/spectral.py`**
   - 在 `apply_transform` 中为 `transform_type in ('fft','dft')` 增加可选参数 `window_type: str | None = None`。在沿 `axis` 做 `np.fft.rfft` 之前，在该轴上应用 window：若 `window_type == 'hann'` 用 `np.hanning(length)`，`'hamming'` 用 `np.hamming(length)`，`'blackman'` 用 `np.blackman(length)`；`None` 表示矩形窗（当前行为）。确保 window 与输入在 axis 上长度一致（broadcast 或 expand_dims 后相乘）。
   - 在模块顶部或 docstring 中注明：FFT 沿 **时间轴（最后一维）** 做，`n_coeffs` 为 `rfft` 输出长度（与时间长度和 padding 一致）。
2. **Ablation 脚本**
   - 新增 `scripts/run_fft_window_ablation.py`：对同一批数据（如固定一个 genre）依次调用 `apply_transform(..., transform_type='fft', window_type=None|'hann'|'hamming')`，每种得到 coeffs 后，沿 (sample, dim) 平均得到一条 33 维“平均谱”；三条曲线画在同一张图，保存到 `results/fft_window_ablation/spectral_profile_by_window.png`。可选：对同一 genre 用三种 window 各跑一次 probe full-embedding accuracy，写入 JSON，便于确认 window 是否改变分类表现。
3. **n_coeffs 对比（可选）**
   - 若时间维可变为 64（例如用两段拼接），在 spectral 中支持 `n_fft` 或 padding 参数，得到 33 vs 65 等不同长度，各跑一次平均谱并对比三峰是否在“比例位置”一致；结果写入 `results/fft_window_ablation/n_coeffs_comparison.txt`。

**复用**：仅改 `apply_transform` 一处；脚本与现有 `run_single_genre_analysis` 共享同一 load_data + probe 逻辑（建议脚本内从 src 导入）。

---

## Task 4：古典作曲家分类（Bach / Beethoven / Liszt / Schubert / Chopin）

利用现有的古典音乐数据集 `/home/evev/noiseloss/datasets/D_asap_100`，设计一个 **作曲家多分类任务**，在与 genre 完全不同的数据分布与标签空间上复用同一套频域 probe 框架，用于检验：

- 三峰 / 频带结构在 **古典作曲家识别** 任务中是否仍然存在；
- 频域 probe 在“同一乐器配置、不同作曲家风格”这样的细粒度任务上是否依然有解释力。

### 4.1 数据与标签定义（D_asap_100）

- **数据源**：`/home/evev/noiseloss/datasets/D_asap_100`，包含若干 15s WAV 片段，文件名模式：
  - `Bach_xxx_midi_score_short.mid_ori_cut15s.wav`
  - `Beethoven_xxx_midi_score_short.mid_ori_cut15s.wav`
  - `Liszt_xxx_midi_score_short.mid_ori_cut15s.wav`
  - `Schubert_xxx_midi_score_short.mid_ori_cut15s.wav`
  - `Chopin_xxx_midi_score_short.mid_ori_cut15s.wav`
- **统计（当前目录下实际数量）**：
  - Bach: 15 条
  - Beethoven: 26 条
  - Liszt: 15 条
  - Schubert: 10 条
  - Chopin: 22 条
  - 总计 88 条样本，均为 15 秒切片。
- **标签空间**：5 类作曲家
  - `0: Bach`
  - `1: Beethoven`
  - `2: Liszt`
  - `3: Schubert`
  - `4: Chopin`

> 约定：本 Task 只使用这 5 位作曲家的样本，其它作曲家（Debussy, Haydn, Mozart, Ravel, Schumann, Scriabin 等）不纳入本实验。

### 4.2 音频裁剪与 CLAP 特征抽取（10 秒窗口，768×2×32）

- **裁剪策略**：
  - 每条 WAV 文件原始长度为 15s；
  - 本实验统一只使用 **中间的 10 秒**：
    - 若采样率为 `sr`，则裁剪区间为 \([2.5\text{s}, 12.5\text{s})\)，在样本点上为 \([2.5·sr, 12.5·sr)\)；
    - 好处：避开开头 / 结尾的过渡 / 静音，更代表乐曲主体。
  - 实现时可使用 `librosa.load(..., sr=48000)` 固定采样率，然后用数组切片完成裁剪。
- **CLAP 特征抽取（与主实验同构）**：
  - 直接复用当前 CLAP 音频 encoder 的 **默认行为**：
    - 使用 `model.get_audio_features(**inputs).last_hidden_state` 作为音频 embedding；
    - **不在我们这边额外改帧长 / hop 或手工切片**，而是让 CLAP 自己把输入音频按 10s 窗口处理，输出形状为 `(768, 2, 32)`。
  - 形状约定（本 Task 专用）：
    - 单条样本的原始输出形状为 `X_raw ∈ R^{768 × 2 × 32}`；
    - 展平成 `X_sample ∈ R^{768*2 × 32} = R^{1536 × 32}`，与主实验的 `(768*2, 64)` 在“频段 × 时间”布局上保持一致，只是时间长度从 64 变为 32。
  - **FFT 约定**：
    - 在时间轴（长度 32）上做 `rfft(32) → 17` 个频率系数，即本 Task 的 `n_coeffs=17`；
    - 与主实验的 `n_coeffs=33` 不同，但不影响整体分析框架（只是频率分辨率较低）。
  - **实现与可配置性约定**：
    - 在 `src/features/extraction.py` 中为 CLAP 抽取增加一个配置开关（或枚举），支持：
      - `mode='fma_main'`：输出 `(768, 2, 64)` → `(1536, 64)`，`n_coeffs=33`；
      - `mode='asap_composer'`：输出 `(768, 2, 32)` → `(1536, 32)`，`n_coeffs=17`；
    - 在 `src/data_processing/loader.py` 中对应增加选项 / 读取统一的 config，而不是在脚本里硬编码形状。

### 4.3 FFT 与作曲家频谱 profile 计算

- **FFT 设置**：
  - 对每条样本的 embedding `X_sample ∈ R^{768 × 64}`：
    - 在时间轴上做 `np.fft.rfft(X_sample, axis=1, norm='ortho')`，得到 `X_freq_sample ∈ R^{768 × 33}`；
    - 取幅度：`|X_freq_sample|`。
  - 所有样本堆叠后得到：
    - `X_freq ∈ R^{N × 768 × 33}`，`N` 为总样本数（约 88）。
- **作曲家“频谱 profile”定义**：
  - 类似 genre 实验，区分两种 profile：
    1. **Mean magnitude profile（per-composer）**
       - 对某个作曲家 c，取其所有样本的 `X_freq[c]`，在样本维与维度维上取平均：
         - `mean_mag_c[k] = mean(|X_freq[c, :, :, k]|)`，得到一条 33 维曲线。
       - 画在同一张图上比较 5 位作曲家的平均谱。
    2. **Learned weight profile（per-composer one-vs-rest probe）**
       - 构造 one-vs-rest 线性 probe，与 genre 实验一致：
         - `y_c = 1` 表示作曲家 c，`0` 表示其他 4 位；
         - 特征为 `X_features = X_freq.reshape(N, 768*33)`。
       - 用 `LogisticRegression(class_weight='balanced')` 训练探针，取 `model.coef_`，按 (n_dims, n_coeffs) reshape 后对维度取平均，归一化到 [0,1]，得到 `weight_profile_c ∈ R^{33}`。
       - 这条曲线即为“在区分作曲家 c vs 其他人时，各频率系数的重要性”。

### 4.4 采样与加权策略（按作曲家数量加权）

- **样本数量不平衡**：
  - 当前统计显示：Beethoven（26）和 Chopin（22）样本较多，Schubert（10）最少。
- **实验原则**：
  - 不进行过于激进的下采样，保持“多少作品就贡献多少信息”的直觉；
  - 在 probe 训练时使用 `class_weight='balanced'` 抵消类别不平衡。
- **具体策略**：
  - 直接使用 5 个作曲家的 **全部样本**；
  - 在报告 / 图标题中标注每位作曲家的样本数（例如 `Beethoven (n=26)`），明确“weighted by count”的事实；
  - 可选扩展：在附录中再做一次“各作曲家随机下采样到 min_count=10”的平衡实验，用于 sanity check（但不是主结果）。

### 4.5 可视化与结果目录

- **结果目录**：在 `results/` 下新增：
  - `results/composer_classical/`：
    - `composer_spectral_summary_fft_metrics.json`：记录每位作曲家的 probe 准确率、mean magnitude profile、learned weight profile 等；
    - `composer_mean_magnitude_fft.png`：5 位作曲家 mean magnitude profile 同图对比；
    - `composer_learned_weight_fft.png`：5 位作曲家 learned weight profile 同图对比；
    - （可选）`composer_<Name>_one_vs_rest_spectral_profile.png`：每位作曲家单独的双面板图（左：mean magnitude，右：learned weight）。
- **画图约定**：
  - 与 genre 图保持一致的风格与坐标轴（x=FFT coefficient index，y=normalized magnitude/weight）；
  - 图例中注明作曲家名字与样本数：`Bach (n=15)` 等。

### 4.6 脚本与复用关系（仅规划，不立刻实现）

- **脚本草案**：`scripts/run_composer_classical_fft.py`（名称暂定）
  - 负责：
    - 遍历 `/home/evev/noiseloss/datasets/D_asap_100` 中 5 位作曲家的 WAV 文件；
    - 裁剪中间 10 秒并送入 CLAP，得到 `(768, 64)` embedding；
    - 堆叠为 `X ∈ R^{N × 768 × 64}`，构建 `y_composer ∈ {0,…,4}`；
    - 调用 `apply_transform(..., transform_type='fft', axis=2, window_type='hann')`（或与主实验保持一致的 window 设定）；
    - 计算 mean magnitude 与 learned weight profile；
    - 写 JSON + 画图到 `results/composer_classical/`。
- **复用模块**：
  - `src.analysis.spectral.apply_transform`：统一 FFT 接口；
  - `src.training.probes.run_probe` 或一个简化版 one-vs-rest probe 工具；
  - 如有需要，可在 `src/data_processing/` 新增 `loader_asap_composer.py` 专门处理 D_asap_100 的文件遍历与 10s 裁剪。

> 本 Task 当前仅处于 **计划阶段**：以上规划先写入 plan，并与现有 Task 1/2/3 的结果一起讨论是否值得完全实现；在确认价值后再分步落地脚本与代码。

---

## 总结与预期输出

执行以上 Task 后，我们希望得到：

- **对“三个 peaks”来源的清晰结论**：是实现/FFT artifact，还是 embedding 与数据的真实频谱结构。
- **一套高维频谱可视化工具**（heatmap + clustering），可以直接在 768×33 尺度上分析模型内部频域模式。
- **更稳健的 FFT 实现与 windowing 配置**，在后续所有实验中统一采用。
- **一个新的音乐分类任务（古典作曲家分类：Bach / Beethoven / Liszt / Schubert / Chopin）上的频域 probe 结果**，用于对比 genre 任务与作曲家任务上的三峰 / 频带模式是否一致，从而支撑“本方法对更广泛音乐属性同样有效”的主张。

---

## 四、执行顺序与依赖 (Execution Order)

建议按以下顺序推进，避免重复劳动：

1. **前置（必做）**
   - **数据形状**：全项目唯一约定为 **(B, 768, 2, 64) → (B, 1536, 64)**，T=64 → rfft 得 n_coeffs=33。`src.data_processing.loader.load_data` 必须且仅返回 (B, 1536, 64)；所有新脚本从 loader 导入 load_data，不再在脚本内做任何其它 reshape。
   - **spectral.py**：删除 `get_spectral_bands` 与 `get_band_energy`（不做 energy、不调 band）；保留 `get_raw_band_features`。
   - **探针**：新脚本一律使用 `src.training.probes.run_probe` / `run_probe_with_predictions`，不再从 `run_single_genre_analysis` 复制 `run_probe`。

2. **Task 3（FFT window）**  
   - 先做：只改 `spectral.py` 一处，即可在后续所有实验里复用；ablation 脚本轻量，便于快速得到“加窗后三峰是否仍存在”的结论。

3. **Task 1（三峰成因）**  
   - 依赖统一后的 load_data 与 apply_transform；可与 Task 3 并行开发，但跑随机对照时建议用“加窗前”的 FFT 先跑一轮，再与加窗后对比。

4. **Task 2（高维可视化）**  
   - 依赖明确的 X_freq 形状（(B, F, n_coeffs)）；可与 Task 1 并行；输出目录独立，不覆盖已有 results。

5. **Task 4（古典作曲家分类）**  
   - 依赖 D_asap_100 数据读取、10 秒裁剪与 CLAP 特征抽取；可在 Task 1/2/3 任一完成后做，用于对比“三峰 / 频带模式在 genre vs 作曲家任务上是否一致或有系统性差异”。

