## 第三次 Sanity Check 计划：时间轴周期性与三峰结构

**目标**：在现有代码与结果基础上，系统性地完成“第三次 sanity check”，聚焦 CLAP / 其他音频模型在时间轴上的固定周期与频域三峰（1/4, 2/4, 3/4 位置），排除实现 / 数据处理层面的歧义，并给出可重复的实验脚本设计。  
本次计划仅包含 **Step 0, Step 1, Step 2, Step 4**，**不再做 Step 3（白噪声/零输入等新合成信号）**。

---

## 0. 现有处理与代码结构总览（只读）

### 0.1 Embedding 与数据集约定

- **FMA / LMD 主实验（`fma_main`）**
  - NPZ：`data_artifacts/clap_embeddings_t64.npz`
  - 原始 CLAP 输出形状：`(B, 768, 2, 64)`  
    - `768`: 通道维（特征维）  
    - `2`: 两段 10s 的分段 / lane  
    - `64`: 时间帧（两段 32 帧在时间轴拼接：`2 × 32 → 64`）
  - 统一 reshape（`src/data_processing/loader.py`）：
    - 输入：`(B, 768, 2, 64)`  
    - 输出：`X ∈ R^{B × 1536 × 64}`，其中 `1536 = 768 × 2`，`T = 64`
  - 频域变换（`src/analysis/spectral.py`）：
    - `apply_transform(X, transform_type="fft", axis=2, window_type=...)`
    - `axis=2` 为时间维 \(T\)，使用 `np.fft.rfft`，长度 `n_fft=64`  
    - `rfft(64) → N_COEFFS = 33`，频域形状：`X_freq ∈ R^{B × 1536 × 33}`

- **ASAP 作曲家任务（`asap_composer`）**
  - 配置：`src/config/spectral_experiments.py` 中 `ASAP_COMPOSER`
  - 原始 CLAP 输出形状：`(B, 768, 2, 32)`（单段 10s，时间帧 `T=32`）
  - reshape：`(B, 768, 2, 32) → (B, 1536, 32)`，`T=32`
  - 频域变换：`rfft(32) → n_coeffs=17`，`X_freq ∈ R^{B × 1536 × 17}`

> 本轮 sanity check 中，**LMD/FMA 使用 `fma_main` 配置，ASAP 使用 `asap_composer` 配置**。所有形状与路径以 `EmbeddingConfig` 为单一真源。

### 0.2 已有三峰排查与可视化代码

- **三峰排查主模块**：`src/analysis/peak_artifact_checks.py`
  - `run_single_sample_fft_check(...)`：单样本单维度手算 `np.fft.rfft` 与 `apply_transform` 对比。
  - `run_random_vector_fft_pipeline(...)`：在随机高斯向量上跑 FFT 管线并画平均谱（已用于确认 FFT 本身不会自动产生现有三峰）。
  - `plot_per_sample_spectra_for_genre(...)`：对某一 genre（FMA）绘制每个样本的频谱 + 平均谱。
  - `run_time_permutation_experiment(...)`：打乱时间维顺序，对比原始与 permuted 的平均谱。
  - `run_layout_variants_experiment(...)`：对比 “全 feature + 时间 FFT” 与 “先对 feature 平均再沿时间 FFT” 的谱形。

- **三峰排查入口脚本**：`scripts/run_peak_artifact_checks.py`
  - 调用上面的函数，并额外：
    - 训练若干 genre 的 `one-vs-rest` FFT probe，绘制 learned spectral profile。
    - 输出到 `results/peak_artifact_investigation/`。

- **统一频域变换实现**：`src/analysis/spectral.py`
  - `apply_transform(embeddings, transform_type='dct'|'fft'|'dft', axis=-1, window_type=None|'hann'|'hamming'|'blackman')`
  - 对 `transform_type in ('fft', 'dft')`：
    - 可选 window：`_apply_window(x, axis, window_type)`  
    - 然后 `np.fft.rfft(x, axis=axis, norm='ortho')` → 返回绝对值频谱。

> 总结：当前代码已经支持 **统一 FFT 管线、window 选择、per-genre 三峰可视化与若干 sanity check**。第三次 sanity check 需要在此基础上扩展 **zero padding + 自相关 + 多模型对比（基于 LMD/ASAP）**。

---

## 1. Step 1：Zero Padding + 高分辨 FFT（LMD & ASAP）

### 1.1 目标与原理

- **目的**：
  - 在 LMD（FMA 主任务）与 ASAP（作曲家任务）上，对 embedding 时间序列做 **zero padding**，提高频率采样分辨率；
  - 观察半谱长附近及 1/4, 2/4, 3/4 处的三峰是否：
    - 在 **归一化频率轴上**稳定存在（与 FFT 长度无关），还是
    - 与输入音乐（例如 Schubert 片段、不同 composer）相关而产生偏移。
- **原理**：
  - 零填充不会改变信号的真实频率成分，只是让频谱在频率轴上被更密集采样；
  - 若峰由**严格的时间周期结构**产生，其频率位置在归一化坐标上应随 padding 保持不变；
  - 若峰由数据/任务中的“真实音乐频率”主导，在高分辨率下会表现为与物理频率相关的峰位，而非精确锁定在 0.25 / 0.5 / 0.75。

### 1.2 新增模块与函数设计

**文件**：`src/analysis/zero_padded_fft.py`（新建）

#### 1.2.1 公共工具函数

- **函数**：`get_embedding_and_config(config_name: str) -> tuple[np.ndarray, np.ndarray, EmbeddingConfig]`
  - **输入**：
    - `config_name: str`：`"fma_main"` 或 `"asap_composer"`
  - **内部行为**：
    - 调用 `get_embedding_config(config_name)`（`src/config/spectral_experiments.py`）
    - 使用 `np.load(config.embeddings_file)` 加载对应 NPZ：
      - `embeddings`，`genres`（或 `composer_labels` 等）
    - 根据 `EmbeddingConfig` 统一 reshape：
      - 原始：`(B, 768, 2, T_frames)`，其中 `T_frames = config.frames_per_segment`
      - 输出：`X ∈ R^{B × (768*2) × T_frames} = R^{B × 1536 × T_frames}`
  - **输出**：
    - `X: np.ndarray`，形状：
      - FMA：`(B_fma, 1536, 64)`
      - ASAP：`(B_asap, 1536, 32)`
    - `y_str: np.ndarray[str]`：genre 名称或作曲家名称
    - `cfg: EmbeddingConfig`：配置对象，含 `n_fft`, `n_coeffs` 等

- **函数**：`zero_pad_along_time(X: np.ndarray, target_factors: list[int]) -> dict[int, np.ndarray]`
  - **输入**：
    - `X ∈ R^{B × F × T}`：时间轴在最后一维  
    - `target_factors`：如 `[1, 2, 4, 8]`，代表 `T_pad = factor * T`
  - **输出**：
    - 字典：`{factor: X_padded}`，其中  
      - `X_padded ∈ R^{B × F × (factor * T)}`，在时间末尾补零。

- **函数**：`compute_fft_magnitude(X_list: dict[int, np.ndarray], window_type: Optional[str]) -> dict[int, np.ndarray]`
  - **输入**：
    - `X_list`：由 `zero_pad_along_time` 生成
    - `window_type`：`None | 'hann' | 'hamming' | 'blackman'`
  - **行为**：
    - 对每个 factor：
      - 调用 `apply_transform(X_padded, transform_type="fft", axis=2, window_type=window_type)`
      - 得到 `coeffs ∈ R^{B × F × n_coeffs_factor}`，其中
        - `n_coeffs_factor = rfft(factor * T)` 长度
      - 对 `(B, F)` 求平均 → `mean_spectrum_factor ∈ R^{n_coeffs_factor}`
  - **输出**：
    - `{factor: mean_spectrum_factor}`，用于后续绘图。

#### 1.2.2 针对 FMA（LMD）与 ASAP 的分析函数

- **函数**：`analyze_zero_padding_for_dataset(config_name: str, factors: list[int], window_type: str | None, out_dir: str) -> None`
  - **输入**：
    - `config_name`：`"fma_main"` 或 `"asap_composer"`
    - `factors`：推荐 `[1, 2, 4]`，避免过大 factor 带来极大 FFT 长度
    - `window_type`：`None` 或 `'hann'`（与主实验保持一致）
    - `out_dir`：如 `"results/zero_padded_fft/fma_main"` 或 `"results/zero_padded_fft/asap_composer"`
  - **内部行为**：
    1. 调用 `get_embedding_and_config(config_name)` 得到 `X, y_str, cfg`
    2. 调用 `zero_pad_along_time(X, factors)` 得到 `X_padded_dict`
    3. 对整个数据集计算平均谱：
       - `mean_spectra_dict = compute_fft_magnitude(X_padded_dict, window_type)`
    4. 对特定子集做对比（可选）：
       - FMA：按 genre 过滤，例如 `["Rock", "Pop", "Electronic"]`  
       - ASAP：按作曲家过滤，例如 `["Bach", "Beethoven", "Schubert"]`  
       - 对每个子集重复 2–3 步，得到 per-genre / per-composer 的 `mean_spectra_dict`
    5. **绘图**：
       - 对每个数据集 & 子集，在**归一化频率轴**上绘制多条曲线：
         - 归一化频率 \( \omega = k / n\_coeffs\_factor \in [0, 1] \)
         - 同一张图中，画 `factor ∈ {1, 2, 4}` 的 `mean_spectrum_factor(ω)`
       - 输出 PNG，例如：
         - `zero_padded_mean_spectrum_all_window=none.png`
         - `zero_padded_mean_spectrum_Rock_window=none.png`
         - `zero_padded_mean_spectrum_Schubert_window=none.png`
    6. **保存数字结果**：
       - 将 `mean_spectra_dict` 以及对应的 `n_coeffs_factor` 序列保存为 `.npz` 或 `.json`，方便后续报告引用。

### 1.3 顶层脚本

**文件**：`scripts/run_zero_padded_fft.py`（新建）

- **接口**（伪代码示意）：

```bash
python scripts/run_zero_padded_fft.py \
  --config_names fma_main asap_composer \
  --factors 1 2 4 \
  --window_type none
```

- **参数**：
  - `--config_names`：一个或多个 embedding 配置名；
  - `--factors`：padding 因子列表；
  - `--window_type`：`none | hann | hamming | blackman`。

- **行为**：
  - 对每个 `config_name` 调用 `analyze_zero_padding_for_dataset`；
  - 结果写入：
    - `results/zero_padded_fft/fma_main/*`
    - `results/zero_padded_fft/asap_composer/*`

---

## 2. Step 2：自相关（Auto-correlation）分析（LMD & ASAP）

### 2.1 目标与原理

- **目的**：
  - 在 LMD 与 ASAP 的 **已有 embedding** 上，计算时间轴自相关；
  - 检查是否存在稳定的 **4-sample、8-sample** 周期，以及与 “8/3” 相关的非整数周期结构；
  - 验证这些周期是否在两个数据集上都存在，且与输入内容（曲目、作曲家）基本无关。

- **原理**：
  - 自相关 \( R_{xx}[\tau] = \sum\_n x[n] x[n+\tau] \) 衡量序列与自身平移后的相似度；
  - 若 embedding 在时间维上存在明显周期 \(T\)，则在 \(\tau = T, 2T, 3T, ...\) 处会出现明显正峰；
  - 对“8/3” 这类非整数周期，会在临近整数 lag 上产生一系列震荡或次峰。

### 2.2 统一的自相关工具

**文件**：`src/analysis/autocorrelation_checks.py`（新建）

#### 2.2.1 基础函数

- **函数**：`compute_autocorrelation_1d(x: np.ndarray, max_lag: int) -> np.ndarray`
  - **输入**：
    - `x ∈ R^{T}`：单条 1D 序列
    - `max_lag: int`：最大 lag（例如 `max_lag=64`）
  - **行为**：
    1. 减去均值：`x_centered = x - x.mean()`
    2. 使用 `np.correlate(x_centered, x_centered, mode="full")`
    3. 取中间到右侧 `max_lag` 部分
    4. 归一化：`rho[τ] = R[τ] / R[0]`
  - **输出**：
    - `rho ∈ R^{max_lag+1}`，`rho[0] = 1`。

- **函数**：`aggregate_autocorrelation_over_features(X: np.ndarray, max_lag: int, mode: str = "mean") -> np.ndarray`
  - **输入**：
    - `X ∈ R^{B × F × T}`：与 Step 1 中相同的 embedding
    - `max_lag`：如 `32` 或 `64`
    - `mode`：
      - `"mean"`：对 `(B, F)` 所有 `(sample, feature)` 的自相关取平均
      - `"per_feature_mean"`：先对每个 feature 在样本轴取平均，再对 feature 轴聚合
  - **行为**：
    - 遍历一部分 feature（例如全部 F，或随机选 F_subset）：
      - 对每个 `(b, f)` 取 `x = X[b, f, :]`，调用 `compute_autocorrelation_1d(x, max_lag)`
    - 对所有曲线在样本轴 / 特征轴做平均，得到一条全局平均自相关
  - **输出**：
    - `rho_mean ∈ R^{max_lag+1}`。

#### 2.2.2 LMD / ASAP 专用接口

- **函数**：`analyze_autocorrelation_for_dataset(config_name: str, max_lag: int, out_dir: str) -> None`
  - **输入**：
    - `config_name`：`"fma_main"` 或 `"asap_composer"`
    - `max_lag`：例如 `32`（ASAP，T=32）或 `64`（FMA，T=64）
    - `out_dir`：如 `"results/autocorrelation/fma_main"` 等
  - **内部行为**：
    1. 使用 `get_embedding_and_config(config_name)` 得到 `X, y_str, cfg`
    2. 计算全数据集平均自相关：
       - `rho_all = aggregate_autocorrelation_over_features(X, max_lag)`
    3. 对特定子集（genre / composer）分别计算：
       - FMA：`rho_Rock`, `rho_Pop`, `rho_Electronic` 等
       - ASAP：`rho_Schubert`, `rho_Beethoven` 等
    4. 绘图：
       - x 轴：lag（样本数）；y 轴：`rho[lag]`
       - 在图中用竖线或注释标出：
         - `lag = 4, 8, 16` 等关键位置；
         - `lag ≈ T/4, T/2, 3T/4` 对应的点。
    5. 保存数值：
       - 每条自相关曲线保存为 `.npy` 或 `.json`，包括：
         - `rho_all`, `rho_Rock`, `rho_Schubert` 等

### 2.3 顶层脚本

**文件**：`scripts/run_autocorrelation_sanity.py`（新建）

- **示例调用**：

```bash
python scripts/run_autocorrelation_sanity.py \
  --config_names fma_main asap_composer \
  --max_lag_fma 64 \
  --max_lag_asap 32
```

- **行为**：
  - 根据 `config_names` 选择合适的 `max_lag`；
  - 对每个数据集调用 `analyze_autocorrelation_for_dataset`；
  - 输出：
    - `results/autocorrelation/fma_main/*.png, *.npy`
    - `results/autocorrelation/asap_composer/*.png, *.npy`

---

## 3. Step 4：多模型对比（CLAP / PaSST / PANN-S / BEATs，输入为 LMD + ASAP）

### 3.1 目标与约束

- **目标**：
  - 在相同输入（LMD / ASAP 音频）的前提下，对比不同音频 embedding 模型在时间轴上的周期性与频域三峰结构；
  - 判断三峰/固定周期是 **CLAP 特有**，还是更普遍的 **架构级现象**。

- **约束**：
  - **输入只能使用已有的 LMD + ASAP 音频/embedding 数据**，不重新引入白噪声/零信号等新合成输入；
  - 若已有其他模型的 embedding 缓存（例如 NPZ），则直接基于这些缓存做 FFT + 自相关；
  - 若需要新抽取其他模型 embedding，抽取逻辑可与现有 CLAP pipeline 类似，但本计划只先定义接口与预期形状。

### 3.2 统一的“模型 × 数据集”表示

**文件**：`src/config/multi_model_sanity.py`（新建）

- 定义简单的数据类：

```python
from dataclasses import dataclass

@dataclass
class ModelEmbeddingSpec:
    name: str                 # 'clap', 'passt', 'pann_s', 'beats'
    dataset: str              # 'fma_main' or 'asap_composer'
    embeddings_file: str      # NPZ / NPY 路径
    feature_dim: int          # F，例如 1536 或其他
    time_length: int          # T，例如 64 / 32 / 128
```

- 提供一个字典 `MODEL_SPECS`，包含：
  - `('clap', 'fma_main')` → 现有 `clap_embeddings_t64.npz`，`F=1536`, `T=64`
  - `('clap', 'asap_composer')` → `clap_embeddings_asap_t32.npz`，`F=1536`, `T=32`
  - `('passt', 'fma_main')` → 预期的 PaSST embedding 缓存（后续由抽取脚本写入）
  - `('pann_s', 'fma_main')` → 同上
  - `('beats', 'fma_main')` → 同上

> 若目前只有 CLAP 的 embedding，可以先实现接口与 CLAP 的 path，后续扩展其他模型时无需改分析代码。

### 3.2.1 模型抽取方式与依赖管理

- **整体原则**：尽量通过 **现有 Python 包 + 轻量封装** 获取 embedding，不在本项目下复制大规模外部代码；只有当某模型没有稳定的 pip/hub 接口时，才考虑在 `external/` 或 `third_party/` 下 vendor 少量源码。

- **CLAP**：继续使用当前已实现的抽取与 NPZ 缓存，不再改动。

- **PaSST（推荐依赖：`hear21passt`）**
  - 通过 PyPI 包 `hear21passt` 加载模型并提取 embedding，无需复制源码：
    - 安装：在环境或 `requirements.txt` 中加入：
      ```txt
      hear21passt
      ```
    - 使用（示意）：
      ```python
      from hear21passt.base import load_model, get_timestamp_embeddings
      model = load_model().cuda()
      embed, ts = get_timestamp_embeddings(audio_tensor, model)
      ```
  - 在本项目中，仅在 `src/features/multi_model_embeddings.py` 中提供一个 `get_passt_embeddings(...)` 封装，内部处理音频加载、模型前向与保存到 NPZ，不在其他模块直接依赖 `hear21passt`。

- **BEATs（推荐依赖：SpeechBrain）**
  - 使用 SpeechBrain 中的 `speechbrain.lobes.models.beats` 封装和官方 checkpoint：
    - 在环境/`requirements.txt` 中加入：
      ```txt
      speechbrain
      ```
    - 在 `get_beats_embeddings(...)` 中通过 SpeechBrain 提供的接口加载模型与 checkpoint，抽取 `(B, F, T)` 形式的 embedding 并写入 NPZ。
  - 同样只在 `multi_model_embeddings.py` 中集中依赖 SpeechBrain。

- **PANN-S**
  - 优先使用 **torch hub / 轻量 wrapper** 的方式加载官方 PANN-S 模型：
    - 若存在稳定的 pip 包，则与 PaSST/BEATs 相同，直接通过包依赖；
    - 若只能通过 GitHub 源码使用，则在本项目中新建一个极小的 `external/panns/` 文件夹，仅放入：
      - 必要的模型定义文件（如 `cnn14.py`）；  
      - 一个下载 checkpoint 的小脚本；  
      - 不复制整个官方 repo。
  - 无论具体实现如何，对外接口仍然是 `get_panns_embeddings(...)`，由 `multi_model_embeddings.py` 负责调用。

### 3.2.2 统一的多模型 embedding 抽取入口

**文件**：`src/features/multi_model_embeddings.py`（新建）

- 提供统一高层接口：

```python
from typing import Literal, Sequence
import numpy as np

ModelName = Literal["clap", "passt", "pann_s", "beats"]

def extract_and_save_embeddings(
    model: ModelName,
    dataset: str,
    file_list: Sequence[str],
    out_path: str,
) -> None:
    """
    对给定数据集的音频文件，用指定模型抽取 embedding 并保存到 NPZ。

    保存格式约定：
        - 'embeddings': np.ndarray, shape (B, F, T)
        - 'labels':     np.ndarray[str] (如 flow/genre/composer)
        - 可选附加字段: 'file_paths' 等
    """
    ...
```

- 内部根据 `model` 分派到具体实现：
  - `get_clap_embeddings(...)`：复用当前 CLAP pipeline；
  - `get_passt_embeddings(...)`：基于 `hear21passt`；
  - `get_panns_embeddings(...)`：基于 torch hub 或 `external/panns/`；
  - `get_beats_embeddings(...)`：基于 SpeechBrain。

- **代码管理约定**：
  - 所有与“如何从音频得到 (B, F, T)”相关的逻辑仅出现在 `src/features/` 下；
  - `src/config/multi_model_sanity.py` 只记录对应模型的 NPZ 路径与 `(F, T)`；
  - `src/analysis/*` 不直接依赖具体模型库，只读已经生成好的 NPZ。

### 3.2.3 PaSST 抽取任务的具体执行步骤

鉴于当前 CLAP 已经完成，这一小节单独规划 **PaSST** 的 embedding 抽取与落盘流程，方便直接接入多模型对比。

**总体目标**：  
在 FMA / ASAP 上，使用 PaSST 抽取与 CLAP 同类的 `(B, F, T)` 形式 embedding，并写入：

- `data_artifacts/passt_embeddings_t64.npz`（FMA 主任务，对应 `fma_main`）
- 如有需要，也可为 ASAP 抽取 `data_artifacts/passt_embeddings_asap_t32.npz`（对应 `asap_composer`）

两者形状约定为：

- FMA：`(B_fma, F_passt, T_passt)`，其中 `T_passt` 尽量与 CLAP 的 64 帧在“时间分辨率等级”上可比（不必完全相等，但要在文档中说明）。
- ASAP：`(B_asap, F_passt, T_passt_asap)`，同理。

#### A. 依赖与环境

- 在环境中确保安装：

```txt
torch
hear21passt
librosa
soundfile
numpy
```

> 不在本项目内 vendor PaSST 源码，优先通过 `hear21passt` 官方包加载模型。

#### B. FMA 上的 PaSST 抽取（建议新建脚本）

**新文件**：`scripts/extract_passt_fma_embeddings.py`

- **输入**：
  - FMA 音频文件列表及对应的 genre 标签；
    - 如果已有 CLAP 抽取用的 CSV/NPZ 中包含 `file_paths` 与 `genres`，则直接读取；
    - 否则在该脚本中以与 CLAP 一致的方式收集 FMA 路径与标签。

- **核心步骤**（逻辑）：

```python
from hear21passt.base import load_model, get_timestamp_embeddings
import torch
import librosa
import numpy as np

def extract_passt_for_file(path: str, model, sr: int = 32000) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sr, mono=True)
    # model 期望输入形状，可参考 hear21passt 文档（一般为 (B, T)）
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # (1, T)
    with torch.no_grad():
        emb, _ = get_timestamp_embeddings(audio_tensor, model)  # (1, T_passt, D)
    # 转成 (D, T_passt)
    emb_np = emb.squeeze(0).cpu().numpy().T
    return emb_np  # (F_passt, T_passt)
```

- **聚合与落盘**：
  - 遍历所有 FMA 文件，调用 `extract_passt_for_file` 收集：
    - `emb_list: List[np.ndarray]`，每个元素形状 `(F_passt, T_passt_i)`；
    - `labels: List[str]`，对应 genre；
    - `file_paths: List[str]`。
  - 为了与 CLAP FFT 轴对齐，建议：
    - 选用固定 PaSST 模式（例如 `hear21passt.base` 的默认 10–15 秒 window）；
    - 若不同文件得到的 `T_passt_i` 不同，可：
      - 要么裁剪/填充到统一长度 `T_passt`（例如最短长度的下界或中位数），只在**时间维末尾补零**；
      - 要么记录 `T_passt_i` 并在后续 `MODEL_SPECS` 中单独为 PaSST 记录一个 `time_length_passt`。
  - 最终堆叠成：

```python
embeddings = np.stack(emb_list, axis=0)  # (B_fma, F_passt, T_passt)
np.savez(
    "data_artifacts/passt_embeddings_t64.npz",
    embeddings=embeddings,
    labels=np.asarray(labels),
    file_paths=np.asarray(file_paths),
)
```

- **与配置联动**：
  - 在 `src/config/multi_model_sanity.py` 中更新：

```python
("passt", "fma_main"): ModelEmbeddingSpec(
    model="passt",
    dataset="fma_main",
    embeddings_file="data_artifacts/passt_embeddings_t64.npz",
    feature_dim=F_passt,      # 实际的通道数
    time_length=T_passt,      # 实际时间帧数
),
```

  - 确保 `feature_dim` 与 `time_length` 与 NPZ 中 `embeddings.shape[1:3]` 一致。

#### C. ASAP 上的 PaSST 抽取（可选）

如果希望在 ASAP/composer 任务上也进行 PaSST 对比，可以类似地新增：

- 脚本：`scripts/extract_passt_asap_embeddings.py`
  - 从 `/home/evev/noiseloss/datasets/D_asap_100` 读取 WAV；
  - 对每个 composer 抽取 PaSST embedding；
  - 写入 `data_artifacts/passt_embeddings_asap_t32.npz`（或其它命名）。
- 在 `MODEL_SPECS` 中添加：

```python
("passt", "asap_composer"): ModelEmbeddingSpec(
    model="passt",
    dataset="asap_composer",
    embeddings_file="data_artifacts/passt_embeddings_asap_t32.npz",
    feature_dim=F_passt,
    time_length=T_passt_asap,
),
```

#### D. 与多模型对比脚本的衔接

完成 PaSST 抽取并填好 `MODEL_SPECS` 后，无需改动分析代码，只需要运行：

```bash
python scripts/run_multi_model_sanity.py \
  --datasets fma_main asap_composer \
  --models clap passt \
  --factors 1 2 \
  --window_type none \
  --max_lag_fma 64 \
  --max_lag_asap 32 \
  --n_features_sample 128
```

即可在：

- `results/multi_model_sanity/fma_main/`  
- `results/multi_model_sanity/asap_composer/`  

下得到 CLAP vs PaSST 的 **zero-padded FFT 对比图** 和 **自相关对比图**，后续只需在报告中解释“PaSST 是否也呈现类似三峰/周期结构，还是谱形明显不同”。

### 3.3 多模型频谱与自相关分析函数

**文件**：`src/analysis/multi_model_periodicity.py`（新建）

#### 3.3.1 统一加载与 reshape

- **函数**：`load_model_embeddings(spec: ModelEmbeddingSpec) -> tuple[np.ndarray, np.ndarray]`
  - **输入**：
    - `spec`：包含 `embeddings_file`, `feature_dim`, `time_length`
  - **行为**：
    - 读取 NPZ/NPY，得到原始 `embeddings` 与标签 `y`（genre 或 composer）
    - 将 embeddings reshape / transpose 到统一形状：
      - `X ∈ R^{B × F × T_spec}`，其中 `F=spec.feature_dim`, `T_spec=spec.time_length`
  - **输出**：
    - `X: np.ndarray (B, F, T_spec)`
    - `y: np.ndarray[str]`

#### 3.3.2 频域对比

- **函数**：`compare_fft_across_models(dataset: str, model_names: list[str], factors: list[int], window_type: str | None, out_dir: str) -> None`
  - **输入**：
    - `dataset`：`"fma_main"` 或 `"asap_composer"`
    - `model_names`：例如 `["clap", "passt", "pann_s", "beats"]`
    - `factors`：与 Step 1 一致的 zero padding 因子列表
    - `window_type`：FFT window 类型
  - **行为**：
    - 对于每个 `model_name`：
      1. 根据 `MODEL_SPECS[(model_name, dataset)]` 加载 `X, y`
      2. 对 `X` 调用 `zero_pad_along_time` 和 `compute_fft_magnitude`（复用 Step 1 的工具）
      3. 得到 `mean_spectra_dict_model[factor]`
    - 按 **归一化频率轴** 绘图：
      - 同一张图中比较不同模型在 `factor=1` 下的平均谱；
      - 可选：对 `factor > 1` 只画 CLAP，以减少图像复杂度；
    - 保存：
      - `fft_mean_spectrum_<dataset>.png`
      - `fft_mean_spectrum_<dataset>_<model>.npy`

#### 3.3.3 自相关对比

- **函数**：`compare_autocorrelation_across_models(dataset: str, model_names: list[str], max_lag: int, out_dir: str) -> None`
  - **行为**：
    - 对每个 `model_name`：
      1. 加载 `X, y`
      2. 计算 `rho_model = aggregate_autocorrelation_over_features(X, max_lag)`
    - 将所有模型的 `rho_model` 画在同一张图上：
      - x 轴：lag；y 轴：自相关值；
      - 标注 `lag=4, 8, ...` 等关键位置；
    - 输出：
      - `autocorr_mean_<dataset>.png`
      - `autocorr_mean_<dataset>_<model>.npy`

### 3.4 顶层脚本

**文件**：`scripts/run_multi_model_sanity.py`（新建）

- **示例调用**：

```bash
python scripts/run_multi_model_sanity.py \
  --datasets fma_main asap_composer \
  --models clap passt pann_s beats \
  --max_lag_fma 64 \
  --max_lag_asap 32 \
  --factors 1 2
```

- **行为**：
  - 对每个数据集调用：
    - `compare_fft_across_models(dataset, models, factors, window_type=None, out_dir=...)`
    - `compare_autocorrelation_across_models(dataset, models, max_lag_dataset, out_dir=...)`
  - 结果目录示例：
    - `results/multi_model_sanity/fma_main/*`
    - `results/multi_model_sanity/asap_composer/*`

---

## 4. 结果组织与后续报告接口

### 4.1 结果目录建议

- `results/zero_padded_fft/`
  - `fma_main/zero_padded_mean_spectrum_all_*.png, *.npy`
  - `asap_composer/zero_padded_mean_spectrum_all_*.png, *.npy`
- `results/autocorrelation/`
  - `fma_main/autocorr_all.png, autocorr_Rock.png, ...`
  - `asap_composer/autocorr_all.png, autocorr_Schubert.png, ...`
- `results/multi_model_sanity/`
  - `fma_main/fft_mean_spectrum_*.png, autocorr_mean_*.png, *.npy`
  - `asap_composer/fft_mean_spectrum_*.png, autocorr_mean_*.png, *.npy`

### 4.2 报告中可直接引用的信息

- 每个实验函数都应在保存 `.npy` / `.json` 时包含：
  - 使用的 `config_name / dataset / model_name`
  - `n_fft`, `n_coeffs`, `padding_factor`
  - 对应的 `lags` 或 `normalized_frequency` 序列
- 报告撰写时，可以直接从这些文件中读取：
  - “在 FMA / ASAP 上，半谱长三峰在 `padding_factor=1,2,4` 下始终出现在归一化频率 ~0.25, 0.5, 0.75”；
  - “在 CLAP vs PaSST/PANN-S/BEATs 的平均自相关中，只有 CLAP 在 lag=4,8 等位置有强峰” 等定量描述。

---

## 5. 本计划与前两轮 Plan 的关系

- **继承**：
  - 完全复用 `EmbeddingConfig`、`loader.load_data`、`apply_transform`、`peak_artifact_checks` 已有逻辑；
  - 不修改已有脚本行为，所有新实验通过新增 `src/analysis/*.py` + `scripts/run_*.py` 实现。
- **删减 / 不再执行的部分**：
  - 本轮 **不再新增** 白噪声 / 静音 / 正弦等合成输入测试（原 Step 3），仅在最终报告中引用已有结果；
  - 不做多层 hidden states probing，不做复杂的高维可视化扩展（这些在 `2nd_analysis_plan.md` 中已有规划）。

> 最终，这个 `3rd_sanity_check_plan` 旨在提供一套 **工程化、可复现** 的脚本规划，让任何人按照这里的文件名和函数签名，就可以完整跑完第三次 sanity check 所需的 zero padding、自相关和多模型对比实验。

