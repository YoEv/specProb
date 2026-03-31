# 1st Experiment Plan: PaSST FFT Results Completion

## 0) 范围收敛（只做两件事）

本阶段只完成以下两个结果目录，不扩展其他分析：

1. `results/passt_fft_composer_specific_analysis`
2. `results/passt_fft_genre_specific_analysis`

目标是把 **图和指标文件做完整**，确保每个类别都有可用产物。

---

## 1) 已确认前提

- PaSST 无周期性已经是结论，本计划不再复核该问题。
- 本计划只关注：`PaSST FFT` 在 composer / genre 两条线上的完整产出。

---

## 2) 产物标准（必须满足）

对每个 composer / genre，都必须至少有：

1. 一张图（`*.png`）
2. 一份指标文件（`*_metrics.json`）

以 composer 为例，目标格式应与已有样例一致：

- `results/passt_fft_composer_specific_analysis/passt_asap_spectral_summary_Bach_fft.png`
- `results/passt_fft_composer_specific_analysis/passt_asap_spectral_summary_Bach_fft_metrics.json`

genre 目录同理，命名规则保持一致。

---

## 3) 执行步骤

## Step A — Composer 结果补齐

- 检查 `results/passt_fft_composer_specific_analysis` 中是否每个 composer 都同时具备：
  - summary 图；
  - metrics json。
- 对缺失 composer 重新运行对应脚本生成；
- 完成后输出一个清单：`composer, has_png, has_metrics_json`。

## Step B — Genre 结果补齐

- 检查 `results/passt_fft_genre_specific_analysis` 中是否每个 genre 都同时具备：
  - summary 图；
  - metrics json。
- 对缺失 genre 重新运行对应脚本生成；
- 完成后输出一个清单：`genre, has_png, has_metrics_json`。

## Step C — 结果一致性检查

- 文件命名是否统一（避免同类文件多种命名）；
- json 字段是否一致（可比较）；
- 图像是否可读（无空白图、无损坏图）。

---

## 4) 验收标准

通过标准只有一个：  
**两个目录内，每个类别都同时有 png + metrics json，并且命名和字段统一。**

---

## 5) 交付物

1. `results/passt_fft_composer_specific_analysis`（完整）
2. `results/passt_fft_genre_specific_analysis`（完整）
3. 一个简短完成报告（可写在 `analysis_report.md` 新增小节）：
   - 总类别数；
   - 缺失并补齐的类别数；
   - 最终完整率（应为 100%）。

---

## 6) 后续阶段：AudioMAE / AudioMAE FT 详细搭建计划

本节目标：在当前 `beluga` 分支独立完成 AudioMAE 环境与推理链路，不依赖其他分支。

### 6.1 官方信息（用于约束实施）

基于官方 `facebookresearch/AudioMAE` README：

- 官方依赖环境偏旧（`python=3.8`, `pytorch=1.7`, `timm=0.3.2`）；
- 需要执行 `timm_patch.sh`；
- 官方提供了预训练与微调后的 checkpoint；
- 官方 inference 入口是 `inf.sh ckpt/finetuned.pth`（面向 AudioSet 评估）。

这意味着：如果直接复刻官方训练/评估脚本，推荐隔离环境，不要污染现有实验环境。

---

### 6.2 环境策略（推荐双轨）

#### Plan A（推荐）：隔离官方兼容环境

用途：先保证 AudioMAE 能稳定跑通，风险最低。

- 新建 conda 环境：`audiomae`
- 在该环境中安装官方依赖（按 `mae_env.yml` 或精简复现）
- 将 AudioMAE 代码放到 `third_party/AudioMAE`

#### Plan B（可选）：项目主环境内最小依赖接入

用途：减少环境切换，但兼容风险更高。

- 只安装推理所需最小依赖；
- 若遇到 `timm`/老 API 兼容问题，回退到 Plan A。

执行顺序建议：先 A，后 B。

---

### 6.3 安装步骤（Plan A）

1. 克隆代码（仅放第三方目录）  
   `third_party/AudioMAE`
2. 新建并激活环境（建议 `python=3.8`）  
3. 安装依赖（优先 conda + pip，必要时参考 `mae_env.yml`）  
4. 执行 `timm_patch.sh`（按官方要求修补 timm）  
5. 下载 checkpoint 到：`third_party/AudioMAE/ckpt/`

建议把 checkpoint 路径抽成环境变量，避免脚本写死绝对路径。

---

### 6.4 冒烟测试（必须先通过）

在正式抽取 embedding 前，先做 3 个最小验证：

1. **Import test**：关键模块可导入；
2. **Single file forward**：单个 wav 可前向；
3. **GPU test**：模型与输入确实在 CUDA 上（若可用）。

只要这三项中有一项失败，不进入全量提取。

---

### 6.5 接入当前项目的数据产物规范

目标对齐你现有 NPZ 规范（和 PaSST/BEATs 一致）：

- FMA 输出：`data_artifacts/audiomae_embeddings_t64.npz`
  - keys: `embeddings`, `genres`, `file_paths`
- ASAP 输出：`data_artifacts/audiomae_embeddings_asap_t32.npz`
  - keys: `embeddings`, `composers`, `file_paths`

新增脚本（命名建议）：

1. `scripts/extract_audiomae_fma_embeddings.py`
2. `scripts/extract_audiomae_asap_embeddings.py`

---

### 6.6 AudioMAE FT 计划

FT 和 base 模型共用同一 extraction 脚本，仅切换 checkpoint：

- base checkpoint -> 产出 `audiomae_embeddings_*.npz`
- finetuned checkpoint -> 产出 `audiomae_ft_embeddings_*.npz`

命名建议：

- `data_artifacts/audiomae_ft_embeddings_t64.npz`
- `data_artifacts/audiomae_ft_embeddings_asap_t32.npz`

---

### 6.7 接入你当前 FFT probing 流水线

复用已有脚本：

- `scripts/run_generic_fft_band_probe.py`

只替换参数：

- `--npz_path` 指向 AudioMAE / AudioMAE FT 的 NPZ
- `--label_key` 使用 `genres` 或 `composers`
- `--results_dir` 使用新目录：
  - `results/audiomae_fft_genre_specific_analysis`
  - `results/audiomae_fft_composer_specific_analysis`
  - `results/audiomae_ft_fft_genre_specific_analysis`
  - `results/audiomae_ft_fft_composer_specific_analysis`

---

### 6.8 交付物与验收

最低交付：

1. 可复现安装文档（环境 + checkpoint + 测试）
2. 4 个 NPZ（AudioMAE / AudioMAE FT × FMA / ASAP）
3. 4 组结果目录（每类 `png + metrics.json`）

验收标准：

- 脚本可从空环境复现安装；
- AudioMAE 与 AudioMAE FT 都能产出 NPZ；
- 结果目录完整率 100%（同 PaSST 标准）。
