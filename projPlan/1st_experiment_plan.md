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

## 6) 后续阶段（本阶段完成后再做）

完成本计划后，再进入模型扩展：

1. `AudioMAE`
2. `AudioMAE FT`

扩展时沿用同样的“每类必须有 png + metrics json”的交付规范。
