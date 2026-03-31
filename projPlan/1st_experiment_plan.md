# 1st Experiment Plan: 跨模型表征趋势与 Genre 解释（Post-3rd-Sanity）

## 0) 背景与目标

我们已经完成 3 次 sanity check。下一步不再只看单模型单现象，而是进入 **跨模型（CLAP / PaSST / BEATs）+ 全 genre 的趋势总结阶段**，回答四个核心问题：

1. **PaSST no periodicity?**  
   PaSST 是否真的没有明显周期性结构，还是因为时间长度和尺度不同导致表象差异？
2. **Complete prediction results**  
   所有 genre 的完整预测结果（不仅 accuracy）是否支持稳定趋势？
3. **We gain?**  
   频域 probing 相比只看原始 embedding，带来了什么可量化收益？
4. **What does this mean?**  
   这些趋势在音乐语义上怎么解释，特别是 **tempo / 节奏相关性** 在 CLAP 与 PaSST 中是否一致？

---

## 1) 已有结果的快速现状（用于设定基线）

从当前 `results/` 可见：

- **CLAP (FMA, one-vs-rest genre)**  
  平均准确率约 `0.9084`（8 genre，min `0.8525`，max `0.9425`）。
- **PaSST (FMA, one-vs-rest genre)**  
  平均准确率约 `0.7289`（min `0.6813`，max `0.8500`）。
- **BEATs (FMA, one-vs-rest genre)**  
  平均准确率约 `0.8633`（min `0.8000`，max `0.9500`）。
- CLAP 已有 `bpm_correlation.txt`，目前全局相关性较弱（r 接近 0），但只做了 CLAP，**PaSST 还没对齐跑 tempo 检查**。

> 结论：现在最缺的是 **统一尺度上的多模型趋势对齐 + 可解释输出**，而不是再做零散单点图。

---

## 2) 第一阶段总原则（本 plan 的边界）

本 `1st_experiment_plan` 只做一件事：  
**把“多模型-全genre-周期性-tempo-解释”串成一条可复现流水线**。

不在这一阶段做：

- 新模型大规模重训练；
- 新数据集扩充；
- 复杂架构改动。

---

## 3) 研究问题与可检验假设

### RQ1: PaSST 是否“无周期性”？

- **H1**: 在归一化频率轴 + 自相关下，PaSST 的周期峰显著弱于 CLAP。  
- **H1-alt**: PaSST 仍有周期性，但峰位/峰宽不同（可能由 `T=601` 与 patch 结构引起）。

### RQ2: 全 genre 的趋势是否跨模型一致？

- **H2**: “更节奏驱动”的 genre（如 Hip-Hop）在多个模型上都表现更稳定；  
  但模型间 absolute accuracy 与 band 依赖会不同。

### RQ3: 频域 probing 的收益是什么？

- **H3**: 相比 full embedding，band/cumulative 提供更强解释性；在部分 genre 上有 accuracy 增益（或接近无损）。

### RQ4: tempo 到底解释了多少？

- **H4**: tempo 与峰值能量相关性整体不强，但某些 genre 或某些模型会出现局部显著关系。

---

## 4) 实验设计（按执行顺序）

## Step A — Consolidate 全部预测结果（必须先做）

### A1. 统一产物格式（CLAP / PaSST / BEATs）

- 输入：
  - `results/fft_genre_specific_analysis/*.json`（CLAP）
  - `results/passt_spectral_fma/passt_fma_spectral_summary_genre.json`
  - `results/beats_spectral_fma/beats_fma_spectral_summary_genre.json`
  - `results/passt_band_spectral_fma/*.json`
  - `results/beats_band_spectral_fma/*.json`
- 目标：
  - 产出统一表格（CSV/JSON）字段：
    - `model`, `genre`, `full_acc`, `best_band_acc`, `auto_acc`, `gain_auto_vs_full`, `n_coeffs`, `T`, `F`
- 输出建议：
  - `results/consolidated/model_genre_trends.csv`
  - `results/consolidated/model_genre_trends.json`

### A2. Complete prediction results（不仅 accuracy）

- CLAP：继续使用现有 one-vs-rest流程，同时补充 confusion 结果：
  - `python scripts/run_probe_confusion_analysis.py --transform-type fft --n-bands 8`
- PaSST / BEATs：
  - 当前脚本主要给 one-vs-rest accuracy 和 profile，建议补一个“generic confusion”脚本（复用 `run_probe_with_predictions`），至少保存：
    - `y_true`, `y_pred`, `class_names`, confusion matrix。
- 输出建议：
  - `results/consolidated/confusion_<model>_<dataset>.json`

> 完成 A 之后，才可以回答 “re-evaluate for all genres - do we see trends?”。

---

## Step B — PaSST periodicity 复核（核心）

### B1. 多模型同轴对比（归一化频率 + autocorr）

运行：

```bash
python scripts/run_multi_model_sanity.py \
  --datasets fma_main \
  --models clap passt beats \
  --factors 1 2 4 \
  --window_type none \
  --max_lag_fma 128 \
  --n_features_sample 128
```

重点看：

- `fft_mean_spectrum_*_all_factors*.png`
- `autocorr_mean_fma_main.png`

### B2. PaSST no-periodicity 判定标准（先定规则，再看图）

定义 3 个量化指标：

1. **Autocorr Peak Ratio**: `max(rho[lag>0]) / rho[0]`
2. **Peak Concentration**: top-k FFT 峰值占比（在归一化频率上）
3. **Permutation Drop**: 时间置换后峰值结构下降比例（可复用已有 permutation 思路）

判定：

- 若 PaSST 在 3 指标均显著低于 CLAP，可称 “PaSST weak periodicity / near no periodicity”；
- 否则应写成 “PaSST periodicity pattern differs from CLAP”，避免绝对化表述。

---

## Step C — Tempo / 平均节奏解释（CLAP + PaSST）

### C1. CLAP 复跑并固定口径

```bash
python scripts/run_bpm_correlation.py
```

保留：

- 全局相关；
- 分 genre 相关；
- 每个 genre 的 `n` 和显著性（p-value）。

### C2. PaSST 对齐 tempo

新增一个 PaSST 版本的 bpm correlation（建议复用 `src/analysis/bpm_correlation.py` 逻辑）：

- 关键差异：
  - CLAP 用 `track_ids` 对齐 metadata；
  - PaSST 用 `file_paths` 对齐 metadata（当前 NPZ 不含 track_id）。
- 输出：
  - `results/peak_artifact_investigation/passt_bpm_correlation.txt`

### C3. “exact average tempo per class” 报告口径

对每个 genre 产出：

- `mean_bpm`, `std_bpm`, `median_bpm`, `n_tracks`
- `corr(bpm, peak_energy_k)`（k为主要峰位）

输出建议：

- `results/consolidated/tempo_per_genre_model.csv`

---

## Step D — “We gain?” 的量化定义（必须回答）

每个模型、每个 genre 报告 3 个 gain：

1. `gain_auto_vs_full = auto_acc - full_acc`
2. `gain_best_band_vs_full = best_band_acc - full_acc`
3. `interpretability_gain`（非 accuracy）：
   - 可用 “少量 band 即达到 95% full_acc” 的 band 数（越小越可解释）

建议最终做一个简明表：

- 行：genre
- 列：`CLAP_gain`, `PaSST_gain`, `BEATs_gain`
- 再加总体均值和标准差

---

## Step E — “What does this mean?” 的解释模板（写报告时直接用）

每个 genre 用固定四句模板，避免解释飘：

1. **Performance**：该 genre 在各模型准确率排序；
2. **Spectral trend**：主要依赖低/中/高频段；
3. **Periodicity**：是否出现稳定周期峰；
4. **Tempo link**：tempo 与峰能量关系强弱（及显著性）。

最终给跨模型结论：

- 哪些趋势是 **模型无关**（稳定规律）；
- 哪些趋势是 **模型特有**（架构偏置）。

---

## 5) 具体执行清单（可直接打勾）

- [ ] 跑并汇总 CLAP / PaSST / BEATs 的全 genre 指标到 `results/consolidated/`
- [ ] 补齐 PaSST / BEATs confusion 级别输出（完整预测）
- [ ] 跑 `run_multi_model_sanity.py` 完成 periodicity 对比
- [ ] 计算并落盘 PaSST 的 bpm correlation
- [ ] 生成 `tempo_per_genre_model.csv`
- [ ] 产出 “gain” 总表与排序图
- [ ] 写 1 页结论（模型共性 vs 模型特性）

---

## 6) 交付物（第一阶段结束标准）

最少交付：

1. `results/consolidated/model_genre_trends.csv`
2. `results/consolidated/tempo_per_genre_model.csv`
3. `results/multi_model_sanity/fma_main/*.png`（FFT + autocorr）
4. `results/peak_artifact_investigation/passt_bpm_correlation.txt`
5. `analysis_report.md` 新增一节：`Cross-model trend + tempo interpretation`

验收标准：

- 能明确回答：
  - PaSST 是否 truly no periodicity；
  - 全 genre 是否存在稳定跨模型趋势；
  - 频域 probing 的实际 gain 是什么；
  - tempo 在解释中占多大比重。

---

## 7) 风险与规避

- **风险 1**：模型时间长度差异（CLAP 64, PaSST 601, BEATs 527）造成“假比较”。  
  **规避**：统一使用归一化频率轴 + 相同统计指标，不直接比绝对 bin index。

- **风险 2**：PaSST/BEATs 缺少 track_id。  
  **规避**：基于 `file_paths` 构建 metadata 映射，先做对齐一致性检查。

- **风险 3**：只看平均曲线导致误判。  
  **规避**：至少保留 per-genre + per-model 的分布统计（mean/std + sample count）。

---

## 8) 本阶段一句话结论目标

从“我们看到了几个图”升级到：  
**我们能定量说明不同音频表征模型如何编码 genre，PaSST 是否缺乏周期性，以及 tempo 在这种编码中到底扮演什么角色。**
