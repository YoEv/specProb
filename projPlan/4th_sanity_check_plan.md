# 第 4 次 Sanity Check 计划：多分类 per-band probe（PaSST，GPU-only）

## 0. 本轮要解决的问题

- 当前 `results/passt_fft_genre_specific_analysis_gpu_small_t64/` 与 `results/passt_fma_medium_gpu_by_shard/` 下所有 `*_spectral_summary_<Class>_fft_metrics.json` / `*_feature_weight_*` 都是 **one-vs-rest 二分类** probe 的产物（`y_binary = (y == target).astype(int)`）：每个 genre 一个独立的二分类 probe，chance 线 ≈ 该 genre 的正样本比例。
- 本轮把两个 PaSST 目录切换到 **多分类 per-band probe**：每个 band 只用这一个 band 的 FFT 系数，在完整 genre 标签空间（fma_small = 8 类；fma_medium 每个 shard 内出现的那些类）上训练一个线性 softmax probe。
- **范围限定**：只改 PaSST 两个目录；ASAP / CLAP / composer 相关产物不动。
- **backend 限定**：**只做 GPU（torch）路径**；CPU / sklearn 路径不在本轮实现，以便保持代码简洁、单一实现。

## 1. 关键设计决策（定死）

- **probe 形式**：GPU 上 `torch.nn.Linear(D, C)` + `CrossEntropyLoss(weight=class_weights)`；`class_weights` 用 sklearn 风格预计算 `w_c = N / (C * n_c)`，在训练集上算一次，传给 CE。优化器 AdamW，`lr=0.05`、`weight_decay=1e-4`、默认 epochs=200（复用当前 GPU 二分类的超参），split 用 `train_test_split(test_size=0.2, stratify=y_int, random_state=42)`，特征沿训练集做 StandardScaler。
- **指标结构（和二分类版同名，读者无感切换）**：
  - `full_embedding` = 全谱多分类 overall accuracy。
  - `band_specific[b]` = 只用 band b 的多分类 overall accuracy。
  - `cumulative_per_band[k]` = 用 B0..Bk 的多分类 overall accuracy。
  - `AUTO_CMAX = max(cumulative_per_band)`；`AUTO_TR` 按 `best_cumulative_prefix` 或 `topk_bands` 再单训一次多分类 probe（与二分类版完全同构）。
  - 辅助字段：`per_class_recall` = 每个 probe 的每类 recall（`full_per_class_recall`、`band_per_class_recall[b]`、`cumulative_per_class_recall[k]`），用于替代原本二分类"Pop/Rock 各自 accuracy"的信息。
- **chance baseline**：图里画两条虚线，`1/n_classes`（均衡基线）+ `max_class_prop`（多数类基线）。
- **spectral profile（learned weight）**：
  - `torch.nn.Linear(D, C).weight` → `(C, D=F*K)`；统一 `abs().reshape(C, F, K)`。
  - **overall**：`profile[k] = mean_c mean_f |W[c, f, k]|`，min-max 归一到 [0,1]，用作主 summary 图右图。
  - **per-class**：`profile_c[k] = mean_f |W[c, f, k]|`，单独 min-max 归一，供每 class 一张 `_feature_weight_mean_std_<Class>_fft.png`。
- **图表决定（per-class 图只画 spectral profile，不画 accuracy/recall bar）**：
  - 在 multiclass 设定下，`band_specific[b]` 是一个全局多分类 accuracy，没有 per-class 对应物；旧二分类版那张"Pop 的 bar"在新设定里没有直接对应的物理量。
  - 因此 per-class 图只保留 **右半边**：`per_class_spectral_profile[cls]` 的 mean curve + mean±std（std 跨 F 维）。文件名继续用 `<prefix>_multiclass_feature_weight_mean_std_<Class>_fft.{png,json,csv}`，便于和 `*_ovr_legacy/` 的旧 per-genre 图并排比较"权重峰位置"这一点。
  - `band_per_class_recall[b][cls]` 仍然**写进 JSON**（来自同一个多分类 probe 的测试集 per-class recall），不画图；将来需要 "band × class recall 热力图" 时一行脚本就能出。
- **文件命名**（避免和旧产物混）：
  - 新目录统一加 `_multiclass` 后缀：
    - `results/passt_fma_small_multiclass_fft/`
    - `results/passt_fma_medium_multiclass_fft_by_shard/`（下一轮）
  - 主 summary：`<prefix>_multiclass_spectral_summary_<label_key>_fft.{json,png}`（每个 run 只出 1 份，不再按 class 拆）。
  - per-class profile：`<prefix>_multiclass_feature_weight_mean_std_<Class>_fft.{png,json,csv}`（N 份，对应 N 个 class）。
- **脚本入口**：改造 `scripts/run_generic_fft_band_probe.py`，新增 `--label_mode {binary,multiclass}`，默认 `binary`（保证现有 CLAP / composer 运行方式零破坏）。本轮运行统一 `--label_mode multiclass --backend gpu`。
  - 当 `--label_mode multiclass --backend cpu` 时直接 `parser.error(...)` 报错，明确告诉用户本轮只做 GPU。
  - `--target_label` 在 multiclass 模式下被忽略（打印 warning），保留只是为了兼容旧命令行。

## 2. 代码改动清单

### 2.1 要改

- `scripts/run_generic_fft_band_probe.py`
  - 新增 `--label_mode`；解析后在 multiclass 分支：
    - `LabelEncoder` 编码 `y_str → y_int`（`np.int64`），记录 `classes = list(le.classes_)`、`class_counts`。
    - `n_classes = len(classes)`；若 `n_classes < 2` 直接退出报错。
    - 用训练集算 `class_weights ∈ R^{C}`（sklearn 风格），转 `torch.tensor(..., device=device)`。
  - 新增函数（全部 GPU）：
    - `_run_probe_gpu_multiclass(X, y_int, class_weights, n_classes, ...) -> (acc, per_class_recall, _TorchLinearProbeModel)`
    - `_run_probe_gpu_batched_same_dim_multiclass(X_tasks, y_int, class_weights, n_classes, ...) -> (accs, per_class_recalls, models)`  
      - W: `(K, D, C)`，bias `(K, C)`；logits `einsum("knd,kdc->knc", X_train_t_batched, W) + b.unsqueeze(1)` 得到 `(K, N, C)`；逐任务 CE loss（用 `F.cross_entropy` 的 `reduction="none"` 再 reshape，或者手写 log-softmax + NLL），对 K 维求 mean 做 backward。
    - `_run_probe_gpu_batched_prefix_multiclass(X_full, prefix_lengths, y_int, class_weights, n_classes, ...) -> (accs, per_class_recalls, models)`  
      - W: `(K, D_full, C)`，训练时对 W 应用 `(K, D_full, 1)` 的 prefix mask，其他同上。
  - 在 `main()` 中：
    - multiclass 分支调用上面 3 个 GPU 函数，填 `full_embedding` / `band_specific` / `cumulative_per_band`；顺手填 `full_per_class_recall` 等字段。
    - 旧二分类分支逻辑原样保留。
  - `_TorchLinearProbeModel` 扩展：允许 `coef_` 形状为 `(C, D)`（多分类）或 `(1, D)`（二分类），下游 profile 函数按 `n_classes` 分派。
  - 绘图分支（multiclass）：
    - **主 summary 图（1 张/run）**：左 bar ORIG/B0…/AUTO_CMAX/AUTO_TR 的 overall multiclass accuracy，chance 双虚线（`1/n_classes` + `max_class_prop`）；右图 overall spectral profile mean + mean±std（std 跨 (C, F)）。
    - **per-class 图（N 张/run）**：**只画 spectral profile**，不画 bar。来自 `learned_weight_profile_stats_multiclass(...)["per_class"][cls]` 的 `mean_raw/std_raw`，min-max 归一化到 [0,1] 后画 mean 黑线 + mean±std 灰带（std 仅跨 F 维）。文件名 `<prefix>_multiclass_feature_weight_mean_std_<Class>_fft.{png,json,csv}`。

- `src/visualization/spectral_profile.py`
  - 新增 `_resolve_weight_matrix_multiclass(model, n_coeffs, n_features, n_classes) -> (C, F, K)`：读 `model.coef_`，`abs()`, reshape。
  - 新增 `learned_weight_profile_multiclass(model, n_coeffs, n_features, n_classes) -> np.ndarray[K]`（overall，归一化）。
  - 新增 `learned_weight_profile_stats_multiclass(model, n_coeffs, n_features, n_classes, class_names) -> dict`，返回 `{overall: {...}, per_class: {cls: {...}}}`。
  - 老 `learned_weight_profile` / `learned_weight_profile_stats` 保持不变，给二分类路径继续用。

- `src/training/probes.py`：**不动**（本轮只走 GPU 路径，CPU 多分类暂不实现）。

### 2.2 要新增

- `scripts/run_passt_fma_small_multiclass.sh`（或直接写成 bash 一行，放报告里留档也行）：
  ```bash
  python scripts/run_generic_fft_band_probe.py \
      --npz_path data_artifacts/passt_embeddings_t64.npz \
      --label_key genres \
      --label_mode multiclass \
      --transform_type fft \
      --band_width 4 \
      --backend gpu --gpu_device cuda \
      --gpu_batch_bands --gpu_batch_cumulative \
      --auto_mode both \
      --results_dir results/passt_fma_small_multiclass_fft \
      --prefix passt_fma_small
  ```
  （multiclass 模式下 `--target_label` 不是必需的；在代码中把它改为 `required=False`，只有 `--label_mode binary` 时强制要求。）

### 2.3 要移动（清理，等 small 跑通再执行）

- `git mv results/passt_fft_genre_specific_analysis_gpu_small_t64 results/passt_fft_genre_specific_analysis_gpu_small_t64_ovr_legacy`
- `git mv results/passt_fma_medium_gpu_by_shard results/passt_fma_medium_gpu_by_shard_ovr_legacy`
- 其他 PaSST / CLAP / composer 目录均不动。

### 2.4 要更新（文档）

- `reports/exp_report.md`："2nd exp / Small-FMA Spectral Probing Result Analysis (8 genres)" 一节目前是按 one-vs-rest 写的；等新 small 结果出来后，整节换成多分类叙事（overall accuracy + per-class recall），并加一段 "旧 one-vs-rest 结果已迁到 `*_ovr_legacy/` 目录做对照"。
- `projPlan/2nd_analysis_plan.md` §1.5 "Learned weight (和论文一致)"：追加一行 "→ 在 PaSST 路径上，第 4 次 sanity check 后改为多分类 per-band probe，详见 `4th_sanity_check_plan.md`"。

## 3. 执行顺序

### 阶段 A：代码改动 + fma_small 小规模打通

1. `src/visualization/spectral_profile.py` 加多分类版 profile 函数（纯新增）。
2. `scripts/run_generic_fft_band_probe.py` 加 `--label_mode multiclass` + 3 个 GPU 多分类路径（单任务 / batched same-dim / batched prefix）；`--backend cpu --label_mode multiclass` 组合直接报错。
3. **smoke test**（只 full_embedding）：在 `data_artifacts/passt_embeddings_t64.npz` 上跑一次 `--label_mode multiclass --gpu_batch_bands=False --gpu_batch_cumulative=False` 但只等 full_embedding 训完（可通过临时 `--band_width 10000` 让 `num_bands=0` 然后在代码里早退；或者直接加 `--only_full` 调试开关但不入主 API），确认：
   - overall accuracy > `1/8 = 0.125`（不然有 bug）；
   - `per_class_recall` 每类都有数，且均值 ≈ overall accuracy。
4. 打开 `--gpu_batch_bands --gpu_batch_cumulative`，跑 fma_small 全量多分类；输出到 `results/passt_fma_small_multiclass_fft/`。
5. 肉眼 check：`passt_fma_small_multiclass_spectral_summary_genres_fft.png` 里 ORIG / 各 band / AUTO_CMAX 的 bar、chance 线、右图 spectral profile 都正确。

### 阶段 B：清理旧产物 + 更新文档

6. `git mv` 两个旧目录到 `*_ovr_legacy/`（即便 medium 这边还没重跑完，也先改名，腾出新目录的 namespace；`_ovr_legacy` 里的东西仍然可用作对照）。
7. 更新 `reports/exp_report.md` Small-FMA 那节为多分类叙事；在 `projPlan/2nd_analysis_plan.md` §1.5 加交叉引用。

### 阶段 C（延后到下一次会话再做）：fma_medium by_shard

8. 新增 `scripts/run_passt_fma_medium_multiclass_by_shard.sh`：`for i in 0..7` 循环 8 个 shard，每个 shard 跑一次 `run_generic_fft_band_probe.py --label_mode multiclass --npz_path data_artifacts/passt_fma_medium_shard{i}.npz --prefix passt_fma_medium_shard{i} --results_dir results/passt_fma_medium_multiclass_fft_by_shard`。
9. 每个 shard 独立 label space（以该 shard 实际出现的 class 为准），在 JSON 里记录 `classes` / `class_counts` / `label_mode="multiclass"`；后续跨 shard 聚合时再对齐。
10. 跑完更新 `reports/exp_report.md` 补一小节 medium-by-shard 对比。

## 4. 风险与权衡

- **GPU 显存**：fma_small PaSST 的 `coeffs.shape = (B≈8000, F=1295, K=301)`（假设 T=601 → rfft(601)=301），`band_width=4` → `num_bands=75`。batched band 的 W `(K=75, D=1295*4=5180, C=8)` ≈ 1.6 亿参数？不对，75*5180*8 ≈ 3.1 M，×4B = 12 MB，OK。batched prefix 的 W `(K=75, D_full=1295*301=389895, C=8)` = 2.3 亿参数 ≈ 940 MB —— **过大，不可接受**。
  - **对策**：batched prefix 不把 W 建在 `D_full` 上，而是每个 prefix 仍然建自己的 W，只是所有 K 次 forward/backward 共用同一个 batched kernel。或者退化成顺序跑 prefix（现有代码的 fallback）。实现时先尝试 batched，OOM 再顺序。需要在代码里加 `try/except RuntimeError` 并 fallback，和现有二分类版本一致的策略。
- **样本量与类别均衡**：fma_small 每类 ≈ 1000 样本，均衡。medium 每 shard ~2100 样本 / ~7 类，少数类（Old-Time/Historic）可能只有几十条，需要 `class_weights`（已纳入设计）。
- **旧产物兼容**：`*_ovr_legacy/` 里的 `band_specific` 是"二分类正类为 `target` vs 其余"的 accuracy，**数值不能直接和新 multiclass accuracy 比**。报告里必须写明这一点，并以 "新值 vs `1/n_classes` chance" 作为主要解读视角。
- **`run_passt_spectral_profiles.py` streaming 路径**：本轮**完全不动**。streaming + multiclass 需要 `SGDClassifier(loss="log_loss")` 第一次 partial_fit 时声明全部 classes，将来如有需要再在第 5 次计划里处理。

## 5. 验证 checklist（smoke 和全量跑完都要过）

- 新 JSON schema 字段齐全：`label_mode == "multiclass"`、`classes`、`class_counts`、`full_per_class_recall: {cls: recall}`、`band_per_class_recall: [per_band_dict]`、`cumulative_per_class_recall: [per_prefix_dict]`、`accuracies.full_embedding`、`accuracies.band_specific[b]`、`accuracies.cumulative_per_band[k]`、`accuracies.cumulative_auto`、`accuracies.trained_auto`、`auto_details.trained_auto_selected_bands`、`spectral_profile`（overall 归一）、`spectral_profile_stats.overall.*`、`spectral_profile_stats.per_class.<Class>.*`。
- overall accuracy 明显高于 `1/n_classes`；若接近或低于，排查数据加载 / label 编码。
- `sum_c (n_c / N) * per_class_recall[c] ≈ overall accuracy`（允许小浮点误差；本质上 micro-averaged recall = accuracy）。
- 图里：
  - bar 的 AUTO_CMAX 等于 `max(cumulative_per_band)`。
  - chance 虚线有两条（`1/n_classes` + `max_class_prop`），在图例里标出。
  - 右图 spectral profile 是 overall，shaded band 合理（std 跨 (C, F)，不为 0）。
- 所有 N 个 class 都有 `_multiclass_feature_weight_mean_std_<Class>_fft.png`，文件数 = `n_classes`，**图里只有 profile curve（mean + mean±std 灰带），没有 accuracy bar**。

## 6. 和前几轮计划的关系

- 继承 `projPlan/2nd_analysis_plan.md` 的数据形状约定（NPZ 中 `embeddings` 直接是 `(B, F, T)`）、`apply_transform(..., transform_type="fft", axis=2)` 一条 FFT 入口、`learned_weight_profile` 的语义。
- `projPlan/3rd_sanity_check_plan.md` 里 zero-padding / 自相关 / 多模型对比部分都走自己的路径，不在本轮射程内；本轮对齐的只是"共享同一个 PaSST NPZ"这一点。
- 和 `projPlan/2nd_exp_plan.md` 里 Milestone B（feature std on weight plots）的 mean±std 约定保持一致：新 plot 依然是 mean（黑线）+ mean±std（灰带），只是值来自多分类的 per-class 或 overall profile。
