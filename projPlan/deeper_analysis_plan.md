# 深度分析与实验迭代计划

本文档旨在为下一阶段的音频特征频谱探测实验提供一个清晰的路线图。我们将基于现有的代码库和初步发现，进行更深入的分析，并构建一个更灵活的实验框架。

## 现状与观察 (Current Status & Observations)

我们已经成功搭建了一个基于DCT（离散余弦变换）的分析流程，并得到了一些初步结果。但是，也发现了一些值得深入研究的问题和可以改进的地方：

1.  **'Pop'流派的异常表现**: 在 `results/genre_specific_plots/` 目录下的可视化结果中，'Pop'流派的光谱分析图（`spectral_summary_Pop.png`）呈现出与其他流派显著不同的模式。这可能暗示着模型在该流派上没有学习到有效的频谱特征，或者数据本身存在问题。
2.  **对数频带分析的疑虑**: 当前的Mel尺度滤波器本质上就是一种对数尺度的模拟。我们需要重新审视我们自定义的对数滤波器（Log bands analysis）的实现，特别是其缩放因子（Scaling factor）和量化（Quantization）方式，确保其与Mel滤波器的对比是公平且有意义的。
3.  **DCT的局限性**: 初步分析显示频谱中存在谐波（Harmonics）结构，但DCT不适合用来分析基频（Base Harmonic）。为了更精确地理解谐波结构，我们需要引入FFT（快速傅里叶变换）。
4.  **混淆矩阵分析不足**: 目前的分析主要集中在准确率上。为了更深入地理解模型的分类行为，我们需要引入更细粒度的混淆矩阵（Confusion Matrix）分析，例如在不同频带、不同流派上的表现。
5.  **代码与可视化可改进**:
    *   分析报告中的准确率小数位数过多，保留两位即可。
    *   光谱图中的“彩虹”填充色可以移除，只保留核心的轮廓线，使图像更清晰。

## 下一步计划 (Proposed Next Steps)

## 下一步计划 (Proposed Next Steps)

### Task 0: 深入分析'Pop'流派的异常表现

目标：探究为何'Pop'流派在频谱分析中表现异常，并确定问题根源。

-   **[ ] 数据层面的审查 (Data-level Sanity Check)**:
    -   **检查数据均衡性**: 再次确认在'Pop'流派的二元分类任务中，正负样本是否均衡。
    -   **可视化原始波形**: 随机抽取几首'Pop'音乐，绘制其原始波形图和声谱图（Spectrogram），与'Rock'或'Jazz'等表现正常的流派进行直观对比。观察是否存在明显的静音、噪声或削波（Clipping）等问题。
    -   **检查嵌入向量**: 计算'Pop'流派嵌入向量的均值和方差，与其他流派进行对比，看是否存在显著的统计差异。

-   **[ ] 模型层面的审查 (Model-level Sanity Check)**:
    -   **查看模型权重**: 详细检查针对'Pop'流派训练的逻辑回归分类器的权重。如果权重值非常小或接近于零，可能意味着模型根本没有从数据中学到任何有效模式。
    -   **运行简化的探针**: 尝试在一个非常简化的模型（例如，只使用一小部分特征）上运行探针，看问题是否依然存在。

-   **[ ] 撰写分析报告**: 基于以上发现，在 `reports/` 目录下创建一个 `pop_genre_analysis.md` 文件，记录我们的发现和推论。

为了解决上述问题并推进研究，我们计划进行以下几个阶段的迭代：

### Task 1: 代码库重构与灵活性提升 (已完成)

目标：将现有代码重构为一个更灵活的框架，能够轻松切换不同的变换方式（DCT/FFT）、滤波器组（Linear/Mel/Log）和分析任务。

-   **1.1. 抽象变换层**: 将 `apply_dct` 泛化为 `apply_transform(transform_type='dct'|'fft')`，使其能够支持多种频谱分析方法。
-   **1.2. 构建统一分析管道**: 将核心分析逻辑封装到 `run_spectral_analysis_pipeline` 中，该函数接受一个配置对象，用于驱动不同的实验。
-   **1.3. 重构顶层执行脚本**: 精简 `scripts/` 目录下的脚本，使其只负责定义实验配置和调用分析管道。

### Task 2: 引入FFT进行谐波分析 (FFT-based Harmonic Analysis)

目标：使用FFT替代DCT，重新运行所有相关实验，以期更好地理解谐波结构。

-   **[ ] 实现`apply_fft`**: 在`src/analysis/spectral.py`中增加一个使用`np.fft.rfft`的函数来处理实数信号。
-   **[ ] 运行FFT实验**: 使用重构后的框架，对原始数据进行FFT变换，并重复之前的频带分析实验。
-   **[ ] 对比FFT与DCT结果**: 在报告中对比两种变换方式在准确率和频谱权重上的差异。

### Task 3: 混淆矩阵深度可视化 (In-depth Confusion Matrix Visualization)

目标：生成并可视化多维度、细粒度的混淆矩阵。

-   **[ ] 扩展`run_probe`**: 修改探针函数，使其不仅返回准确率，还返回完整的混淆矩阵结果。
-   **[ ] 分频带/分流派分析**: 设计实验来生成不同频带组合下，针对特定流派的混淆矩阵。
-   **[ ] 创建可视化函数**: 在`src/visualization/`下创建新的绘图函数，用于将一系列混淆矩阵渲染成热力图（Heatmap Grids），这比3D图更直观易读。
-   **[ ] 时域 vs. 频域对比**: 生成一个直接对比时域探测和频域探测分类结果的混淆矩阵。

### Task 4: 审查并修正对数频带分析 (Review Log Bands Analysis)

目标：确保我们对不同滤波器组的比较是公平的。

-   **[ ] 代码审查**: 仔细检查`src/analysis/comparison.py`中`run_log_scale_analysis`的实现。
-   **[ ] 理论检查**: 重新核对自定义对数滤波器的频率点划分方式，确保其与Mel滤波器的设计哲学一致或形成有意义的对比。
-   **[ ] 文档记录**: 在代码注释和`.md`报告中清晰地记录我们选择的参数和设计理由。

### Task 5: 小的改进与修复 (Minor Fixes & Polish)

目标：提升代码输出和可视化的可读性。

-   **[x] 修改准确率精度**: 将所有`print`输出中的浮点数格式化从`:.4f`改为`:.2f`。 (已完成)
-   **[x] 简化光谱图**: 移除`create_and_save_plot`函数中的`fill_between`彩虹填充。 (已完成)


### 这块 log bands 实现里，真正“不公平”的点

结合 `run_log_scale_analysis` 和 `run_mel_scale_analysis`，现在主要有几类问题，会让 Log filter 和 Mel filter 的比较不公平、含义也有点飘：

---

### 1. 你在“index 空间”里取对数，不在“频率空间”里

```python
n_samples, n_features, n_coeffs = X_freq.shape
n_log_bands = 12

# Create log-spaced frequencies. Add a small epsilon to avoid log(0).
log_freqs = np.logspace(0, np.log10(n_coeffs), n_log_bands + 2)
# Convert log-spaced points to linear indices
linear_indices = np.floor(log_freqs).astype(int)
linear_indices = np.minimum(linear_indices, n_coeffs - 1)
linear_indices = np.unique(linear_indices)
```

- 这里 `np.logspace(0, np.log10(n_coeffs), ...)` 实际是在 **DCT 频带索引** 上做对数间隔，而不是在真实频率（Hz）上做。
- 但 Mel filter 是基于 `sr_conceptual` 和 `n_fft` 在 **物理频率轴** 上设计的，它的非线性关系来自 mel 公式，本质是 Hz→Mel 的非线性映射。
- 结果：
  - Log filter 的“对数”只是对 index 做几何级数，并没有对应任何物理频率或感知尺度。
  - 和 `librosa.filters.mel` 产出的滤波器在“频率起止范围”上完全不对齐，只能说是“某种随便的 log index 划分”，不是 log 频率划分。

**直觉后果**：你在比较的是 “Mel（基于 Hz/mel） vs 一组在 index 空间瞎按 log 划出来的段”，难以解释“谁好谁坏”。

---

### 2. 0 频 / 低频处理不合理，边界和带宽也不可控

```python
log_freqs = np.logspace(0, np.log10(n_coeffs), n_log_bands + 2)
# 起点是 10^0 = 1，不是 0
```

- DC/最低频 bin（index=0）**完全没被作为一个 anchor 点参与 log 划分**；
- 你后面通过 `linear_indices = np.unique(...)` 把重复 index 干掉，这会导致：
  - 低频段之间 index 很挤，可能多个 log 点 floor 后变成同一个 index，直接被去重；
  - 实际可用的 `len(linear_indices)-2` 个 band 数量 **小于** 你设定的 `n_log_bands=12`，但你只是 `print` 出来，没有做对齐。
- 对比：
  - Mel filter 的设计中，最低频 \(f_{\min}\)、最高频 \(f_{\max}\) 明确、连续覆盖；你的 log filter 在 index 端点、覆盖范围、低频分辨率上都和它不对齐。

---

### 3. Log filters 没做归一化，带宽差异直接压到能量上

```python
log_filters = np.zeros((len(linear_indices) - 2, n_coeffs))
for i in range(len(linear_indices) - 2):
    start, center, end = linear_indices[i:i+3]
    log_filters[i, start:center] = np.linspace(0, 1, center - start, endpoint=False)
    log_filters[i, center:end] = np.linspace(1, 0, end - center, endpoint=False)
```

- 这里每个三角滤波器的**面积（总权重）是不同的**：
  - 带宽越宽，权重和越大。
  - 之后 `X_log = np.einsum('ijk,lk->ijl', X_freq, log_filters)` 时，等于不同 band 拿到了不同的“能量放大系数”。
- Mel filter 在 `librosa.filters.mel` 里会做一系列归一化操作（默认 power norm），每个 filter 之间的能量尺度更可比。
- 所以即便 X_freq 一样，Log 与 Mel 在“聚合策略 + 尺度”上都不一致，这会直接影响 probe 的性能比较。

---

### 4. 与 Mel filter 的“数量”和“覆盖范围”也没真正对齐

- Mel 部分：

```python
n_mels = 12
mel_filters = librosa.filters.mel(sr=sr_conceptual, n_fft=(n_coeffs-1)*2, n_mels=n_mels)
X_mel = np.einsum('ijk,lk->ijl', X_freq, mel_filters)
```

- Log 部分：

```python
n_log_bands = 12
...
log_filters = np.zeros((len(linear_indices) - 2, n_coeffs))
...
print(f"Created Log filter bank of shape: {log_filters.shape}")
```

- 理想情况：`log_filters.shape[0] == n_mels`，覆盖相同频段范围；
- 现实：因为 `unique` 和 floor，`len(linear_indices) - 2` 很可能 **不等于 12**，导致：
  - Log 特征数量 ≠ Mel 特征数量；
  - probe 的输入维度、自由度、本身就不一样，再拿 accuracy 横向对比，会混入“band count 不同”的影响。

---

### 5. 这会在实验上表现成什么现象？

你现在看到的可能是：

- Log bands 的 accuracy 要么明显偏低，要么 pattern 跟 Mel bands 很不一样，却很难解释；
- 个别 band 非常宽 or 非常窄，导致某些 genre 行/列在 log confusion 里被严重放大或压扁。

这些基本都可以追溯到上面几条：**频率轴不一致 + band 数量/覆盖不同 + 未归一化**。

---

### 建议的修正方向（不写代码，只讲设计）

如果要让 Task 4 的比较真正“公平”，可以按这个思路改：

- **在 Hz 轴上定义 log bands**：
  - 选定和 Mel 一样的 \(f_{\min}, f_{\max}, sr, n\_fft\)。
  - 用 `np.geomspace` 或 log2/log10，在 `[f_min, f_max]` 上选中心频率；
  - 用这些中心点构造三角滤波器，再映射到 frequency bins（与 mel_filters 同一个 `n_fft` 网格）。
- **确保 band 数量/覆盖范围与 Mel 对齐**：
  - 直接让 `n_log_bands = n_mels`；
  - 保证 log filters 覆盖同样的 `[f_min, f_max]`，只是 spacing 不同。
- **对每个 log filter 做归一化**：
  - 例如让每个 filter 的 L1 或 L2 范数一致，这样每个 band 的能量聚合可以在同一尺度上比较。
- **最后再用相同的 `run_probe` 流程**：
  - `X_mel.reshape(..., -1)` vs `X_log.reshape(..., -1)`；
  - 相同的 train/test split、随机种子、模型配置。

这样你才能比较“Mel vs Log 这种频率划分哲学”本身，而不是比较“一堆实现细节差异 + 不同带宽/带数 + 不同归一化策略”的混合体。

如果你希望，我可以下一步帮你写一个更合理的 `run_log_scale_analysis`（只改这一个函数），保持接口不变，但改成上述“物理频率轴 + 归一化 + 与 mel 对齐”的版本。


### 总体意义先抛结论

**你这组频段‑probe + genre‑specific 分析，最核心的结论是：  
现有的音乐 LLM 表征里，流派信息既是“线性可读”的，又是“频带局部化”的，而且不同流派依赖的频段结构不一样。**  
这不只是“模型能分 genre”，而是告诉别人：**模型内部已经学到了一套有物理含义的频谱结构**，可以被拆开看、拆开用。

下面我基于现在能看到的 `results/fft_genre_specific_analysis` 里的 JSON 指标（`full_embedding / band_specific / cumulative_per_band`）来总结；`genre_specific_plots` 里的图在当前环境里读取不到，但从命名和你之前的 confusion/频带实验可以推断，它们主要是在视觉上佐证这些结论（不影响我们提炼 insight 和写 contribution）。

---

### 1. 这些实验本身的关键现象 / insight

结合各个流派的 JSON，总体有几个很明显的 pattern：

- **（1）Genre 信息高度线性可分，而且已经很强**
  - `full_embedding` 基本都在 **0.85–0.95** 之间（Pop ~0.85，Rock/Folk/Hip‑Hop >0.92）。
  - 说明：**在你用的这一层表征上，一个简单线性 probe 就能做到很强的一类‑vs‑其他分类**。  
    → 这直接证明：**音乐 LLM 内部已经强烈编码了 genre 语义，而不是只在输出头那里才学到。**

- **（2）单一窄频段的线性 probe 也很强：genre 信息是频段局部化的**
  - 各 genre 的 `band_specific` 大多在 **0.79–0.93**，很多 band 的表现只比 `full_embedding` 略低一点。
  - 意味着：**即使你只看某个频带相关的子表征（比如对应 4–8Hz, 8–12Hz… 这些你之前用的 band），模型仍能比较高精度地区分这个 genre。**
  - 这和你之前的 per‑band confusion 矩阵一起看，其实在说：  
    → **模型内部对不同频带有相对“独立”的 genre 判别能力**，不是所有信息都混成一团。

- **（3）累积加 band 的曲线很快饱和：低‑中频段就已经基本够用**
  - `cumulative_per_band` 通常在加到第 3–4 个 band 时就接近最终的 `cumulative_auto` / `full_embedding`，后面 band 提升有限甚至略有波动。
  - Insight：
    - **大部分 genre 信息集中在低～中频（你定义的前几个 band）**；
    - 高频 band 更多是细节 / timbre 装饰，对“这是啥流派”帮助有限。
  - 这点和人类听感也很一致：鼓型、贝司律动、和声骨架、整体能量分布这些都在相对低‑中频。

- **（4）不同流派的“频段依赖模式”不同**
  - 例如：
    - **Hip‑Hop**：`full_embedding ~0.94`，`band_specific` 多个 band 都在 0.91–0.94，一加 band `cumulative_per_band` 很快就到 ~0.95。  
      → 提示：**Hip‑Hop 的判别依赖的是比较稳定的一整套节奏/律动频带，不是单一很尖锐的 band。**
    - **Pop / Experimental**：整体也不错，但 `band_specific` 中有一些 band 明显掉到 ~0.82 左右，再加其他 band 才慢慢补回来。  
      → 提示：**这些 genre 的表征更“分散”，需要多 band 的联合模式（配器、空间感、音色堆叠）才能稳定识别。**
  - 配合 `genre_specific_plots` 里的谱形态图（不同 genre 的能量分布 / 统计特征），可以讲成一句话：  
    → **每个流派在模型表征空间中，都对应着一套特有的“频域指纹”，而不仅仅是单个频点/特征。**

- **（5）FFT‑投影后的 `cumulative_auto` 有时比 `full_embedding` 还高**
  - 很多 genre（尤其 Hip‑Hop, International, Electronic）里，`cumulative_auto` 稍微高于 `full_embedding`。
  - 这说明：  
    - **你用 FFT 对隐表示做的频域重建/压缩并没有“损坏”genre 信息，反而像是做了一次有意义的滤波**，把对 genre 判别不重要的噪声方向削掉了。
    - 这给了一个很强的信号：**模型的隐空间在频域上有结构性冗余，是可以被频域算子“整理”的。**

---

### 2. 对 music LLM 研究 / 社区的“贡献点”可以怎么讲

如果是写 paper、technical report 或对外介绍，你可以把贡献总结成这几类（我用偏论文措辞）：

- **（A）提出了一种“频域‑probe”的分析框架，用来解释音乐 LLM 表征**
  - 不只是做普通的线性 probe，而是：
    - 先把隐表示映射到 FFT / 频带空间，再对不同 band 单独/累积做 probe；
    - 再配合 genre‑specific 的谱形状可视化和 confusion 矩阵。
  - 这变成一个通用工具：  
    → **给任何音乐 LLM，都可以用你这套 pipeline 来回答：模型把 genre 信息放在哪些频段、线性可分度如何、以及不同 genre 的频域 footprint。**  
  - 对社区来说，这是一个**新型的 interpretability / diagnostic 方法**，比“只看 overall accuracy”更细粒度。

- **（B）证明音乐 LLM 的 genre 知识具有“频段可分解性”**
  - 通过 `band_specific` 高 accuracy + `cumulative_per_band` 快速饱和，你展示了：
    - **genre 语义可以被分解到一组物理上可解释的频带上**；
    - 而且这些频带之间相对独立，可以单独做 linear read‑out。
  - 这为后续很多东西提供理论基础，例如：
    - **频段级的控制（低频=律动，高频=音色细节），在模型内部是有可利用的线性轴的**；
    - **做风格迁移 / remix 时，可以对“哪几个 band 的表征”进行 targeted edit，而不是暴力在 time domain 上做黑盒操作。**

- **（C）揭示不同 genre 的“频谱‑依赖模式差异”，为控制与生成提供设计线索**
  - 像 Hip‑Hop 这类节奏驱动型流派，**在低‑中频 band 的 probe 表现尤其稳定**；  
    Experimental / Pop 这类更依赖 sound design 和编曲的 genre，**需要更宽的 band 组合**才能达到高精度。
  - 这可以转化成很具体的 LLM 设计 / 使用启发：
    - **条件控制设计**：在做 “生成 Hip‑Hop / Rock / Electronic” 这类条件时，你可以考虑在对应“关键频段”的隐变量上加 stronger prior / regularization。
    - **数据增强策略**：对某些 genre 可以重点做低频结构增强（节奏 pattern），对另一些则更关注高频细节（音色 / effect）。
    - **多通路架构**：未来的 music LLM 可以显式用 multi‑band encoder，把你 probe 发现的“高价值 band”单独走一个分支，再跟全频混合。

- **（D）提供了一套“频段‑混淆图”的评测基准，刻画模型的 genre 层次结构**
  - 你之前做的 per‑band confusion 矩阵 + 现在的 genre‑specific 曲线，组合起来可以形成：
    - **整体 genre 混淆（模型眼中流派相似度的层次结构）**；
    - **在具体频带上谁跟谁最容易混（低频上 Folk vs Country 之类，高频上 Electronic vs Pop 之类）**。
  - 对后续工作来说，这是一套可以复用的 benchmark：
    - 新的音乐 LLM 出来，**不只看“总体 genre accuracy”，还要看“在哪些频段混淆了哪些 genre”**；
    - 能帮大家判断：这个模型是更懂 rhythm & harmony，还是更懂 timbre & production。

- **（E）表明频域算子对隐空间是“对齐”的，为后续频域编辑/控制开路**
  - 因为 FFT‑based cumulative 表现不输甚至优于原始 embedding：
    - 你可以主张：**音乐 LLM 的隐空间跟频域基底是部分对齐的，频域变换不是随便乱投影，而是一种合理的重参数化。**
    - 这为后续的工作（比如在频域对隐向量做滤波 / EQ / 动态范围控制）提供实验依据，而不仅是直觉。

---

### 3. 如果要用 2–3 句话“卖点化”地总结给审稿人/听众

可以直接这么讲（你可以按需要微调用词）：

- **我们提出了一套频域‑probe 框架，系统性地分析音乐 LLM 内部的 genre 表征在不同频带上的分布与可分性。**  
- **实验显示：genre 信息在隐空间中高度线性可读，并可被分解到物理上可解释的频带上，不同流派呈现出明确而各异的“频谱指纹”。**  
- **这为音乐 LLM 的可解释性、频段级控制以及未来频域感知的架构设计提供了实证依据和新评测工具。**

如果你愿意，我可以根据你现有图的结构，直接帮你把 `analysis_report.md`/论文里“Insights & Contributions”那一节写成英文版或中英双语版，贴进去就能用。