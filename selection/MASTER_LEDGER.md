# Idea 选择主 ledger(5 评委 ensemble + 对抗复核)

评委:router(deepseek-v3.2 / gpt-4o / kimi-k2.5) · 外部面板 A(reasoning 模型) · 外部面板 B(含对抗复核)

池子 76 → 评分去重 → 应用对抗裁剪 9 项 → **共识 shortlist 35**

## 共识 shortlist(按综合分,满20)

| 排名 | idx | title | 综合分 | 各源分 | 重负载注 |
|---|---|---|---|---|---|
| 1 | 28 | The Regularization Interference Effect: When C | **17.75** | deepseek-v3.2=18.0; gpt-4o=19.0; kimi-k2.5=17.0; panel_b=17.0 |  |
| 2 | 73 | The Brittleness Signature: Simple Diagnostic T | **17.5** | deepseek-v3.2=19.0; gpt-4o=17.0; kimi-k2.5=18.0; panel_b=16.0 |  |
| 3 | 20 | The Calibration Paradox: Why Data Augmentation | **17.5** | deepseek-v3.2=19.0; gpt-4o=17.0; kimi-k2.5=16.0; panel_b=18.0 |  |
| 4 | 37 | Noise as a Litmus Test: Using Controlled Label | **17.0** | deepseek-v3.2=19.0; gpt-4o=19.0; kimi-k2.5=15.0; panel_b=15.0 |  |
| 5 | 8 | Reinforce or Diversify? Selective Sample Dupli | **17.0** | deepseek-v3.2=18.0; gpt-4o=17.0; kimi-k2.5=17.0; panel_b=16.0 |  |
| 6 | 1 | The Informative Noise: Can High-Loss Samples U | **17.0** | deepseek-v3.2=17.0; gpt-4o=17.0; kimi-k2.5=17.0 |  |
| 7 | 33 | The Memorization Test: Can Simple Synthetic Ta | **16.75** | deepseek-v3.2=18.0; gpt-4o=19.0; kimi-k2.5=16.0; panel_a=14.0 |  |
| 8 | 3 | The Augmentation Paradox: When Simple Data Aug | **16.75** | deepseek-v3.2=19.0; gpt-4o=17.0; kimi-k2.5=16.0; panel_a=15.0 |  |
| 9 | 69 | Confidence-Maximizing Augmentation: A Simple I | **16.75** | deepseek-v3.2=17.0; gpt-4o=17.0; kimi-k2.5=17.0; panel_b=16.0 |  |
| 10 | 5 | The Training Oracle: Can Simple Training Stati | **16.4** | deepseek-v3.2=16.0; gpt-4o=16.0; kimi-k2.5=17.0; panel_a=17.0; panel_b=16.0 | 重负载:含 DistilBERT/SST-2 文本 leg(T2 creep)→ 若入选须删文本 leg 只留 vision |
| 11 | 22 | The Augmentation Pathway: Does Consistent Orde | **16.33** | deepseek-v3.2=17.0; gpt-4o=16.0; kimi-k2.5=16.0 |  |
| 12 | 35 | The Right Momentum: Adaptive Batch Normalizati | **16.33** | deepseek-v3.2=15.0; gpt-4o=17.0; kimi-k2.5=17.0 |  |
| 13 | 14 | Stop When It Compresses: Representation Compre | **16.25** | deepseek-v3.2=17.0; gpt-4o=16.0; kimi-k2.5=17.0; panel_a=15.0 |  |
| 14 | 2 | The Warmup Dilemma: Does Early Weight Decay Du | **16.25** | deepseek-v3.2=16.0; gpt-4o=16.0; kimi-k2.5=16.0; panel_a=17.0 |  |
| 15 | 40 | The First Step Matters Most: How Initial Weigh | **16.0** | deepseek-v3.2=15.0; gpt-4o=18.0; kimi-k2.5=17.0; panel_a=14.0 |  |
| 16 | 60 | The Test-Time Simplicity Gap: How Small-Datase | **16.0** | deepseek-v3.2=15.0; gpt-4o=17.0; kimi-k2.5=18.0; panel_a=14.0; panel_b=16.0 |  |
| 17 | 65 | The Learning Order Imbalance: How Uneven Class | **16.0** | deepseek-v3.2=17.0; gpt-4o=17.0; kimi-k2.5=16.0; panel_a=14.0 |  |
| 18 | 10 | The Learning Rate Plateau: Does Stalling Learn | **16.0** | deepseek-v3.2=16.0; gpt-4o=16.0; kimi-k2.5=16.0 |  |
| 19 | 52 | The Receptive Field Timeline: Does Placing Lar | **16.0** | deepseek-v3.2=16.0; gpt-4o=16.0; kimi-k2.5=16.0 |  |
| 20 | 55 | Schedule the Noise: Why Data Augmentation Stre | **16.0** | deepseek-v3.2=15.0; gpt-4o=18.0; kimi-k2.5=15.0 |  |
| 21 | 38 | Dense or Sparse? How Skip Connection Density S | **16.0** | deepseek-v3.2=14.0; gpt-4o=18.0; kimi-k2.5=14.0; panel_a=18.0 |  |
| 22 | 25 | The Forgotten Knob: Tuning BatchNorm Gamma Ini | **15.75** | deepseek-v3.2=15.0; gpt-4o=18.0; kimi-k2.5=17.0; panel_a=13.0 |  |
| 23 | 67 | Clipping the Outliers: Gradient Clipping as Un | **15.75** | deepseek-v3.2=16.0; gpt-4o=17.0; kimi-k2.5=14.0; panel_b=16.0 |  |
| 24 | 51 | The Order Within: How Staging Regularization W | **15.67** | deepseek-v3.2=14.0; gpt-4o=17.0; kimi-k2.5=16.0 |  |
| 25 | 75 | Cleaning the Noise: Selective Suppression of H | **15.6** | deepseek-v3.2=15.0; gpt-4o=17.0; kimi-k2.5=16.0; panel_a=15.0; panel_b=15.0 |  |
| 26 | 17 | The Calibration Shake: Can Small Intentional O | **15.5** | deepseek-v3.2=16.0; gpt-4o=16.0; kimi-k2.5=17.0; panel_a=13.0 |  |
| 27 | 27 | Consistency Matters: Why Fixed-Order Data Augm | **15.5** | deepseek-v3.2=15.0; gpt-4o=17.0; kimi-k2.5=16.0; panel_a=14.0 |  |
| 28 | 43 | Progressive Augmentation Scheduling: Growing D | **15.5** | deepseek-v3.2=15.0; gpt-4o=17.0; kimi-k2.5=15.0; panel_a=15.0 |  |
| 29 | 16 | The Challenge Batch: Does Grouping Similar Sam | **15.33** | deepseek-v3.2=15.0; gpt-4o=16.0; kimi-k2.5=15.0 |  |
| 30 | 56 | The Order of Operations: Does Temporal Sequenc | **15.33** | deepseek-v3.2=14.0; gpt-4o=17.0; kimi-k2.5=15.0 |  |
| 31 | 62 | Breaking the Symmetry: Decoupling Sample Selec | **15.33** | deepseek-v3.2=14.0; gpt-4o=17.0; kimi-k2.5=15.0 |  |
| 32 | 36 | Time-Dependent Optimizer Scheduling: Dynamical | **15.25** | deepseek-v3.2=13.0; gpt-4o=16.0; kimi-k2.5=16.0; panel_a=16.0 |  |
| 33 | 42 | The Small-Batch Advantage: How Batch Size Serv | **15.25** | deepseek-v3.2=15.0; gpt-4o=17.0; kimi-k2.5=13.0; panel_b=16.0 |  |
| 34 | 23 | One Hyperparameter to Rule Them All? A Scaling | **15.2** | deepseek-v3.2=17.0; gpt-4o=16.0; kimi-k2.5=14.0; panel_a=13.0; panel_b=16.0 | 重负载:含 Tiny-ImageNet + 上千次训练 → 若入选须砍 Tiny-ImageNet、缩网格 |
| 35 | 46 | The Early Bird Gets the Feature: How Architect | **15.0** | deepseek-v3.2=12.0; gpt-4o=18.0; kimi-k2.5=15.0 |  |

## 被对抗复核裁剪的(带理由)

- **[71] The Thinking Gap: Inserting a Computational Pause ** — 退化:forward/backward 插暂停机制 hand-wavy,效应在 full CIFAR 淹于噪声(最弱)(各源分:{'router:deepseek-v3.2': 14.0, 'router:gpt-4o': 14.0, 'router:kimi-k2.5': 17.0, 'panel_b': 16.0})
- **[29] Before or After? How the Order of Operations Withi** — 退化:WD 步序 before/after 效应 <0.5% < seed 方差(~±0.3-0.5%)→ 大概率 null(各源分:{'router:deepseek-v3.2': 14.0, 'router:gpt-4o': 18.0, 'router:kimi-k2.5': 14.0, 'panel_a': 17.0, 'panel_b': 16.0})
- **[64] The Activation Filter: Do Smoother Activations Act** — 退化:换激活(ReLU vs Swish/Mish)CIFAR 上差异 <1% 且文献已知,饱和任务测不出(各源分:{'router:deepseek-v3.2': 16.0, 'router:gpt-4o': 15.0, 'router:kimi-k2.5': 14.0, 'panel_b': 16.0})
- **[26] One Good Twist: Learning a Single Test-Time Transf** — 退化:单个固定 TTA 在 clean CIFAR 提升 <0.3%,full data 测不出(各源分:{'router:deepseek-v3.2': 11.0, 'router:gpt-4o': 17.0, 'router:kimi-k2.5': 13.0, 'panel_a': 16.0, 'panel_b': 15.0})
- **[68] Clipping as a Knob: Gradient Norm Thresholding as ** — 硬重复:与 67 abstract 逐字相同(grad-clip);保留 67(各源分:{'router:deepseek-v3.2': 16.0, 'router:gpt-4o': 12.0, 'router:kimi-k2.5': 14.0, 'panel_a': 16.0, 'panel_b': 16.0})
- **[4] When More Data Hurts: How Standard Data Augmentati** — 簇收敛:与 20(calibration)/8(duplication)同机制(aug 放大标签噪声),保留 20+8(各源分:{'router:deepseek-v3.2': 19.0, 'router:gpt-4o': 16.0, 'router:kimi-k2.5': 16.0, 'panel_b': 18.0})
- **[13] The Compression Test: Does Representation Compress** — 簇收敛:与 15 同 compressibility 机制;15 更具体可执行,保留 15(各源分:{'router:deepseek-v3.2': 17.0, 'router:gpt-4o': 16.0, 'router:kimi-k2.5': 17.0, 'panel_b': 15.0})
- **[48] Confidence-Guided Augmentation: Adapting Data Dist** — 簇收敛:与 57(aug 强度调度)同题,保留 57(各源分:{'router:deepseek-v3.2': 13.0, 'router:gpt-4o': 16.0, 'router:kimi-k2.5': 14.0, 'panel_b': 15.0})
- **[63] The Augmentation Pathway: Does Consistent Ordering** — 残留重复:与 22 标题完全相同(The Augmentation Pathway,固定增强顺序),保留 22(各源分:{'router:deepseek-v3.2': 16.0, 'router:gpt-4o': 17.0, 'router:kimi-k2.5': 17.0, 'panel_a': 15.0})

## 需简化后再用的重负载项

- **[5] The Training Oracle: Can Simple Training Statistic** 在shortlist — 重负载:含 DistilBERT/SST-2 文本 leg(T2 creep)→ 若入选须删文本 leg 只留 vision
- **[23] One Hyperparameter to Rule Them All? A Scaling Law** 在shortlist — 重负载:含 Tiny-ImageNet + 上千次训练 → 若入选须砍 Tiny-ImageNet、缩网格
