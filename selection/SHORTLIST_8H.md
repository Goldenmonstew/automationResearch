# 8h-aware 共识 shortlist(timeout 提到 8h,剔 feasibility 惩罚,捞回重 idea)

排序分 = manifestability + novelty + icbinb_fit(满15,不含 feasibility)。重负载项标 [需8h/T2]。

全池 76 → 去重 33 unique → 裁 3 → **shortlist 30**

| 排名 | idx | title | 排序分 | manif | nov | icbinb | feas | 评委 | 标记 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 28 | The Regularization Interference Effect: Wh | **13.25** | 4.5 | 3.75 | 5.0 | 4.5 | 4 |  |
| 2 | 1 | The Informative Noise: Can High-Loss Sampl | **13.0** | 4.0 | 4.0 | 5.0 | 4.0 | 3 |  |
| 3 | 73 | The Brittleness Signature: Simple Diagnost | **13.0** | 4.0 | 4.0 | 5.0 | 4.5 | 4 |  |
| 4 | 20 | The Calibration Paradox: Why Data Augmenta | **12.75** | 4.25 | 3.75 | 4.75 | 4.75 | 4 |  |
| 5 | 37 | Noise as a Litmus Test: Using Controlled L | **12.75** | 4.5 | 3.75 | 4.5 | 4.25 | 4 |  |
| 6 | 8 | Reinforce or Diversify? Selective Sample D | **12.5** | 4.0 | 4.0 | 4.5 | 4.5 | 4 |  |
| 7 | 33 | The Memorization Test: Can Simple Syntheti | **12.5** | 4.5 | 3.75 | 4.25 | 4.25 | 4 |  |
| 8 | 5 | The Training Oracle: Can Simple Training S | **12.2** | 3.8 | 3.8 | 4.6 | 4.2 | 5 |  |
| 9 | 60 | The Test-Time Simplicity Gap: How Small-Da | **12.2** | 3.4 | 4.0 | 4.8 | 3.8 | 5 |  |
| 10 | 69 | Confidence-Maximizing Augmentation: A Simp | **12.0** | 4.0 | 3.5 | 4.5 | 4.75 | 4 |  |
| 11 | 72 | The Data-Aware Initialization: A Single Sc | **12.0** | 4.33 | 3.67 | 4.0 | 3.0 | 3 |  |
| 12 | 2 | The Warmup Dilemma: Does Early Weight Deca | **11.75** | 4.0 | 3.5 | 4.25 | 4.5 | 4 |  |
| 13 | 19 | The Calibration Shake: Can Small Intention | **11.67** | 4.0 | 3.67 | 4.0 | 4.33 | 3 |  |
| 14 | 55 | Schedule the Noise: Why Data Augmentation  | **11.66** | 4.33 | 3.33 | 4.0 | 4.33 | 3 |  |
| 15 | 62 | Breaking the Symmetry: Decoupling Sample S | **11.66** | 3.33 | 4.0 | 4.33 | 3.67 | 3 |  |
| 16 | 38 | Dense or Sparse? How Skip Connection Densi | **11.5** | 4.25 | 3.5 | 3.75 | 4.5 | 4 |  |
| 17 | 49 | Learning to Regularize: Predicting Optimal | **11.5** | 4.0 | 3.75 | 3.75 | 3.25 | 4 |  |
| 18 | 74 | Diversity-Driven Stochastic Forward Passes | **11.5** | 3.5 | 4.0 | 4.0 | 3.25 | 4 |  |
| 19 | 46 | The Early Bird Gets the Feature: How Archi | **11.34** | 4.0 | 3.67 | 3.67 | 3.67 | 3 |  |
| 20 | 59 | The Early Architecture Test: Predicting Op | **11.33** | 3.33 | 4.0 | 4.0 | 2.33 | 3 | [需8h/T2] |
| 21 | 52 | The Receptive Field Timeline: Does Placing | **11.33** | 4.0 | 3.33 | 4.0 | 4.67 | 3 |  |
| 22 | 24 | The Winding Path to Better Generalization: | **11.25** | 3.5 | 3.75 | 4.0 | 3.25 | 4 |  |
| 23 | 25 | The Forgotten Knob: Tuning BatchNorm Gamma | **11.25** | 4.0 | 3.75 | 3.5 | 4.5 | 4 |  |
| 24 | 75 | Cleaning the Noise: Selective Suppression  | **11.2** | 3.6 | 3.6 | 4.0 | 4.4 | 5 |  |
| 25 | 31 | Consistency Matters: Why Fixed-Order Data  | **11.0** | 4.0 | 3.33 | 3.67 | 5.0 | 3 |  |
| 26 | 53 | The Shape of Data: How Optimal Neural Arch | **10.75** | 4.0 | 3.0 | 3.75 | 2.75 | 4 | [需8h/T2] |
| 27 | 61 | Learning in Sparse Neighborhoods: Adaptive | **10.67** | 3.0 | 4.0 | 3.67 | 3.0 | 3 |  |
| 28 | 39 | Architectural Augmentation Synergy: Which  | **10.5** | 4.0 | 3.25 | 3.25 | 3.25 | 4 |  |
| 29 | 34 | Accelerating Alignment: Steering Gradients | **9.34** | 2.67 | 3.67 | 3.0 | 2.33 | 3 | [需8h/T2] |
| 30 | 0 | The Silent Drift: How Learning Rate Schedu | **9.33** | 3.0 | 3.0 | 3.33 | 2.67 | 3 | [需8h/T2] |

## 仍裁(与 timeout 无关)

- [71] The Thinking Gap: Inserting a Computational Pa — 退化:forward/backward 插暂停 hand-wavy,效应淹于噪声
- [29] Before or After? How the Order of Operations W — 退化:WD 步序 before/after 效应 <0.5% < seed 方差 → null
- [64] The Activation Filter: Do Smoother Activations — 退化:换激活 CIFAR 差异 <1% 且已知,饱和测不出
- [26] One Good Twist: Learning a Single Test-Time Tr — 退化:单个固定 TTA clean CIFAR <0.3%
- [68] Clipping as a Knob: Gradient Norm Thresholding — 硬重复:与 67 逐字相同,保留 67
- [63] The Augmentation Pathway: Does Consistent Orde — 重复:与 22 同题(The Augmentation Pathway),保留 22
- [4] When More Data Hurts: How Standard Data Augmen — 簇收敛:与 20/8 同机制(aug 放大标签噪声),保留 20+8
- [13] The Compression Test: Does Representation Comp — 簇收敛:与 15 同 compressibility,保留 15
- [48] Confidence-Guided Augmentation: Adapting Data  — 簇收敛:与 57 同题(aug 强度调度),保留 57
