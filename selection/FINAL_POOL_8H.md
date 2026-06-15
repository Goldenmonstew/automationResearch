# Phase 2 最终 idea 池(41 个,8h timeout)

= 1h-aware 共识 35 + 重负载 add-back 6(8h 后捞回)。timeout 8h,算力管够,效果优先。

## 重负载 add-back(需 8h / T2,当初因 feasibility 被压下)
- [59] The Early Architecture Test: Predicting Optimal Architecture from Initial Training Dynamics on Small Datasets
- [53] The Shape of Data: How Optimal Neural Architecture Depth-Width Ratio Scales with Dataset Size
- [7] Mimicking the Beneficial Gradient Alignment of Label Noise Without the Corruption
- [32] The Orthogonality Oracle: Inter-Layer Gradient Alignment as an Early Predictor of Generalization
- [50] Hyperparameter Signatures: Mapping Architecture Families to Optimal Configurations for Small-Dataset Training
- [0] The Silent Drift: How Learning Rate Schedules Affect Batch Normalization Statistics in Fine-Tuning

## add-back 时的去重/裁剪
- 58 → 重复于 53(depth-width ratio),弃
- 34 → steering toward sharp minima 前提可疑+退化风险,弃

## 全 41 个(base 35 见 MASTER_LEDGER,adds 6 见上)
池子索引: [28, 73, 20, 37, 8, 1, 33, 3, 69, 5, 22, 35, 14, 2, 40, 60, 65, 10, 52, 55, 38, 25, 67, 51, 75, 17, 27, 43, 16, 56, 62, 36, 42, 23, 46, 59, 53, 7, 32, 50, 0]
