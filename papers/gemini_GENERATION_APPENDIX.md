# 生成透明度附录:树搜索过程

> 本附录披露本论文由 **The AI Scientist v2** 自动生成的完整过程:走了哪些搜索策略、产生多少树节点、如何逐层选择/裁剪、agent 如何判断决策。
>
> **写作模型**: gemini-2.5-pro

## A. 搜索策略概述

系统不使用人写模板,从一个研究 idea **零起点**出发,做 **4 阶段 best-first 并行树搜索**。每个节点是一份完整可执行的实验代码;agent 用三种原语扩展树:

- **draft(草稿)**:阶段根节点 — 从上一阶段最佳节点 seed,或全新起草一个实现方向
- **debug(调试)**:父节点运行**报错/buggy** → 读 traceback,定位并修复
- **improve(改进)**:父节点运行**成功** → 在其基础上调超参/换设计/加分析,争取更高指标

每个节点用**多 seed(本次 3 seed)**重复评估取稳定指标。每阶段结束按指标选最佳节点 seed 给下一阶段,其余分支被**裁剪**(不再扩展)。四阶段:initial→baseline_tuning→creative_research→ablation。

## B. 各阶段搜索树

### Stage 1: initial_implementation 初始实现

**6 节点**(good 3 / buggy 3)。搜索树:

```
[0] draft/buggy m=—  (根)
[1] draft/good m=Original Distrib=65.9  (根)
├─ [5] improve/good m=—
├─ [2] improve/buggy m=Final Original D=0.658
├─ [4] improve/good m=Final Original D=73.8
└─ [3] improve/buggy m=Original Distrib=73.2
```

| idx | 动作 | 父 | 状态 | metric | agent 计划/分析摘要 |
|---|---|---|---|---|---|
| 0 | draft | — | buggy | — | The script failed with a `TypeError` in the `forward` method of the `SimpleCNN` model. The call to `torch.dropout` is incorrect. The error m… |
| 1 | draft | — | good | Original Distrib=65.9 | To investigate the "distributional oversmoothing" hypothesis, I will conduct a controlled continual learning experiment using the MNIST data… |
| 2 | improve | 1 | buggy | Final Original D=0.658 | The experiment produces an unexpected result where accuracy on the original task drops and then significantly recovers after training on two… |
| 3 | improve | 1 | buggy | Original Distrib=73.2 | The RotatedDataset class is implemented in a convoluted and incorrect manner. In the __getitem__ method, a PIL image is loaded and then imme… |
| 4 | improve | 1 | good | Final Original D=73.8 | Seed node |
| 5 | improve | 1 | good | — | Aggregate results from multiple seeds |

### Stage 2: baseline_tuning 基线调参

**17 节点**(good 8 / buggy 9)。搜索树:

```
[0] draft/good m=Original Distrib=65.9  (根)
├─ [2] improve/buggy m=Original Distrib=63.7
│  ├─ [3] debug/buggy m=—
│  │  ├─ [6] debug/buggy m=—
│  │  └─ [5] debug/buggy m=—
│  │     └─ [8] debug/buggy m=—
│  │        └─ [11] debug/buggy m=—
│  └─ [4] debug/buggy m=—
│     └─ [7] debug/buggy m=—
│        └─ [10] debug/buggy m=—
├─ [9] improve/good m=Final Original D=69.1
│  ├─ [13] improve/good m=Final Original D=73.2
│  ├─ [14] improve/good m=Final Original D=74.9
│  ├─ [16] improve/good m=—
│  └─ [15] improve/good m=Final Original D=73.3
├─ [12] improve/good m=Original Distrib=66.2
└─ [1] improve/good m=Final Original D=66.2
```

| idx | 动作 | 父 | 状态 | metric | agent 计划/分析摘要 |
|---|---|---|---|---|---|
| 0 | draft | — | good | Original Distrib=65.9 | To investigate the "distributional oversmoothing" hypothesis, I will conduct a controlled continual learning experiment using the MNIST data… |
| 1 | improve | 0 | good | Final Original D=66.2 | The experiment ran successfully, and the results provide strong preliminary evidence for the "distributional oversmoothing" hypothesis. In a… |
| 2 | improve | 0 | buggy | Original Distrib=63.7 | The `RotatedDataset` class has a flawed implementation. In the `__getitem__` method, it converts a PIL image to a tensor, then squeezes it, … |
| 3 | debug | 2 | buggy | — | The script failed with a `DatasetNotFoundError` because it could not find the "kmnist" dataset on the Hugging Face Hub. The line `raw_train … |
| 4 | debug | 2 | buggy | — | The script failed with a `datasets.exceptions.DatasetNotFoundError` because the dataset 'kmnist' was not found on the Hugging Face Hub. The … |
| 5 | debug | 3 | buggy | — | The script failed with a `datasets.exceptions.DatasetNotFoundError` because the dataset identifier 'emnist' is not correct on the Hugging Fa… |
| 6 | debug | 3 | buggy | — | The script failed to run because of a `DatasetNotFoundError` for the 'kmnist' dataset. The path provided in the `DATASET_CONFIG` dictionary,… |
| 7 | debug | 4 | buggy | — | The script failed with a `datasets.exceptions.DatasetNotFoundError`. The dataset 'rois-codh/kmnist' does not exist on the Hugging Face Hub. … |
| 8 | debug | 5 | buggy | — | The script crashed with a `datasets.exceptions.DatasetNotFoundError` while trying to download the EMNIST dataset. The dataset identifier 'Hu… |
| 9 | improve | 0 | good | Final Original D=69.1 | The experiment ran successfully, and the results provide preliminary evidence for the 'distributional oversmoothing' hypothesis. As predicte… |
| 10 | debug | 7 | buggy | — | The script failed with a `datasets.exceptions.DatasetNotFoundError`. The `load_dataset` function was unable to find the 'kmnist' dataset on … |
| 11 | debug | 8 | buggy | — | The script failed with a `datasets.exceptions.DatasetNotFoundError` because it could not find the `emnist` dataset on the Hugging Face Hub. … |
| 12 | improve | 0 | good | Original Distrib=66.2 | Hyperparam tuning name: weight_decay. My proposed solution will introduce a hyperparameter tuning loop for the `weight_decay` parameter. I w… |
| 13 | improve | 9 | good | Final Original D=73.2 | The experiment ran successfully and the results support the core hypothesis of "distributional oversmoothing." After achieving near-perfect … |
| 14 | improve | 9 | good | Final Original D=74.9 | Seed node |
| 15 | improve | 9 | good | Final Original D=73.3 | The experiment ran successfully, and the results align with the research hypothesis. The model initially achieved high accuracy (over 98%) o… |
| 16 | improve | 9 | good | — | Aggregate results from multiple seeds |

### Stage 3: creative_research 创造性探索

**16 节点**(good 3 / buggy 13)。搜索树:

```
[0] draft/good m=Final Original D=69.1  (根)
├─ [12] improve/buggy m=Final Original D=71.7
├─ [14] improve/good m=Final Original D=70.6
├─ [2] improve/buggy m=—
│  ├─ [5] debug/buggy m=—
│  │  └─ [7] debug/buggy m=Final Accuracy=0.566
│  │     └─ [10] debug/buggy m=—
│  └─ [4] debug/buggy m=Accuracy=70.7
├─ [13] improve/buggy m=Final Original D=71.5
├─ [15] improve/good m=—
├─ [9] improve/buggy m=—
├─ [8] improve/buggy m=—
└─ [1] improve/buggy m=—
   └─ [3] debug/buggy m=—
      └─ [6] debug/buggy m=—
         └─ [11] debug/buggy m=—
```

| idx | 动作 | 父 | 状态 | metric | agent 计划/分析摘要 |
|---|---|---|---|---|---|
| 0 | draft | — | good | Final Original D=69.1 | The experiment ran successfully, and the results provide preliminary evidence for the 'distributional oversmoothing' hypothesis. As predicte… |
| 1 | improve | 0 | buggy | — | The script failed with a `datasets.exceptions.DatasetNotFoundError` because the 'kmnist' dataset was not found on the Hugging Face Hub. Addi… |
| 2 | improve | 0 | buggy | — | The code failed to run because the KMNIST dataset is not available on the Hugging Face Hub. The `load_dataset` function threw a `DatasetNotF… |
| 3 | debug | 1 | buggy | — | The code failed with a `datasets.exceptions.DatasetNotFoundError`. The script attempts to load a dataset named 'kmnist' from the HuggingFace… |
| 4 | debug | 2 | buggy | Accuracy=70.7 | There is a minor bug in the final summary's print statements. The code checks if the accuracy and CDM values are instances of `float` before… |
| 5 | debug | 2 | buggy | — | The script failed to download the 'usps' dataset from the HuggingFace hub, resulting in a `DatasetNotFoundError`. The execution was halted w… |
| 6 | debug | 3 | buggy | — | The script failed with a `datasets.exceptions.DatasetNotFoundError`. It attempts to load the KMNIST dataset using an incorrect Hugging Face … |
| 7 | debug | 5 | buggy | Final Accuracy=0.566 | The script terminated with a `datasets.exceptions.DatasetNotFoundError` when trying to load the 'kmnist' dataset. The Hugging Face Hub datas… |
| 8 | improve | 0 | buggy | — | The script failed to execute due to a `datasets.exceptions.DatasetNotFoundError`. The dataset identifier 'rois-codh/kmnist' for the KMNIST d… |
| 9 | improve | 0 | buggy | — | The script failed with a `datasets.exceptions.DatasetNotFoundError`. The dataset 'rois-codh/kmnist' could not be found on the Hugging Face H… |
| 10 | debug | 7 | buggy | — | The execution failed due to a `datasets.exceptions.DatasetNotFoundError`. The script attempts to load a dataset named 'kuzushiji-mnist', whi… |
| 11 | debug | 6 | buggy | — | The script failed because the Hugging Face Hub identifier for the KMNIST dataset is incorrect. The code attempts to load a dataset named 'km… |
| 12 | improve | 0 | buggy | Final Original D=71.7 | The experiment aims to demonstrate "distributional oversmoothing," where adding diverse data sequentially degrades performance on the origin… |
| 13 | improve | 0 | buggy | Final Original D=71.5 | The experiment aims to demonstrate that performance on an original task continuously degrades as a model is trained on new, different data d… |
| 14 | improve | 0 | good | Final Original D=70.6 | The experiment executed successfully, and the results provide strong preliminary evidence for the research hypothesis. **Key Findings:** 1. … |
| 15 | improve | 0 | good | — | Aggregate results from multiple seeds |

### Stage 4: ablation 消融实验

**3 节点**(good 3 / buggy 0)。搜索树:

```
[0] draft/good m=Final Original D=69.1  (根)
├─ [1] improve/good m=Final Original D=63.9
└─ [2] improve/good m=Final Original D=70.5
```

| idx | 动作 | 父 | 状态 | metric | agent 计划/分析摘要 |
|---|---|---|---|---|---|
| 0 | draft | — | good | Final Original D=69.1 | The experiment ran successfully, and the results provide preliminary evidence for the 'distributional oversmoothing' hypothesis. As predicte… |
| 1 | improve | 0 | good | Final Original D=63.9 | Ablation name: Re-initialize Classifier Head Between Phases. My plan is to implement the "Re-initialize Classifier Head" ablation by modifyi… |
| 2 | improve | 0 | good | Final Original D=70.5 | The experiment ran successfully without any errors. The code correctly implements the ablation study, iterating through different optimizers… |

### Stage 4: ablation 消融实验

**18 节点**(good 13 / buggy 5)。搜索树:

```
[0] draft/good m=Final Original D=60.4  (根)
[1] draft/good m=Final Original D=70.6  (根)
[2] draft/buggy m=Final Original D=69.3  (根)
└─ [4] debug/buggy m=—
   └─ [6] debug/good m=Final Original D=40.5
[3] draft/good m=Final Original D=70.8  (根)
[5] draft/good m=Final Original D=66.2  (根)
[7] draft/good m=Final Original D=48.6  (根)
[8] draft/buggy m=accuracy_on_orig=76.6  (根)
└─ [10] debug/good m=Final Original D=53.9
[9] draft/good m=Final Original D=72.9  (根)
[11] draft/good m=Final Original D=99.4  (根)
[12] draft/good m=Final Original D=65.6  (根)
[13] draft/good m=Final Original D=66.9  (根)
[14] draft/buggy m=Best Final Origi=57.3  (根)
└─ [17] debug/good m=Final Accuracy o=28
[15] draft/buggy m=Final Task 1 tes=0  (根)
└─ [16] debug/good m=Final Original D=25.1
```

| idx | 动作 | 父 | 状态 | metric | agent 计划/分析摘要 |
|---|---|---|---|---|---|
| 0 | draft | — | good | Final Original D=60.4 | Ablation name: Ablate Optimizer: Adam vs. SGD. My plan is to introduce an outer loop that iterates through the optimizers to be tested, spec… |
| 1 | draft | — | good | Final Original D=70.6 | Ablation name: Ablate Activation Function: ReLU vs. LeakyReLU. My approach is to introduce a parameter in the `SimpleCNN` model to select th… |
| 2 | draft | — | buggy | Final Original D=69.3 | The code does not implement the specified research goals. The main objective of this sub-stage was to conduct a "Dataset Generalization Stud… |
| 3 | draft | — | good | Final Original D=70.8 | Ablation name: Ablate Model Capacity by Varying Filter Count. My approach is to modify the `SimpleCNN` model to accept the number of filters… |
| 4 | debug | 2 | buggy | — | The script failed with a `TypeError` in the data processing pipeline. The error occurs because the `RotatedDataset` class's transform compos… |
| 5 | draft | — | good | Final Original D=66.2 | Ablation name: Ablate Normalization Layer. My approach is to first define a new CNN model, `SimpleCNN_BN`, which is architecturally identica… |
| 6 | debug | 4 | good | Final Original D=40.5 | The code executed successfully, running the experiment on all three datasets (MNIST, FashionMNIST, and KMNIST) as intended. The results supp… |
| 7 | draft | — | good | Final Original D=48.6 | The code executed successfully, and the results align with the research hypothesis. The experiment demonstrates that introducing new data di… |
| 8 | draft | — | buggy | accuracy_on_orig=76.6 | The code does not meet the specified research goals for the current sub-stage. The objective was to integrate and test on Fashion-MNIST and … |
| 9 | draft | — | good | Final Original D=72.9 | The script executed successfully, and the results are consistent with the research hypothesis. The observed drop in accuracy is the phenomen… |
| 10 | debug | 8 | good | Final Original D=53.9 | The previous implementation was designed for a single dataset (MNIST) and did not meet the new requirements for a broader generalization stu… |
| 11 | draft | — | good | Final Original D=99.4 | Ablation name: Ablate by Introducing Data Rehearsal. My proposed solution involves modifying the training data composition for the second an… |
| 12 | draft | — | good | Final Original D=65.6 | The script executed successfully, completing the ablation study as intended. The results consistently demonstrate the hypothesized phenomeno… |
| 13 | draft | — | good | Final Original D=66.9 | Ablation name: Ablate Weight Initialization Scheme. My proposed solution introduces an ablation study to compare PyTorch's default weight in… |
| 14 | draft | — | buggy | Best Final Origi=57.3 | The final summary calculation for 'catastrophic forgetting' is bugged. The code uses the accuracy progression from the last experiment run (… |
| 15 | draft | — | buggy | Final Task 1 tes=0 | The experimental setup is flawed for the stated hypothesis. The goal is to investigate 'distributional oversmoothing,' a subtle phenomenon d… |
| 16 | debug | 15 | good | Final Original D=25.1 | The experiment ran successfully, and the results align with the research hypothesis. The code successfully integrated and ran the continual … |
| 17 | debug | 14 | good | Final Accuracy o=28 | The previous implementation conducted an incorrect ablation study on the number of classes instead of generalizing across different datasets… |

## C. 汇总统计

- **总节点数 60**(good 30 / buggy 30,成功率 50%)
- 最终论文 = 全部阶段最佳节点的实验结果聚合而成;上表每个 buggy 节点都经 agent 自我 debug(部分修复成功转 good,部分被裁剪)
- 动作推断规则:无父=draft(阶段根/seed),父 buggy→debug,父 good→improve