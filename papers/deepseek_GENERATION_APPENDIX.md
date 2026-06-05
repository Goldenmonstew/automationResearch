# 生成透明度附录:树搜索过程

> 本附录披露本论文由 **The AI Scientist v2** 自动生成的完整过程:走了哪些搜索策略、产生多少树节点、如何逐层选择/裁剪、agent 如何判断决策。
>
> **写作模型**: deepseek-v3.2

## A. 搜索策略概述

系统不使用人写模板,从一个研究 idea **零起点**出发,做 **4 阶段 best-first 并行树搜索**。每个节点是一份完整可执行的实验代码;agent 用三种原语扩展树:

- **draft(草稿)**:阶段根节点 — 从上一阶段最佳节点 seed,或全新起草一个实现方向
- **debug(调试)**:父节点运行**报错/buggy** → 读 traceback,定位并修复
- **improve(改进)**:父节点运行**成功** → 在其基础上调超参/换设计/加分析,争取更高指标

每个节点用**多 seed(本次 3 seed)**重复评估取稳定指标。每阶段结束按指标选最佳节点 seed 给下一阶段,其余分支被**裁剪**(不再扩展)。四阶段:initial→baseline_tuning→creative_research→ablation。

## B. 各阶段搜索树

### Stage 1: initial_implementation 初始实现

**7 节点**(good 5 / buggy 2)。搜索树:

```
[0] draft/good m=—  (根)
[1] draft/good m=test accuracy=100  (根)
├─ [4] improve/buggy m=—
├─ [5] improve/good m=test accuracy=100
├─ [6] improve/good m=—
└─ [3] improve/buggy m=test accuracy=100
[2] draft/good m=—  (根)
```

| idx | 动作 | 父 | 状态 | metric | agent 计划/分析摘要 |
|---|---|---|---|---|---|
| 0 | draft | — | good | — | Assume all packages are already installed. ## Code To Use You can use the following code as a starting point. It is a simple continual learn… |
| 1 | draft | — | good | test accuracy=100 | I'll implement a simple experiment to test the distributional oversmoothing hypothesis in continual learning. First, I'll create synthetic d… |
| 2 | draft | — | good | — | Please note that the environment is offline, so you cannot download any external data. You can use built-in datasets (e.g., from torchvision… |
| 3 | improve | 1 | buggy | test accuracy=100 | The execution output reveals a critical bug: the model is achieving 100% accuracy across all distributions in all experiments, which makes i… |
| 4 | improve | 1 | buggy | — | Seed node |
| 5 | improve | 1 | good | test accuracy=100 | Seed node |
| 6 | improve | 1 | good | — | Aggregate results from multiple seeds |

### Stage 2: baseline_tuning 基线调参

**17 节点**(good 7 / buggy 10)。搜索树:

```
[0] draft/good m=test accuracy=100  (根)
├─ [1] improve/buggy m=—
│  └─ [6] debug/good m=Baseline test ac=100
│     ├─ [16] improve/good m=—
│     ├─ [13] improve/good m=baseline test ac=100
│     ├─ [15] improve/good m=Baseline test ac=100
│     └─ [14] improve/good m=Distribution 1 f=99.9
├─ [7] improve/buggy m=—
│  └─ [10] debug/buggy m=test accuracy=100
├─ [11] improve/buggy m=—
├─ [2] improve/buggy m=—
│  ├─ [4] debug/buggy m=test accuracy=58.3
│  │  └─ [5] debug/buggy m=—
│  │     └─ [8] debug/good m=test_accuracy=59.5
│  └─ [3] debug/buggy m=—
│     └─ [9] debug/buggy m=—
└─ [12] improve/buggy m=—
```

| idx | 动作 | 父 | 状态 | metric | agent 计划/分析摘要 |
|---|---|---|---|---|---|
| 0 | draft | — | good | test accuracy=100 | I'll implement a simple experiment to test the distributional oversmoothing hypothesis in continual learning. First, I'll create synthetic d… |
| 1 | improve | 0 | buggy | — | The execution failed due to a typo in the model definition. In line 163, the variable 'hidden432dim' is referenced but not defined. It shoul… |
| 2 | improve | 0 | buggy | — | The execution failed due to a NameError in the SimpleNN class definition. The variable `hidden_flim` is referenced but not defined - it shou… |
| 3 | debug | 2 | buggy | — | The execution output shows a SyntaxError at line 1 of the runfile.py file. The error indicates that the Python interpreter encountered inval… |
| 4 | debug | 2 | buggy | test accuracy=58.3 | The execution output reveals a critical bug in the learning rate tuning phase: the validation loss is reported as 0.0000 for learning rates … |
| 5 | debug | 4 | buggy | — | The execution output shows a critical bug: an ImportError due to incompatible huggingface-hub version. The transformers library requires hug… |
| 6 | debug | 1 | good | Baseline test ac=100 | The bug in the previous implementation is a typographical error in the SimpleNN class initialization where `hidden432dim` should be `hidden_… |
| 7 | improve | 0 | buggy | — | The execution failed due to a NameError: 'test_1loader1' is not defined. This occurs at line 279 where the code attempts to use a variable n… |
| 8 | debug | 5 | good | test_accuracy=59.5 | The bug is caused by a version conflict with the `huggingface-hub` package. The error message indicates that `huggingface-hub>=1.5.0` is req… |
| 9 | debug | 3 | buggy | — | The code execution failed due to a DatasetNotFoundError when trying to load the STL10 dataset from HuggingFace. The dataset identifier 'stan… |
| 10 | debug | 7 | buggy | test accuracy=100 | The execution output reveals several bugs: 1. **Dataset Loading Errors**: The HuggingFace datasets failed to load properly. The Iris dataset… |
| 11 | improve | 0 | buggy | — | The execution output shows a critical bug in the code. The main issue is that the model is achieving 100% accuracy on all distributions for … |
| 12 | improve | 0 | buggy | — | The execution failed due to a typo in the variable name. Line 249 has 'baseline_predاف' instead of 'baseline_preds'. This appears to be a Un… |
| 13 | improve | 6 | good | baseline test ac=100 | Seed node |
| 14 | improve | 6 | good | Distribution 1 f=99.9 | Seed node |
| 15 | improve | 6 | good | Baseline test ac=100 | Seed node |
| 16 | improve | 6 | good | — | Aggregate results from multiple seeds |

### Stage 3: creative_research 创造性探索

**17 节点**(good 6 / buggy 11)。搜索树:

```
[0] draft/good m=Baseline test ac=100  (根)
├─ [2] improve/buggy m=—
│  └─ [4] debug/buggy m=—
│     └─ [6] debug/buggy m=—
│        └─ [10] debug/buggy m=—
├─ [13] improve/good m=Baseline test ac=100
├─ [12] improve/buggy m=—
├─ [8] improve/buggy m=—
├─ [15] improve/good m=baseline test ac=100
├─ [5] improve/buggy m=—
│  └─ [9] debug/buggy m=—
├─ [11] improve/buggy m=—
├─ [16] improve/good m=—
├─ [7] improve/buggy m=—
├─ [3] improve/buggy m=—
├─ [1] improve/good m=training accurac=93.8
└─ [14] improve/good m=training accurac=100
```

| idx | 动作 | 父 | 状态 | metric | agent 计划/分析摘要 |
|---|---|---|---|---|---|
| 0 | draft | — | good | Baseline test ac=100 | The bug in the previous implementation is a typographical error in the SimpleNN class initialization where `hidden432dim` should be `hidden_… |
| 1 | improve | 0 | good | training accurac=93.8 | To investigate distributional oversmoothing in continual learning, I propose enhancing our analysis with three key improvements: First, intr… |
| 2 | improve | 0 | buggy | — | The execution failed with a RuntimeError: "mat1 and mat2 shapes cannot be multiplied (64x3072 and 784x128)". This occurs when training on th… |
| 3 | improve | 0 | buggy | — | The execution failed due to a shape mismatch error in the CNN model. The error occurs when computing the linear layer: "mat1 and mat2 shapes… |
| 4 | debug | 2 | buggy | — | The execution failed due to a RuntimeError when loading state_dict for the FlexibleNN model. The error occurs during Experiment 2 (Sequentia… |
| 5 | improve | 0 | buggy | — | The execution output shows a KeyError: 'lr_1e-05' at line 578. This occurs because the learning rate 0.00001 (1e-05) is being used as a key … |
| 6 | debug | 4 | buggy | — | The execution output shows a NameError in the compute_dsi function: "NameError: name 'class_sort' is not defined". This bug occurs because t… |
| 7 | improve | 0 | buggy | — | The execution failed with a RuntimeError due to shape mismatch when training on the second dataset (EMNIST). The error occurred because the … |
| 8 | improve | 0 | buggy | — | The execution failed due to a NameError: 'train_test_split' is not defined. This occurred when the code tried to create synthetic datasets a… |
| 9 | debug | 5 | buggy | — | The execution output shows a bug in the code where there is a size mismatch error when trying to transfer weights from a model trained on Fa… |
| 10 | debug | 6 | buggy | — | The code execution failed with a NameError: 'hidden1_dim' is not defined. This occurs in the FlexibleNN class constructor where self.fc2 = n… |
| 11 | improve | 0 | buggy | — | The execution output shows a bug in the code. The error is an IndexError: list index out of range at line 436 in run_sequential_experiment f… |
| 12 | improve | 0 | buggy | — | The execution output shows a SyntaxError at the beginning of the code. The issue is that the code starts with a descriptive text paragraph r… |
| 13 | improve | 0 | good | Baseline test ac=100 | Seed node |
| 14 | improve | 0 | good | training accurac=100 | Seed node |
| 15 | improve | 0 | good | baseline test ac=100 | Seed node |
| 16 | improve | 0 | good | — | Aggregate results from multiple seeds |

### Stage 4: ablation 消融实验

**23 节点**(good 8 / buggy 15)。搜索树:

```
[0] draft/good m=Baseline test ac=100  (根)
├─ [16] improve/buggy m=—
├─ [22] improve/good m=—
├─ [19] improve/good m=baseline test ac=100
├─ [9] improve/buggy m=—
├─ [12] improve/buggy m=—
├─ [14] improve/buggy m=—
│  └─ [15] debug/buggy m=—
├─ [2] improve/buggy m=—
│  ├─ [4] debug/buggy m=test accuracy=1
│  │  └─ [5] debug/good m=test accuracy=1
│  └─ [3] debug/buggy m=—
│     └─ [7] debug/buggy m=—
│        └─ [18] debug/buggy m=—
├─ [21] improve/good m=training accurac=100
├─ [11] improve/buggy m=—
├─ [13] improve/buggy m=—
│  └─ [17] debug/buggy m=—
├─ [1] improve/good m=Baseline perform=100
├─ [8] improve/buggy m=baseline test ac=90
├─ [20] improve/good m=Baseline test ac=100
├─ [10] improve/buggy m=training accurac=100
└─ [6] improve/good m=validation loss=0.0017
```

| idx | 动作 | 父 | 状态 | metric | agent 计划/分析摘要 |
|---|---|---|---|---|---|
| 0 | draft | — | good | Baseline test ac=100 | The bug in the previous implementation is a typographical error in the SimpleNN class initialization where `hidden432dim` should be `hidden_… |
| 1 | improve | 0 | good | Baseline perform=100 | Ablation name: <ablation name>. I'll implement an ablation study on the learning rate to analyze its impact on distributional degradation ra… |
| 2 | improve | 0 | buggy | — | The execution failed due to a syntax error in the Python code. The issue is that the code block starts with a natural language description (… |
| 3 | debug | 2 | buggy | — | The execution output shows a bug in the code. There's a NameError: 'output2' is not defined in the SimpleNN class initialization. The code t… |
| 4 | debug | 2 | buggy | test accuracy=1 | The execution failed with a TypeError at the end of the script. The error occurs in the plotting section where the code tries to access `euc… |
| 5 | debug | 4 | good | test accuracy=1 | The bug occurs because the experiment_data structure stores loss values incorrectly for ablation study comparison. In the plotting section, … |
| 6 | improve | 0 | good | validation loss=0.0017 | Ablation name: Layer Normalization Replacement for Batch Normalization. I'll implement an ablation study comparing Batch Normalization vs La… |
| 7 | debug | 3 | buggy | — | The execution failed due to a NameError: 'train_metric' is not defined. The code attempts to append train_metric to experiment_data, but thi… |
| 8 | improve | 0 | buggy | baseline test ac=90 | The execution output reveals several critical bugs and issues: 1. **Inconsistent and Illogical DDR Values**: The Distributional Degradation … |
| 9 | improve | 0 | buggy | — | The execution failed due to a NameError: `train_1loader` is not defined. The code attempts to use `train_1loader` on line 489, but the corre… |
| 10 | improve | 0 | buggy | training accurac=100 | The execution output reveals a critical bug: the models are achieving 100% accuracy on all distributions, making it impossible to detect dis… |
| 11 | improve | 0 | buggy | — | The execution failed due to a NameError in the SimpleNNNoDropout class initialization. The variable 'output_dim' is referenced but not defin… |
| 12 | improve | 0 | buggy | — | The execution output shows a syntax error in the code. The error occurs at line 2 with "SyntaxError: unterminated string literal (detected a… |
| 13 | improve | 0 | buggy | — | The execution output shows a bug in the code. The error occurs when training with a replay buffer size of 0.1 (10%). The error message is: "… |
| 14 | improve | 0 | buggy | — | The execution output shows a critical bug in the code. The error occurs in the `train_with_replay_buffer` function when trying to use the re… |
| 15 | debug | 14 | buggy | — | The execution failed due to a device mismatch error in the compute_dci function. The error occurs because the previous_model is on CPU while… |
| 16 | improve | 0 | buggy | — | The execution output shows no code was provided for execution. The user provided an empty Python code block with only triple backticks and n… |
| 17 | debug | 13 | buggy | — | The execution output shows a bug in the code. There's a device mismatch error when computing DCI (Distributional Confusion Index). The error… |
| 18 | debug | 7 | buggy | — | The execution output shows a bug in the code. The error occurs at line 92: "TypeError: list indices must be integers or slices, not list". T… |
| 19 | improve | 0 | good | baseline test ac=100 | Seed node |
| 20 | improve | 0 | good | Baseline test ac=100 | Seed node |
| 21 | improve | 0 | good | training accurac=100 | Seed node |
| 22 | improve | 0 | good | — | Aggregate results from multiple seeds |

## C. 汇总统计

- **总节点数 64**(good 26 / buggy 38,成功率 41%)
- 最终论文 = 全部阶段最佳节点的实验结果聚合而成;上表每个 buggy 节点都经 agent 自我 debug(部分修复成功转 good,部分被裁剪)
- 动作推断规则:无父=draft(阶段根/seed),父 buggy→debug,父 good→improve