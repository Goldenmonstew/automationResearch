# 生成透明度附录:树搜索过程

> 本附录披露本论文由 **The AI Scientist v2** 自动生成的完整过程:走了哪些搜索策略、产生多少树节点、如何逐层选择/裁剪、agent 如何判断决策。
>
> **写作模型**: gpt-4o

## A. 搜索策略概述

系统不使用人写模板,从一个研究 idea **零起点**出发,做 **4 阶段 best-first 并行树搜索**。每个节点是一份完整可执行的实验代码;agent 用三种原语扩展树:

- **draft(草稿)**:阶段根节点 — 从上一阶段最佳节点 seed,或全新起草一个实现方向
- **debug(调试)**:父节点运行**报错/buggy** → 读 traceback,定位并修复
- **improve(改进)**:父节点运行**成功** → 在其基础上调超参/换设计/加分析,争取更高指标

每个节点用**多 seed(本次 3 seed)**重复评估取稳定指标。每阶段结束按指标选最佳节点 seed 给下一阶段,其余分支被**裁剪**(不再扩展)。四阶段:initial→baseline_tuning→creative_research→ablation。

## B. 各阶段搜索树

### Stage 1: initial_implementation 初始实现

**6 节点**(good 5 / buggy 1)。搜索树:

```
[0] draft/good m=IFD score=15.2  (根)
├─ [4] improve/good m=IFD score=14.9
├─ [5] improve/good m=—
├─ [2] improve/good m=IFD score=15
└─ [3] improve/good m=IFD score=15.5
[1] draft/buggy m=—  (根)
```

| idx | 动作 | 父 | 状态 | metric | agent 计划/分析摘要 |
|---|---|---|---|---|---|
| 0 | draft | — | good | IFD score=15.2 | To implement the first stage of this experiment, we'll use a simple synthetic dataset where features and labels are partitioned into "distri… |
| 1 | draft | — | buggy | — | The error occurs because the number of informative, redundant, and repeated features must sum to less than the total number of features when… |
| 2 | improve | 0 | good | IFD score=15 | Seed node |
| 3 | improve | 0 | good | IFD score=15.5 | Seed node |
| 4 | improve | 0 | good | IFD score=14.9 | Seed node |
| 5 | improve | 0 | good | — | Aggregate results from multiple seeds |

### Stage 2: baseline_tuning 基线调参

**17 节点**(good 15 / buggy 2)。搜索树:

```
[0] draft/good m=IFD score=15.2  (根)
├─ [6] improve/good m=Train Loss=0.26
├─ [5] improve/good m=Train Loss=0.357
├─ [10] improve/good m=Final Training L=147
├─ [7] improve/buggy m=—
│  └─ [11] debug/buggy m=Intra-class Feat=14.5
├─ [2] improve/good m=training loss=11.7
├─ [8] improve/good m=Final IFD Value=15.8
├─ [12] improve/good m=IFD (Inter-class=13.1
├─ [9] improve/good m=ifd=17.7
├─ [4] improve/good m=train loss=0.0497
│  ├─ [14] improve/good m=train loss=0.0425
│  ├─ [15] improve/good m=train loss=0.0587
│  ├─ [16] improve/good m=—
│  └─ [13] improve/good m=train loss=0.0577
├─ [3] improve/good m=Intra-Feature Di=14
└─ [1] improve/good m=train loss=0.0988
```

| idx | 动作 | 父 | 状态 | metric | agent 计划/分析摘要 |
|---|---|---|---|---|---|
| 0 | draft | — | good | IFD score=15.2 | To implement the first stage of this experiment, we'll use a simple synthetic dataset where features and labels are partitioned into "distri… |
| 1 | improve | 0 | good | train loss=0.0988 | Hyperparam tuning name: epochs. To implement hyperparameter tuning for the number of epochs, I will introduce a loop to iterate over several… |
| 2 | improve | 0 | good | training loss=11.7 | Hyperparam tuning name: learning_rate. To implement hyperparameter tuning for the learning rate, I will modify the training loop to test mul… |
| 3 | improve | 0 | good | Intra-Feature Di=14 | Hyperparam tuning name: batch_size. To implement hyperparameter tuning for batch size, I will set up an experiment loop to test multiple bat… |
| 4 | improve | 0 | good | train loss=0.0497 | Hyperparam tuning name: Hidden Layer Size. To implement hyperparameter tuning for the hidden layer size, I will define a range of values for… |
| 5 | improve | 0 | good | Train Loss=0.357 | Hyperparam tuning name: Activation Function. To implement hyperparameter tuning for activation functions, we will modify the `SimpleMLP` cla… |
| 6 | improve | 0 | good | Train Loss=0.26 | Hyperparam tuning name: Weight Initialization. To implement hyperparameter tuning for weight initialization, I will add functions to initial… |
| 7 | improve | 0 | buggy | — | The code execution failed because the 'optuna' module is not installed. Optuna is required for hyperparameter tuning in the script. To fix t… |
| 8 | improve | 0 | good | Final IFD Value=15.8 | Hyperparam tuning name: Optimizer Type. To implement hyperparameter tuning for the optimizer type, I will extend the experiment loop to iter… |
| 9 | improve | 0 | good | ifd=17.7 | Hyperparam tuning name: Number of Layers. To implement hyperparameter tuning for the number of layers, I will modify the `SimpleMLP` model t… |
| 10 | improve | 0 | good | Final Training L=147 | The execution of the training script was successful. The model was trained on two synthetic data distributions, and the training loss decrea… |
| 11 | debug | 7 | buggy | Intra-class Feat=14.5 | The execution output reveals a significant issue with the TSDS metric computation. The TSDS values are negative, which is unexpected for a m… |
| 12 | improve | 0 | good | IFD (Inter-class=13.1 | Hyperparam tuning name: Batch Normalization. We will introduce batch normalization between the layers of the model and conduct hyperparamete… |
| 13 | improve | 4 | good | train loss=0.0577 | Seed node |
| 14 | improve | 4 | good | train loss=0.0425 | Seed node |
| 15 | improve | 4 | good | train loss=0.0587 | Seed node |
| 16 | improve | 4 | good | — | Aggregate results from multiple seeds |

### Stage 3: creative_research 创造性探索

**17 节点**(good 5 / buggy 12)。搜索树:

```
[0] draft/good m=train loss=0.0497  (根)
├─ [8] improve/buggy m=—
├─ [13] improve/good m=train loss=0.0577
├─ [7] improve/buggy m=—
├─ [4] improve/buggy m=—
│  └─ [11] debug/buggy m=—
├─ [14] improve/good m=train loss=0.0425
├─ [16] improve/good m=—
├─ [5] improve/buggy m=—
│  ├─ [9] debug/buggy m=—
│  └─ [10] debug/buggy m=—
├─ [15] improve/good m=train loss=0.0587
├─ [2] improve/buggy m=—
│  └─ [3] debug/buggy m=—
├─ [1] improve/buggy m=—
├─ [6] improve/buggy m=—
└─ [12] improve/buggy m=—
```

| idx | 动作 | 父 | 状态 | metric | agent 计划/分析摘要 |
|---|---|---|---|---|---|
| 0 | draft | — | good | train loss=0.0497 | Hyperparam tuning name: Hidden Layer Size. To implement hyperparameter tuning for the hidden layer size, I will define a range of values for… |
| 1 | improve | 0 | buggy | — | The execution failed due to a missing dataset. Specifically, the 'emnist' dataset with the name 'letters' could not be found on the HuggingF… |
| 2 | improve | 0 | buggy | — | The script fails because the dataset 'iris' is not available on the HuggingFace Hub. The error occurs when attempting to load the 'iris' dat… |
| 3 | debug | 2 | buggy | — | The script encountered an error during training due to an incorrect assumption about the structure of the DataLoader batches. Specifically, … |
| 4 | improve | 0 | buggy | — | The execution failed due to a KeyError when attempting to access the 'image' key in the dataset. This indicates that the dataset structure m… |
| 5 | improve | 0 | buggy | — | The script encountered a bug during execution. Specifically, the `load_hf_dataset` function attempts to convert images from the HuggingFace … |
| 6 | improve | 0 | buggy | — | The error occurs because the datasets module from torchvision is being treated as a callable object when it is actually a dictionary-like ob… |
| 7 | improve | 0 | buggy | — | The script fails to load the 'domainnet' dataset from the HuggingFace Hub, resulting in a 'DatasetNotFoundError'. This issue arises because … |
| 8 | improve | 0 | buggy | — | The execution failed due to a DatasetNotFoundError for the 'emnist' dataset. The error indicates that the dataset 'emnist' doesn't exist on … |
| 9 | debug | 5 | buggy | — | The script encountered a bug while attempting to load the 'cifar10' dataset using the HuggingFace datasets library. Specifically, the error … |
| 10 | debug | 5 | buggy | — | The script encountered a KeyError when attempting to access the 'image' key in the 'cifar10' dataset. This issue arises because the dataset … |
| 11 | debug | 4 | buggy | — | The script encountered a bug while trying to load the CIFAR-10 dataset. The error message indicates an 'Unexpected data format' for the data… |
| 12 | improve | 0 | buggy | — | The execution failed due to a KeyError: 'image'. This occurred because the dataset loaded from the Hugging Face library does not contain an … |
| 13 | improve | 0 | good | train loss=0.0577 | Seed node |
| 14 | improve | 0 | good | train loss=0.0425 | Seed node |
| 15 | improve | 0 | good | train loss=0.0587 | Seed node |
| 16 | improve | 0 | good | — | Aggregate results from multiple seeds |

### Stage 4: ablation 消融实验

**23 节点**(good 18 / buggy 5)。搜索树:

```
[0] draft/good m=train loss=0.0497  (根)
├─ [21] improve/good m=train loss=0.0587
├─ [2] improve/buggy m=—
│  └─ [4] debug/good m=IFD Metric=14
├─ [6] improve/buggy m=IFD Metric=7.19
│  └─ [9] debug/good m=validation_loss=42.8
├─ [12] improve/buggy m=—
│  └─ [14] debug/good m=Train Loss=1.47
├─ [15] improve/buggy m=—
├─ [13] improve/good m=train loss=0.18
├─ [19] improve/good m=train loss=0.0577
├─ [22] improve/good m=—
├─ [18] improve/good m=Training IFD=15.1
├─ [8] improve/good m=IFD=7.12
├─ [10] improve/good m=IFD Metric=10.3
├─ [5] improve/good m=Train Loss=48.7
├─ [1] improve/good m=Ifd (e.g., IFD)=25.3
├─ [17] improve/good m=IFD Metric=14.6
├─ [3] improve/good m=training loss=0.575
├─ [16] improve/good m=IFD Metric (Inte=26.3
├─ [11] improve/buggy m=—
├─ [7] improve/good m=Final Training L=0.469
└─ [20] improve/good m=train loss=0.0587
```

| idx | 动作 | 父 | 状态 | metric | agent 计划/分析摘要 |
|---|---|---|---|---|---|
| 0 | draft | — | good | train loss=0.0497 | Hyperparam tuning name: Hidden Layer Size. To implement hyperparameter tuning for the hidden layer size, I will define a range of values for… |
| 1 | improve | 0 | good | Ifd (e.g., IFD)=25.3 | Ablation name: Activation Function Ablation. To implement the activation function ablation study, I will modify the `SimpleMLP` class to all… |
| 2 | improve | 0 | buggy | — | Ablation name: Optimizer Type Ablation. To implement the optimizer type ablation study, the experiment should evaluate the model's performan… |
| 3 | improve | 0 | good | training loss=0.575 | Ablation name: Synthetic Dataset Distribution Variation. To implement the ablation study, I will introduce an additional synthetic dataset w… |
| 4 | debug | 2 | good | IFD Metric=14 | The bug in the previous implementation lies in not properly capturing representation geometry preservation across distributions. The Represe… |
| 5 | improve | 0 | good | Train Loss=48.7 | Ablation name: Learning Rate Ablation. The ablation study modifies the baseline implementation to vary learning rates systematically (e.g., … |
| 6 | improve | 0 | buggy | IFD Metric=7.19 | The training script contains a bug in the computation of the IFD (Inter-Class Feature Distance) metric. Specifically, the code attempts to s… |
| 7 | improve | 0 | good | Final Training L=0.469 | Ablation name: Batch Size Ablation. To implement the Batch Size Ablation study, I will modify the base code to allow evaluating different ba… |
| 8 | improve | 0 | good | IFD=7.12 | Ablation name: Weight Initialization Ablation. To implement the ablation study, we'll augment the provided script to investigate the impact … |
| 9 | debug | 6 | good | validation_loss=42.8 | The issue in the previous implementation is that the intermediate feature extraction used in computing the IFD metric assumed feature slicin… |
| 10 | improve | 0 | good | IFD Metric=10.3 | Ablation name: Feature Dimensionality Ablation. To perform the Feature Dimensionality Ablation, we will loop through increasing feature dime… |
| 11 | improve | 0 | buggy | — | The execution failed due to a mismatch in tensor dimensions when using the MeanSquaredError loss function. Specifically, the model's output … |
| 12 | improve | 0 | buggy | — | The code fails with an AttributeError because it attempts to call the 'unique' method on a numpy.ndarray object, which does not have this me… |
| 13 | improve | 0 | good | train loss=0.18 | Ablation name: Output Layer Size Ablation. To implement the ablation study for output layer size, we will modify the base code so that it ex… |
| 14 | debug | 12 | good | Train Loss=1.47 | To fix the `AttributeError` in the previous implementation, we need to replace the `unique()` method call on a numpy array with an equivalen… |
| 15 | improve | 0 | buggy | — | The error occurs in the forward method of the SimpleMLP class. Specifically, the issue lies in the line `return self.fc1(x), self.fc2(x)`. H… |
| 16 | improve | 0 | good | IFD Metric (Inte=26.3 | Ablation name: Gradient Clipping Ablation. To add the gradient clipping ablation study, I will modify the training process to include gradie… |
| 17 | improve | 0 | good | IFD Metric=14.6 | Ablation name: Data Augmentation Technique Ablation. To implement the ablation study for the impact of data augmentation techniques (Gaussia… |
| 18 | improve | 0 | good | Training IFD=15.1 | Ablation name: Loss Backpropagation Scheduling Ablation. To implement the loss backpropagation scheduling ablation, I will modify the traini… |
| 19 | improve | 0 | good | train loss=0.0577 | Seed node |
| 20 | improve | 0 | good | train loss=0.0587 | Seed node |
| 21 | improve | 0 | good | train loss=0.0587 | Seed node |
| 22 | improve | 0 | good | — | Aggregate results from multiple seeds |

## C. 汇总统计

- **总节点数 63**(good 43 / buggy 20,成功率 68%)
- 最终论文 = 全部阶段最佳节点的实验结果聚合而成;上表每个 buggy 节点都经 agent 自我 debug(部分修复成功转 good,部分被裁剪)
- 动作推断规则:无父=draft(阶段根/seed),父 buggy→debug,父 good→improve