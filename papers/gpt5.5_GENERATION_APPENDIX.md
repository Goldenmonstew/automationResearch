# 生成透明度附录:树搜索过程

> 本附录披露本论文由 **The AI Scientist v2** 自动生成的完整过程:走了哪些搜索策略、产生多少树节点、如何逐层选择/裁剪、agent 如何判断决策。
>
> **写作模型**: gpt-5.5

## A. 搜索策略概述

系统不使用人写模板,从一个研究 idea **零起点**出发,做 **4 阶段 best-first 并行树搜索**。每个节点是一份完整可执行的实验代码;agent 用三种原语扩展树:

- **draft(草稿)**:阶段根节点 — 从上一阶段最佳节点 seed,或全新起草一个实现方向
- **debug(调试)**:父节点运行**报错/buggy** → 读 traceback,定位并修复
- **improve(改进)**:父节点运行**成功** → 在其基础上调超参/换设计/加分析,争取更高指标

每个节点用**多 seed(本次 3 seed)**重复评估取稳定指标。每阶段结束按指标选最佳节点 seed 给下一阶段,其余分支被**裁剪**(不再扩展)。四阶段:initial→baseline_tuning→creative_research→ablation。

## B. 各阶段搜索树

### Stage 1: initial_implementation 初始实现

**6 节点**(good 6 / buggy 0)。搜索树:

```
[0] draft/good m=train accuracy=0.999  (根)
├─ [3] improve/good m=train accuracy=0.999
├─ [2] improve/good m=train accuracy=0.999
├─ [4] improve/good m=train accuracy=0.999
└─ [5] improve/good m=—
[1] draft/good m=original validat=0.944  (根)
```

| idx | 动作 | 父 | 状态 | metric | agent 计划/分析摘要 |
|---|---|---|---|---|---|
| 0 | draft | — | good | train accuracy=0.999 | Code-only draft (model emitted no separate plan). |
| 1 | draft | — | good | original validat=0.944 | Code-only draft (model emitted no separate plan). |
| 2 | improve | 0 | good | train accuracy=0.999 | Seed node |
| 3 | improve | 0 | good | train accuracy=0.999 | Seed node |
| 4 | improve | 0 | good | train accuracy=0.999 | Seed node |
| 5 | improve | 0 | good | — | Aggregate results from multiple seeds |

### Stage 2: baseline_tuning 基线调参

**17 节点**(good 9 / buggy 8)。搜索树:

```
[0] draft/good m=train accuracy=0.999  (根)
[1] draft/buggy m=train accuracy=0.545  (根)
├─ [3] debug/good m=train accuracy=0.789
└─ [4] debug/good m=original distrib=0.905
   ├─ [15] improve/buggy m=original distrib=0.905
   ├─ [16] improve/good m=—
   ├─ [14] improve/good m=original distrib=0.905
   └─ [13] improve/buggy m=original distrib=0.905
[2] draft/buggy m=original validat=0.961  (根)
└─ [8] debug/good m=anchor original =0.905
[5] draft/buggy m=ODAD=0.0111  (根)
└─ [12] debug/good m=Distributional O=0.0729
[6] draft/buggy m=original validat=0.947  (根)
[7] draft/buggy m=final original v=0.944  (根)
└─ [9] debug/good m=final Distributi=0.0406
[10] draft/buggy m=final ODAD=0.013  (根)
└─ [11] debug/good m=anchor original-=0.86
```

| idx | 动作 | 父 | 状态 | metric | agent 计划/分析摘要 |
|---|---|---|---|---|---|
| 0 | draft | — | good | train accuracy=0.999 | Code-only draft (model emitted no separate plan). |
| 1 | draft | — | buggy | train accuracy=0.545 | The run completed without a runtime crash, but the experiment has serious implementation/protocol bugs relative to the stated sub-stage goal… |
| 2 | draft | — | buggy | original validat=0.961 | The run completed without crashing, but the experiment implementation has major protocol bugs. First, despite the sub-stage requirement, it … |
| 3 | debug | 1 | good | train accuracy=0.789 | The previous run likely under-tested the hypothesis because the rotated distributions were normalized with per-pixel statistics from the unr… |
| 4 | debug | 1 | good | original distrib=0.905 | The previous run likely failed to learn the rotated distributions because normalization divided rotated edge pixels by near-zero standard de… |
| 5 | draft | — | buggy | ODAD=0.0111 | Execution completed without crashing, but the experiment has major methodological/implementation bugs. The sub-stage explicitly required int… |
| 6 | draft | — | buggy | original validat=0.947 | The script executed to completion and saved results, but there are substantive experimental/implementation bugs relative to the stated sub-s… |
| 7 | draft | — | buggy | final original v=0.944 | The run completed without crashing, but the experiment implementation has substantive bugs relative to the stated sub-stage goals and likely… |
| 8 | debug | 2 | good | anchor original =0.905 | The main fix is to replace the unstable per-pixel normalization from the original digits distribution with robust scalar normalization fitte… |
| 9 | debug | 7 | good | final Distributi=0.0406 | The main bug was the per-pixel normalization computed only from the original digits distribution: many 8x8 border pixels have near-zero vari… |
| 10 | draft | — | buggy | final ODAD=0.013 | The run completed without a runtime crash, but the experiment implementation has major protocol bugs relative to the stage goals and likely … |
| 11 | debug | 10 | good | anchor original-=0.86 | The main bug was the per-pixel normalization inherited from the original digits distribution: rotated/noisy images activated pixels whose or… |
| 12 | debug | 5 | good | Distributional O=0.0729 | The main bug was unsafe per-pixel normalization fitted only on the original digit distribution, which made rotated/noisy images explode at l… |
| 13 | improve | 4 | buggy | original distrib=0.905 | The run completed and saved artifacts, but the log/code reveal evaluation bugs and experimental-design issues. ODAD is computed against a mo… |
| 14 | improve | 4 | good | original distrib=0.905 | Seed node |
| 15 | improve | 4 | buggy | original distrib=0.905 | The run completed and saved artifacts, but there are experimental/metric bugs. Most importantly, ODAD is computed against a moving anchor du… |
| 16 | improve | 4 | good | — | Aggregate results from multiple seeds |

### Stage 3: creative_research 创造性探索

**17 节点**(good 9 / buggy 8)。搜索树:

```
[0] draft/good m=original distrib=0.905  (根)
├─ [1] improve/buggy m=train accuracy=0.534
│  ├─ [4] debug/good m=original accurac=0.642
│  └─ [3] debug/good m=training accurac=0.502
├─ [16] improve/good m=—
├─ [9] improve/buggy m=—
│  └─ [12] debug/good m=final accuracy b=0.626
├─ [2] improve/buggy m=—
│  └─ [5] debug/buggy m=—
│     └─ [7] debug/good m=original accurac=0.656
├─ [14] improve/buggy m=original distrib=0.905
├─ [11] improve/good m=original accurac=0.571
├─ [15] improve/buggy m=original distrib=0.905
├─ [13] improve/buggy m=original distrib=0.905
├─ [6] improve/buggy m=best train accur=0.792
│  └─ [8] debug/good m=final original a=0.681
└─ [10] improve/good m=best train accur=0.766
```

| idx | 动作 | 父 | 状态 | metric | agent 计划/分析摘要 |
|---|---|---|---|---|---|
| 0 | draft | — | good | original distrib=0.905 | The previous run likely failed to learn the rotated distributions because normalization divided rotated edge pixels by near-zero standard de… |
| 1 | improve | 0 | buggy | train accuracy=0.534 | The script completed, but the execution log is truncated, so not all dataset/method outcomes are visible. More importantly, there is a metho… |
| 2 | improve | 0 | buggy | — | Execution failed before any experiment ran. In transform_images(), the RNG seed is computed as seed + int(angle * 10) + int(noise * 1000). F… |
| 3 | debug | 1 | good | training accurac=0.502 | The fix locks old-domain anchors immediately after each domain is learned and computes DSI against those frozen anchors for every previously… |
| 4 | debug | 1 | good | original accurac=0.642 | The fix freezes a true post-original anchor model/geometry snapshot and computes DSI from L2-normalized representation geometry, avoiding th… |
| 5 | debug | 2 | buggy | — | Execution failed after epoch 5 on the first dataset/variant due to a CKA shape mismatch. `anchor_features` is computed from the original tra… |
| 6 | improve | 0 | buggy | best train accur=0.792 | The run completed and saved outputs, but there are experiment-validity issues. The log is truncated, so full MNIST/Fashion-MNIST results are… |
| 7 | debug | 5 | good | original accurac=0.656 | The bug occurs because CKA was computed between feature matrices from different sample sets: the anchor features came from the original trai… |
| 8 | debug | 6 | good | final original a=0.681 | I will fix the experiment by using robust HuggingFace dataset loading candidates so the run actually attempts three HF datasets rather than … |
| 9 | improve | 0 | buggy | — | Execution failed during hf_mnist / geom_guard at epoch 6 with a PyTorch autograd RuntimeError caused by an in-place modification of the dist… |
| 10 | improve | 0 | good | best train accur=0.766 | Code-only draft (model emitted no separate plan). |
| 11 | improve | 0 | good | original accurac=0.571 | Code-only draft (model emitted no separate plan). |
| 12 | debug | 9 | good | final accuracy b=0.626 | The crash is caused by an in-place write into the `torch.cdist` output inside the geometric prototype loss, which invalidates PyTorch’s auto… |
| 13 | improve | 0 | buggy | original distrib=0.905 | The run completed and saved artifacts, but the experiment does not satisfy the stated sub-stage requirement to use THREE HuggingFace dataset… |
| 14 | improve | 0 | buggy | original distrib=0.905 | Execution completed and saved artifacts, but the experiment does not satisfy the stated sub-stage requirement to use THREE HuggingFace datas… |
| 15 | improve | 0 | buggy | original distrib=0.905 | The run completed and saved artifacts, but there are experimental/design bugs. The sub-stage required three HuggingFace datasets in total, y… |
| 16 | improve | 0 | good | — | Aggregate results from multiple seeds |

### Stage 4: ablation 消融实验

**23 节点**(good 18 / buggy 5)。搜索树:

```
[0] draft/good m=original distrib=0.905  (根)
├─ [17] improve/good m=final original v=0.902
├─ [18] improve/good m=final original a=0.91
├─ [5] improve/good m=train accuracy=0.873
├─ [3] improve/buggy m=original distrib=0.916
│  └─ [11] debug/buggy m=shifted training=0.907
│     ├─ [14] debug/good m=best original va=0.925
│     │  ├─ [20] improve/good m=best original va=0.925
│     │  ├─ [21] improve/good m=best original va=0.925
│     │  ├─ [22] improve/good m=—
│     │  └─ [19] improve/good m=best original va=0.924
│     └─ [13] debug/good m=validation accur=0.862
├─ [12] improve/good m=original distrib=0.902
├─ [6] improve/good m=training accurac=0.922
├─ [7] improve/buggy m=AdamW original d=0.901
│  └─ [9] debug/good m=shifted final OD=0.00435
├─ [4] improve/buggy m=base_all_trainab=0.903
│  └─ [8] debug/good m=shifted final va=0.825
├─ [15] improve/good m=original distrib=0.906
├─ [16] improve/buggy m=std floor baseli=0.906
├─ [1] improve/good m=train accuracy=0.9
├─ [2] improve/good m=training accurac=0.89
└─ [10] improve/good m=original validat=0.894
```

| idx | 动作 | 父 | 状态 | metric | agent 计划/分析摘要 |
|---|---|---|---|---|---|
| 0 | draft | — | good | original distrib=0.905 | The previous run likely failed to learn the rotated distributions because normalization divided rotated edge pixels by near-zero standard de… |
| 1 | improve | 0 | good | train accuracy=0.9 | Ablation name: No Experience Replay. The implementation below runs the original replay baseline and the requested “No Experience Replay” abl… |
| 2 | improve | 0 | good | training accurac=0.89 | Ablation name: Shallow MLP (Remove Second Hidden Layer). The implementation below runs the original two-hidden-layer MLP baseline and the sh… |
| 3 | improve | 0 | buggy | original distrib=0.916 | Execution completed and saved artifacts, but the log is truncated and several methodological/metric bugs are visible. Most importantly, ODAD… |
| 4 | improve | 0 | buggy | base_all_trainab=0.903 | The run completed and saved artifacts, but there are methodological/implementation issues that likely invalidate parts of the ablation. (1) … |
| 5 | improve | 0 | good | train accuracy=0.873 | Ablation name: Linearized MLP — Remove ReLU Nonlinearities. The script below runs the original ReLU MLP and the linearized MLP ablation unde… |
| 6 | improve | 0 | good | training accurac=0.922 | Ablation name: Distribution Order / Curriculum Reversal. The implementation below runs a curriculum-order ablation by comparing the original… |
| 7 | improve | 0 | buggy | AdamW original d=0.901 | Execution completed and saved artifacts, but the experiment has methodological/implementation bugs that make the ablation unreliable. (1) OD… |
| 8 | debug | 4 | good | shifted final va=0.825 | The previous implementation computed “Distributional Oversmoothing Gap” as a simple best-accuracy drop on old distributions, which conflates… |
| 9 | debug | 7 | good | shifted final OD=0.00435 | The bug is that the previous “Distributional Oversmoothing Gap” was computed as ordinary old-task degradation, so it conflated oversmoothing… |
| 10 | improve | 0 | good | original validat=0.894 | Ablation name: Classifier Head Continuity Reset. This implementation runs a control condition with the persistent classifier head and an abl… |
| 11 | debug | 3 | buggy | shifted training=0.907 | The run completed and saved artifacts, but the implementation contains a metric/indexing bug in final control evaluation. `final_evals_c` is… |
| 12 | improve | 0 | good | original distrib=0.902 | Ablation name: Remove Gradient Clipping. This program runs the original continual-learning setup twice per dataset: once with gradient clipp… |
| 13 | debug | 11 | good | validation accur=0.862 | I will fix the data-generation bug by ensuring that stochastic distribution shifts, especially additive noise, use independent random seeds … |
| 14 | debug | 11 | good | best original va=0.925 | The fix makes the ablation stage genuinely systematic and removes a metric-alignment bug in the matched control: control evaluations are now… |
| 15 | improve | 0 | good | original distrib=0.906 | Ablation name: Rotation-Only Shift — Remove Additive Noise. The implementation below runs the rotation-only ablation by keeping the ±20° dis… |
| 16 | improve | 0 | buggy | std floor baseli=0.906 | Execution completed and saved artifacts, but the ablation output reveals a likely experimental/implementation bug rather than a clean normal… |
| 17 | improve | 0 | good | final original v=0.902 | Ablation name: Remove Weight Decay Regularization. I will run the original baseline configuration and a matched no-weight-decay ablation con… |
| 18 | improve | 0 | good | final original a=0.91 | Ablation name: Reflect-Padded Rotation Boundary Ablation. This script runs the original constant-zero rotation pipeline and a reflect-padded… |
| 19 | improve | 14 | good | best original va=0.924 | Seed node |
| 20 | improve | 14 | good | best original va=0.925 | Seed node |
| 21 | improve | 14 | good | best original va=0.925 | Seed node |
| 22 | improve | 14 | good | — | Aggregate results from multiple seeds |

## C. 汇总统计

- **总节点数 63**(good 42 / buggy 21,成功率 67%)
- 最终论文 = 全部阶段最佳节点的实验结果聚合而成;上表每个 buggy 节点都经 agent 自我 debug(部分修复成功转 good,部分被裁剪)
- 动作推断规则:无父=draft(阶段根/seed),父 buggy→debug,父 good→improve