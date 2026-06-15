# The AI Scientist v2 方法论分析:算力、决策机制、Token 与跨平台

> 一份 **复现分析 / experience report**(非提出新方法):基于在自建 router + 8×H20 服务器上的实测,系统拆解 The AI Scientist v2 的树搜索方法论——哪些 idea 不需重度 GPU、AI 每一步怎么判断与依据什么、产生了哪些树、用了多少 token、超时怎么规范、以及如何在 Mac/Windows/Linux 上带显卡跑。
>
> 文档结构参考 LLM-for-ontology-construction 系列论文(LLMs4Life/NeOn-GPT、LLMs4OL),并补上它们普遍缺失的 **资源/超时/survivorship 量化**(第 6 节)。
>
> 数据口径:所有"实测"均来自 2026-06 的满预算 run(deepseek-v3.2=Overall 4 / gpt-4o=Overall 3 / gemini-2.5-pro=Overall 3),配置 `num_workers=8, num_seeds=3, stage iters=20/12/12/18, exec.timeout=3600`。代码行号针对本仓库 `AI-Scientist-v2/`。

---

## 1. 定位与研究问题

这不是提新方法,而是把一个公开系统(Sakana AI《Towards end-to-end automation of AI research》, Nature 651, 2026)在真实环境里跑通后,回答四个工程问题:

1. **哪些初始 idea 可以不用重度 GPU?**(第 4 节)
2. **AI 在树搜索里怎么判断、依据是什么?**(第 5 节)
3. **产生了哪些树、用了多少 token、超时怎么规范?**(第 6 节)
4. **怎么在 Mac/Win/Linux 带显卡跑、怎么加速、重头再来怎么自动化?**(第 7–8 节)

**一句话总纲**(后文反复用到):**这套系统的真瓶颈是 LLM 往返(~85–95% wall-clock),不是 GPU(util 0–12%)也不是 CPU。** 所以"用什么 GPU""跨什么平台"对多数 idea 是 *可选项而非必需项*;真正的提速旋钮全在 LLM 侧。

---

## 2. 框架骨架(先借一个可引用的标准 spine)

| | v1 (template-based) | v2 (template-free,本文主角) |
|---|---|---|
| 流程 spine | generate_ideas → perform_experiments(Aider 改模板)→ writeup → review | ideation → **4 阶段 best-first 并行树搜索** → 聚合画图 → writeup → review |
| 入口 | `launch_scientist.py` | `launch_scientist_bfts.py` |
| 实验来源 | 人写代码模板,Aider 迭代 | 零起点,agent 现写 `experiment.py` |

v2 的 4 阶段(`agent_manager.py:143-148` main_stage_dict):

```
Stage 1 initial_implementation  → Stage 2 baseline_tuning
        → Stage 3 creative_research → Stage 4 ablation
```

每阶段是一棵 best-first 搜索树;阶段结束选最佳节点 seed 给下一阶段。下面所有分析都挂在这个 spine 上。

### 2.1 树搜索到底在搜什么:节点 = 可执行实验(两层执行模型)

**最关键的一张图。** 容易误解为"树搜索 = LLM 在脑子里搜策略"(纯符号、不跑代码 → 不该有超时)。实际不是:**这棵树的每个节点是一份真在跑的实验代码,给节点打分 = 真把它执行一遍读真实 metric。** 系统因此分成上下两层,超时只活在下层:

```
━━━ 搜索/编排层  (agent_manager.py · parallel_agent.py) ━━━━━━━━━━━━━
  = 你以为的"策略搜索":LLM 驱动,快(API 往返,秒~分钟)

    best-first 选一个节点
         │
    LLM 出"策略动作"(= 树的边):   draft │ debug │ improve
         │   生成 / 改写这个节点的实验代码
         ▼   把代码交给执行层"求值"
  ┄┄┄┄┄┄┄│┄┄┄  ← exec.timeout=3600(1h)的边界:掐表就发生在这里  ┄┄┄
         ▼
━━━ 执行层  (interpreter.py 子进程) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  = 真跑实验:慢(可能 1h)、不可信(LLM 写的任意代码)

    python runfile.py  →  训练真模型 / 加载真数据集
         ├─ 跑完    →  写 experiment_data.npy  →  抽【真实】metric
         ├─ 抛异常  →  判 buggy
         └─ 超 1h   →  SIGINT,+60s 强杀  →  判 buggy(TimeoutError)
         │
         ▲  把【真实 metric / buggy】回传 → 搜索层据此决定下一个节点
```

**为什么会有超时(结构性原因)**:既然"访问/求值一个节点" = 执行一段 LLM 现写的任意 ML 代码,它就可能死循环、卡在数据下载、或无限训练。一个必须 *执行* 每个节点的搜索,遇到不返回的节点就会被永久卡住——所以执行层必须给每次求值一个墙钟预算。**超时是"搜索引擎(快、可信)"与"被搜索的实验(慢、不可信)"之间的边界**,不是给 LLM 想策略设的。

**一个比喻让它 click**(同是树搜索,差别全在"评估一个节点有多贵"):

| | AlphaGo MCTS | AI-Scientist 树搜索 |
|---|---|---|
| 节点是什么 | 一个棋局状态 | 一份可执行实验代码(`.py`) |
| 怎么评估节点 | 神经网络一次前向(~毫秒) | **真跑一次训练(秒 ~ 1h)** |
| rollout 成本 | 廉价 → **无需超时** | 昂贵且会挂 → **必须超时** |

所以:`draft/debug/improve`(树的边)是 LLM 的策略动作、受 §6.3 的 ② LLM-API 超时管;**而节点求值(树的"评估")是真实程序执行、受 ① `exec.timeout` 管**。两个超时管的是两层不同的东西。

---

## 3. 算力画像与瓶颈(实测)

满预算 deepseek run(wall-clock 274 min),三档拆解:

| 环节 | 实测 | 占 wall-clock |
|---|---|---|
| **GPU 算** | util 0–12%、每进程 GPU 显存 474–870 MiB、每卡 96GB 只用 1.4–2.0GB、功耗 ~115/500W | **~0%** |
| **代码实际执行**(CPU+IO) | 60 节点 `exec_time` 累加 79.6 min,中位 20s/节点(8 worker 并行 → 折 wall-clock ~10–30 min);多半是 CPU + 数据 IO | ~5–15% |
| **等 LLM 往返 + 编排** | ~280–680 次 LLM 调用,每次重发完整上下文 | **~85–95%** |

**为什么 GPU 几乎零负载**:这个 idea 的实验是玩具级 `SimpleNN`(3 层 `nn.Linear`、hidden=64、batch=32、合成数据),~6k 参数;实测每进程占的 474–870 MiB 绝大部分是 CUDA context + cuDNN/cuBLAS kernel 的固定"入场费",不是模型真用了那么多。**8×H20(768GB)在此 idea 上等于用机柜跑一个能在笔记本 CPU 上完成的脚本。**

**并发模型**(`parallel_agent.py`):1 worker = 1 进程 = 1 实验节点,`acquire_gpu` 独占一张卡(`min(available_gpus)`,1121-1132),不共享;`num_workers=8` 被 `min(num_workers, num_gpus)` 夹到卡数(1222);`num_seeds=3` 在主搜索 *结束后* 对 best_node 多卡并行重跑(1317-1387);`num_drafts=3` 只控 stage1 初始草稿数。峰值并发 8(主搜索)或 3(seed 阶段),不叠加。

---

## 4. Idea 的 GPU 分级:哪些可以不用重度 GPU

### 4.1 核心洞察

> **同一个 idea JSON 的真实 GPU 需求不由措辞决定,而由 agent 现场写的代码决定**——措辞只给"倾向性",且框架有**系统性偏置把它往轻量拉**:
> 1. `exec.timeout=3600`(1h/节点)硬墙 + 这条限制被写进 LLM 提示(`parallel_agent.py:352` "complete within {1 hour}"),任何撞墙的大规模方案会被 SIGINT 杀掉、判节点失败、被树搜索淘汰 → agent 被激励选小模型小数据;
> 2. 无 nvidia-smi 自动降级 CPU-only(accel_kind 判定 1204-1217)→ 框架本就能在无重度 GPU 下产出结果;
> 3. **实测铁证**:`compositional_regularization` 措辞写"IWSLT 翻译 + GeoQuery 解析"(本应 T2),但 agent 实际只跑了合成 SCAN、完全跳过两个真实集,writeup 还 hallucinate 了没做的实验。
>
> **结论:负面结果/失败模式类(ICBINB 主题)的 idea 普遍 GPU-light**——"证明某方法失效"用小规模就能演示,反而不需要堆算力。这正是该 workshop 主题与框架算力舒适区高度契合的根本原因。

### 4.2 GPU 需求档位(T0–T3)

| 档 | 典型 idea | 模型/数据规模 | 单节点显存 | 最低 GPU | 纯 CPU? |
|---|---|---|---|---|---|
| **T0 玩具/合成** | seir、2d_diffusion、grokking;本 benchmark 的 SimpleNN | 解析解/几千–几万参数,合成数据 | <1GB | 不需要 GPU | 可,甚至更快(省 H2D 拷贝) |
| **T1 小模型+小真实集** | mobilenetV3@CIFAR、sketch_rnn、nanoGPT_lite、earthquake | <10M 参数,MNIST/CIFAR/QuickDraw 量级 | 1–4GB | RTX 3060(12G)/T4 | 勉强,慢 10–50× |
| **T2 中等 benchmark** | tensorf(NeRF)、MACE、ResNet50/BERT-base、真做的 compositional(seq2seq@IWSLT) | 10–100M 参数,GB 级真实数据 | 8–24GB | RTX 4090(24G) | 不可行(撞 1h timeout) |
| **T3 微调 LLM/大视觉** | probes(gpt-j-6b)、pest_detection(YOLOv8+EfficientNet+ensemble) | 1B–数十B / 完整大数据集 | 40–80GB+ | A100/H100/H20 | 不可行 |

### 4.3 明确可不用重度 GPU 的清单(T0/T1,消费卡甚至纯 CPU)

| idea / 模板 | 档 | GPU-light? | 依据 |
|---|---|---|---|
| **compositional_regularization_nn** | T0–T1 | 是 | SCAN/COGS 合成小语料 + 小 seq2seq,<1–2GB;实测只跑合成 SCAN。三者中唯一被人评接收(6.33)——轻量恰契合舒适区 |
| **interpretability_failure_modes** | 可压到 T1 | 是(条件性) | 命题"解释方法何时失效"不依赖大模型;small CNN(MNIST/CIFAR)+Grad-CAM 的 sanity check 即可证伪 |
| seir (v1) | T0 | 是 纯 CPU | `scipy.integrate.odeint` 解 ODE,完全不 import torch |
| 2d_diffusion (v1) | T0 | 是 | 256-dim 3 层 MLP DDPM,2D toy 数据,秒–分钟级 |
| grokking (v1) | T0 | 是 | tiny Transformer 在 mod-p 算术合成数据,单卡/CPU 分钟级 |
| mobilenetV3 (v1) | T1 | 是 4–8GB | CIFAR-10 + MobileNetV3-Small,batch128 |
| sketch_rnn (v1) | T1 | 是 | LSTM VAE on QuickDraw 单类 |
| nanoGPT_lite (v1) | T1 | 是 | 刻意缩小的 char-level nanoGPT(全量 nanoGPT 才升 T2) |
| earthquake-prediction (v1) | T1 | 是 | ConvLSTM,batch_size=1,网格小 |

### 4.4 认真做会被迫上重度 GPU 的(避雷)

- **real_world_pest_detection**(T2–T3,最重):YOLOv8 检测训练 + EfficientNetB3 + ensemble + 域适应钉死,需真实带框标注数据集(本地无、HF 难拉)。**实测三篇里质量最差(4.67 被拒)**——框架做不动就退化,是最易因 1h timeout + 数据缺失全程崩的 idea。
- **probes**(v1, T3):`EleutherAI/gpt-j-6b` 经 transformer_lens 加载,fp16 也要 ~12–13GB。
- tensorf(NeRF)、MACE(等变 GNN+MD,`device='cuda'` 硬编码):T2。
- 判档口诀:**看措辞里有无钉死的大模型/大数据集标志词**——LLM 微调、6B+ 预训练权重、ImageNet 全量、YOLOv8/检测器训练、NeRF 多视角。有则被迫 T2/T3;无、或命题不依赖规模,则天然倾向 T0–T1。

---

## 5. AI 怎么判断、依据是什么(决策机制)

这是树搜索的核心。**三原语(draft/debug/improve)的选择是硬规则、不由 LLM 判断;LLM 只决定每个原语写什么代码、以及"选哪个节点更好"。**

### 5.1 节点选择(best-first,`_select_parallel_nodes` 1993-2114)

每个 `step()` 填满 8 个 worker 槽位,每槽独立做一次 best-first 决策(一步可并行扩展 8 个混合类型节点),优先级:

1. **drafting 优先**(2014):`draft_nodes < num_drafts(3)` → 槽位填 `None`(新 draft)。每 stage 先凑够 3 个根。
2. **debug(概率门控)**(2026):`random.random() < debug_prob(0.5)`。从 `buggy_nodes` 里筛 `is_leaf and debug_depth <= max_debug_depth(3)`,`random.choice` 随机选(buggy 无有效 metric,不排序)。
3. **stage4/stage2 特判**(2072):stage4 固定重复 `best_stage3_node`、stage2 固定重复 `best_stage1_node`(让所有 ablation/hyperparam 基于同一 baseline)。
4. **improve(正常 best-first)**(2087):取 `good_nodes`,用 `get_best_node` 拿当前最佳扩展;若该树已被本轮选过,按 `metric` 降序找下一棵未处理树。`processed_trees` 集合保证一步内尽量覆盖不同 draft 树(探索多样性)。

### 5.2 三原语何时选哪个(dispatch in `_process_node_wrapper` 1549-1583)

| 条件 | 原语 | 含义 |
|---|---|---|
| `parent is None` | **draft**(`_draft`) | 零起点写 baseline |
| `parent.is_buggy` | **debug**(`_debug`) | 读父节点报错修复 |
| else(父 good) | **improve**(`_improve`) | 在父基础上改进;stage2→hyperparam 节点、stage4→ablation 节点 |

与 `Node.stage_name`(journal.py:158-168)一致:`draft`(无父)/`debug`(父 buggy)/`improve`(父 good)是**硬规则**。

### 5.3 buggy 判定(任一为真即 buggy)

1. **exec 异常 + LLM review**(parse_exec_result, 693):`is_buggy = response["is_bug"] or exc_type is not None`。前者是 feedback-model 看 code+term_out 的**软判定**(没抛异常也能判 bug),后者是脚本抛异常的**硬信号**。
2. **metric 解析失败**:`valid_metrics_received==False` / 解析代码自身异常 / metric 含 None → 兜底 `WorstMetricValue`、`is_buggy=True`。
3. **VLM 看图**(_analyze_plots_with_vlm, 904):设**独立字段** `is_buggy_plots`(图空/无意义)。它排除出 `good_nodes`(不能当 best),但**不触发 debug**(debug 只看 `is_buggy`)。

角色分工:exec 异常=最硬门槛;LLM review=语义软判定;metric 缺失=数值健全性门槛;VLM=图有效性门槛。

### 5.4 指标抽取与"更好"的定义

**不是正则抽 stdout,是两段 LLM**:
1. code model 写一段解析代码,加载 `experiment_data.npy` 按 dataset 打印 metric(要求清晰命名 + 每 dataset 只打 best/final);
2. 跑解析代码 → feedback model 用 `metric_parse_spec` 把 term_out 转成结构化 `{metric_names:[{metric_name, lower_is_better, data:[{dataset_name, final_value, best_value}]}]}` 存进 `MetricValue`。

**"更好"**(`MetricValue.__gt__`, utils/metric.py:171-204):取跨所有 metric/dataset 的 `final_value` 均值;方向由**第一个 metric 的 `lower_is_better`** 决定;buggy 节点的 `WorstMetricValue` 恒垫底。

**选 best_node**(journal.py:420-502):默认 `only_good=True`;非纯指标模式下**用 LLM(gpt-4o, temp 0.3)综合选**——把每候选的 id/metric/analysis/VLM feedback 喂给它(提示明确"别只看 validation loss,多 dataset 要综合"),LLM 失败则兜底 `max(key=metric)`。

### 5.5 剪枝(没有显式删节点,剪枝=不再选某节点扩展)

- `max_debug_depth=3`:连续 debug >3 步的 buggy 叶子不再进 debug 候选 → 坏分支停。
- 只 debug `is_leaf`(已有子节点的 buggy 不重复 debug)。
- 坏树整体跳过:一棵 draft 树若所有叶子都 buggy,不算 viable tree,improve/best-first 不选它。
- improve 只从 `good_nodes` 扩展;buggy→`WorstMetricValue` 保证排序恒垫底,不会被误选 seed 下一阶段。
- `debug_prob=0.5` 概率门控:避免算力被 buggy 链吞掉,强制探索新 draft/improve。

### 5.6 阶段推进与跨阶段 seed

- **stage 结束**:`len(nodes) >= max_iterations`(20/12/12/18)兜底;stage1 要 `good_nodes>0`;stage2+ 要 best≠base 且 LLM 看 VLM 反馈判 `is_complete`。stage1 到上限仍无 working impl → 直接终止整个实验。
- **seed 到下一阶段**:stage 完成先跑 multi-seed eval(3 seed 多卡重跑)+ 聚合画图;进新 stage 时 `deepcopy` 上阶段 best、**清掉 parent/children**、作为新树的根。

### 5.7 每类决策 LLM 实际看到的"依据"(输入)

| 决策 | LLM 看到的输入 |
|---|---|
| draft | task_desc + memory_summary(历次成功/失败汇总)+ metrics 定义 + guideline;**不含父节点** |
| debug | task_desc + 父 `code` + 父 `term_out`(含报错)+ 父 VLM 反馈 + exec_time 反馈;**不直接喂 traceback**(靠 term_out 间接) |
| improve | task_desc + memory_summary + 父 VLM 反馈 + 父 `code`;**不喂 metric 数值**(metric 只用于选哪个父节点) |
| buggy review | task_desc + code + term_out → 输出 is_bug + analysis |
| metric 解析 | 仅解析脚本的 stdout |
| VLM 看图 | task_desc + plot 图片(base64,最多 10 张)→ plot_analyses + vlm_feedback_summary |
| 选 best | 每候选的 id + metric + analysis + VLM 反馈(gpt-4o 综合) |
| stage 完成 | best_node 的 VLM 反馈 + datasets_tested + stage goals |

**模型分工**(bfts_config.yaml:50-71):code model 写所有代码;feedback model 做 review/metric/stage 判定;vlm model 看图;select/summary 默认 gpt-4o。

---

## 6. 资源与约束量化(参照论文普遍薄弱、本文重点)

### 6.1 产生了哪些树(实测树结构)

| run | 总节点 | good/buggy | 成功率 | 各阶段(1/2/3/4) | 最终评分 |
|---|---|---|---|---|---|
| **deepseek 满预算** | 64 | 26/38 | 41% | 7/17/17/23 | Overall 4 |
| **gpt-4o 满预算** | 63 | 43/20 | 68% | 6/17/17/23 | Overall 3 |

**对比本身是数据点**:gpt-4o 树干净(stage2 几乎一路 improve),deepseek 踩坑多(满屏 NameError/typo/版本冲突,反复 debug),但**写出的论文反而更高分**。逐节点的树形 + agent 真实 debug 推理见配套**生成透明度附录**:`papers/deepseek_GENERATION_APPENDIX.md`、`papers/gpt4o_GENERATION_APPENDIX.md`(每篇含 ASCII 搜索树 + 每节点 draft/debug/improve 动作、状态、metric、agent 计划/分析摘要)。

### 6.2 Token 用量(实测,按"约束→应对"披露)

| run | 框架 TokenTracker(只抓 gpt-4o 辅助) | 真实交互日志 |
|---|---|---|
| deepseek 满预算 | gpt-4o: prompt 283K + completion 55K | **~3.96M**(主模型 deepseek-v3.2 收发)→ 全程 **≈ 4.3M token/篇** |
| gpt-4o 满预算 | gpt-4o: prompt 268K + completion 60K | 0.39M(欠采,见下) |

- 折算:deepseek ~280 次主模型调用 × 平均 14K/次——**每次调用都重发完整上下文**,这是"瓶颈在 LLM"的根因实锤。
- **约束→应对 1**:`MAX_NUM_TOKENS` 4096→16384(整篇 LaTeX + reasoning 模型留空间,否则 writeup 静默 return False);reasoning 模型 max_tokens 12000→32000。
- **约束→应对 2**:gpt-4o 走原生 usage 路径,没全写进交互日志 → 其 0.39M 是**采样缺口**,按调用数估实际也是数百万量级。
- **框架 TokenTracker 对 router 模型基本失效**(只抓到 gpt-4o,主力 deepseek/gemini 记 0,因 router 不返回 `completion_tokens_details`)——这本身是个坑。deepseek 的 ~4M 是交互日志实测的可靠下界。

### 6.3 超时怎么规范(三层,别混)

实际有**三层不同的"超时"**,管的是三样东西(详见 §2.1 的两层执行模型——① 在执行层,② 在搜索层的 LLM 调用):

| 层 | 管什么 | 值 | 执行点 |
|---|---|---|---|
| **① 实验代码执行**(核心,文档里反复说的"1h") | 一个树节点的 `.py` 实验能跑多久(= 节点求值) | `exec.timeout=3600`(1h) | interpreter.py:278-292:`running_time>timeout` 发 SIGINT,`+60s` 强杀,异常名改 `TimeoutError` 判 buggy;收集端 parallel_agent.py:2226 `except TimeoutError: continue` 跳过该节点,**不杀整个 run** |
| **② LLM 单次 API 调用** | 问 gpt-5.5/deepseek 一次、它思考+生成花多久 | openai SDK 默认 ~600s(10min)/次,**未显式设**;`backoff.expo, max_value=60`,**无 max_tries → 一直重试到成功** | backend_openai.py:52 `backoff_create` + utils.py:18 |
| **③ 整个 run 墙钟**(运维层,非框架) | 整篇论文从头到尾最多跑多久 | 我们编排脚本套的 `timeout 172800`(48h) | shell `timeout` 包住 launch |

- **三者无关**:gpt-5.5 单次调用只几秒~2-3min(受 ②),**绝不会单次跑满 1h**;那 1h 是 ① 给实验代码的。整篇 ~57min/4.6h 是几百次 ② 累加,不是某步卡 1h。早先 deepseek "5h 超时没出论文"是 ③(run 墙钟初期设置过短),后调整为 48h。
- **制度性副作用**:① 的 1h 还被写进 LLM 提示(parallel_agent.py:352 "it should complete within {1 hour}")主动引导"做轻量实验"——这是实验普遍小、GPU 闲置的根因之一。

**设计局限(批判性讨论)**:① 的杀进程是**纯看墙钟、完全不看还在不在产出**——interpreter.py:279 只判 `running_time > timeout`,不看 stdout 还在不在刷、loss 还在不在降;280 行原作者自留 `# [TODO] handle this in a better way`,连他们都知道这刀切得糙。它把"一个有前途、只是需要多几个 epoch 的实验"也一刀切掉判 buggy(假阴性),是**拿正确性换搜索吞吐**。理由是固定预算并行 best-first 需要节点间公平分时、且没有便宜 oracle 区分"慢但在进步"和"慢且在烧";代价是 T2/T3 真训练类 idea(如 pest_detection 要训 YOLOv8)塞不进 1h 被迫降规模/退化(实测质量最差被拒)。更好的设计会做框架现在**没做**的:checkpoint+续跑 / loss-plateau 检测给延期 / 用 FLOPs 预算替墙钟。这是 **Sakana/AIDE 上游的核心设计,非本复现新增**;论文 Methods 也报告了每节点运行时长 cap。

### 6.4 Survivorship / 鲁棒性(参照论文集体缺失的维度)

- 满预算全流程成功率:framework crash fix 前 2/5,修复后 4/5(`parallel_agent.py` 某 worker 异常被 catch 后又 `raise` → 层层上抛杀掉整个 run;去掉 `raise` 改为记录+跳过坏节点)。
- gemini 全系做不了树搜索(router 上 forced tool_choice 失败,只能 L0 写作);minimax 限流;glm router-400。

---

## 7. 跨平台与加速(Mac / Windows / Linux + 真正的杠杆)

### 7.1 当前设备处理(代码核实)

- 官方栈的检测层只认 NVIDIA:`get_gpu_count()`(1142)跑 nvidia-smi,失败 fallback CVD,再不行 return 0;`launch_scientist_bfts.py:134-137` 用 `torch.cuda.device_count()`。这两处不探测 MPS。
- 真正决定 device 的提示词模板在 `parallel_agent.py:304-309`,把 `if torch.cuda.is_available()` / `elif ...mps.is_available()` / `else cpu` 的三路分支作为**提示词文本**喂给 code LLM。框架自己不建 tensor。
- **本仓 patch 已为 Mac (Apple Silicon) 新增 MPS 三处探测**(行号针对本仓库 `AI-Scientist-v2/`,已核对):① device 提示词模板加 `mps` 分支(`parallel_agent.py:306-307`,并在 312 说明 MPS float32-only / CPU fallback);② 加速器判定 `accel_kind`(`parallel_agent.py:1212-1214` 经 `mps_available()` 1164,无 NVIDIA 时取 `mps`)+ worker 数封顶到 2(`parallel_agent.py:1228` 共享单设备防 OOM);③ worker 进程对 MPS 跳过 CVD、设 `PYTORCH_ENABLE_MPS_FALLBACK=1`(`parallel_agent.py:1502-1507`)。**因此 Mac 可降级跑**(单一共享 MPS 设备,慢),但 MPS 对 T0/T1 idea 相对 CPU 仅快约 1.2-3×,总时长仍被 LLM 往返主导。

### 7.2 三平台启用 GPU

| 平台 | 现状 | 改动 |
|---|---|---|
| **Linux + N卡** | 原生主路径(H20 服务器) | 无需改;多 run 分区务必设好各自 CVD(GPUManager 已修 nvidia-smi 无视 CVD 的坑) |
| **Windows + N卡** | 检测/CVD 逻辑可移植 | 主要坑:spawn-pickle(无 fork)、SIGINT 超时杀进程语义不同、LLM 生成代码硬编码 `/` 路径。**强烈推荐 WSL2 + CUDA 直通**,当 Linux 跑 |
| **Mac (Apple Silicon)** | 本仓 patch 已支持 MPS(官方栈仅 CPU) | MPS 三处探测已 ship:device 模板(304-309)三路探测 + 加速器判定(1212-1214,`mps_available()` 1164)+ worker(1502-1507)对 MPS 跳过 CVD、设 `PYTORCH_ENABLE_MPS_FALLBACK=1`。详见 `tools/` 补丁 |

**Mac MPS 关键 caveat**:MPS 是**单一共享设备**,不能像多卡那样把 8 worker 钉到 8 张"卡"。推荐做法:`num_workers` 调到 2–3、所有 worker 共享 MPS、调大 timeout。MPS 不支持 float64、部分算子(FFT/sparse/边角 scatter)无实现需 CPU fallback。**对 T0/T1 idea,开 MPS 相对 CPU 只快 1.2–3×,且总时长仍被 LLM 主导,体感几乎无差。**

### 7.3 真正的加速杠杆(瓶颈在 LLM,按收益排序)

| 杠杆 | 预期收益 | 做法 | 代价 |
|---|---|---|---|
| 1. **换快模型** | **3–8×** | bfts_config.yaml:51/57 的 code/feedback gpt-5.5→gpt-4o(reasoning→非 reasoning,单次往返从数十秒降到数秒) | 代码质量/创造性下降,buggy 率可能升(有 debug 兜底) |
| 2. **抬 router 并发** | 1.5–N× | 解除 `num_workers=min(,num_gpus)` 封顶(T0/T1 GPU 闲,无需按卡限 worker),让 16–32 CPU-bound worker 打满 router 配额 | 撞 429,需退避 |
| 3. **减冗余调用** | 1.2–2× | 缓存 `_define_global_metrics`、feedback 用小模型、`debug_prob` 0.5→0.3 | — |
| 4. **prompt 缓存** | 1.1–1.5×(省钱更明显) | 相同 system prompt 每次重发 → 开 prompt caching/memoize | — |
| 5. 调小 num_seeds/iters | 线性 | 3→1 砍掉尾部 3× 重跑 | 失去统计性,质量降,仅调试用 |
| 6. **本地小模型跑 code/feedback** | 潜在 5–10×(工程量大) | 本卡起 vLLM/ollama OpenAI 兼容端点接管,消除网络往返+限流 | reasoning 弱推高 buggy 率;建议混合(feedback 本地,code 仍 gpt-4o) |

**组合建议**:gpt-4o 替 gpt-5.5 → 解除 worker 封顶+抬并发 → debug_prob 降 0.3 + metric 缓存。**这套不动 GPU 平台就拿到大部分加速。**

### 7.4 诚实结论

对 **T0/T1 idea**,换 GPU 平台(Mac MPS / 多卡)几乎不提速——GPU 本就在闲。跨平台 GPU 的真实价值不是"更快",而是:**(1) 让 T2/T3 idea 能在 Mac/消费卡上算完(突破 CPU 撞 1h timeout);(2) 解锁"本地小模型接管 code/feedback"——那才是 GPU 真正能帮到这套系统的地方(加速的是推理 LLM,不是实验训练)。** 两件事正交,别指望 MPS 补丁让现有 T0/T1 跑得更快。

---

## 8. 重头再来:复现自动化

每次全新部署都需重打适配 patch(清单见附录 C / `patches/README.md`)。已封装为一键脚本(见 `tools/setup_ai_scientist.sh`):打 patch + 装依赖(psutil/scikit-learn)+ 写 router 配置。以后"重头再来 = 跑一条命令"。完整 patch 清单见附录 C。

---

## 附录

- **A. 生成透明度附录**(逐篇的搜索树 + 节点决策):`papers/*_GENERATION_APPENDIX.md`,生成器 `tools/gen_gen_appendix.py`。
- **B. Token 测量工具**:`tools/measure_tokens.py`(递归 tiktoken 求和交互日志,绕开失效的 TokenTracker)。
- **C. 本地 patch 清单**:见 `patches/README.md`。
- **关键代码引用**:决策机制 `treesearch/parallel_agent.py:1549-2114`、`journal.py:158-502`、`utils/metric.py:171-340`、`agent_manager.py:143-829`;设备 `parallel_agent.py:304-309/1142/1502-1507`;配置 `bfts_config.yaml`。
