# The AI Scientist 研究报告

> 面向研究者的源码级整合报告。对照论文《Towards end-to-end automation of AI research》(Nature, Vol 651, 2026, DOI 10.1038/s41586-026-10265-5) 与 arXiv:2504.08066《The AI Scientist-v2》。本报告整合三个仓库的深度源码分析：v1 模板系统 (`AI-Scientist/`)、v2 树搜索系统 (`AI-Scientist-v2/`)、ICLR2025 workshop 实验 (`AI-Scientist-ICLR2025-Workshop-Experiment/`)。所有路径均为绝对路径，关键引用保留 `文件:行号`。

---

## 0. 论文与仓库映射总览

| 论文章节 / 术语 | 仓库 | 关键文件 / 函数 | 行号 |
|---|---|---|---|
| Template-based AI Scientist (v1) | `AI-Scientist/` | `launch_scientist.py` | `:321-420` 主入口；`:154-318` `do_idea` |
| Template-free AI Scientist (v2) | `AI-Scientist-v2/` | `launch_scientist_bfts.py` | `:182-256` 编排 |
| Generalized idea generation | v2 | `ai_scientist/perform_ideation_temp_free.py` | `:128-266` |
| 文献查重 / novelty (Semantic Scholar) | v1+v2 | v1 `generate_ideas.py:405-492`；v2 `tools/semantic_scholar.py` | — |
| Experiment progress manager（4 阶段管理） | v2 | `ai_scientist/treesearch/agent_manager.py` | `:143-167` 阶段定义；`:692-829` `run` |
| Parallelized agentic tree search | v2 | `ai_scientist/treesearch/parallel_agent.py` | `ParallelAgent`/`MinimalAgent` |
| Experimental journal / node | v2 | `ai_scientist/treesearch/journal.py` | `:43-128` `Node`；`:361-613` `Journal` |
| 每节点=一次实验（≤1h） | v2 | `bfts_config.yaml:exec.timeout:3600` + `interpreter.py` | `:213-313` |
| Generalized dataset access (HuggingFace) | v2 | `agent_manager.py` 阶段 prompt + `bfts_utils.py` data/ 目录 | `:155-163`；`:64-67` |
| VLM integration (GPT-4o 批图) | v2 | `ai_scientist/vlm.py` + `perform_vlm_review.py`；树搜索内 `parallel_agent.py:_analyze_plots_with_vlm` | `:1020-1023` |
| Enhanced manuscript writing | v2 | `perform_plotting.py` + `perform_writeup.py` + `perform_icbinb_writeup.py` | — |
| Automated Reviewer (v2) | v2 | `ai_scientist/perform_llm_review.py` | `:125-233` |
| Automated Reviewer (复现/打分实物) | workshop | `ai-reviewing/perform_review.py` | `:126-243` |
| 首篇通过同行评审的全 AI 论文 / Table 1 | workshop | `README.md` + 三篇论文目录 | `README.md:22-26` |

一个贯穿全报告的命名提示：v2 树搜索内部大量沿用开源 **AIDE** ML agent 的术语（journal / node / draft / debug / improve），论文未使用这些名字。例如 `perform_experiments_bfts_with_agentmanager.py:201` 的 UI 标题写 "AIDE is working on..."。

---

## 1. v2 端到端 pipeline 全景

v2 是**两阶段解耦**架构：先离线生成想法存为 JSON，再把某个想法喂给树搜索主流程。唯一接口是 `ideas/*.json` 文件——可以人工编辑 JSON 跳过生成阶段（仓库自带 3 个预生成想法）。

### 阶段 0：Ideation（想法生成）

- **入口**：`ai_scientist/perform_ideation_temp_free.py:269-319`。参数 `--workshop-file`（主题种子 `.md`）、`--model`（默认 `gpt-4o-2024-05-13`，论文用 o3）、`--max-num-generations`、`--num-reflections`（默认 5）。
- **核心**：`generate_temp_free_idea`（`:128-266`）是双层循环——外层 `max_num_generations` 个高层提案（`:148`），内层 `num_reflections` 轮反思精炼（`:158`）。
- **ReAct 协议**：系统提示（`:61-96`）强制模型输出 `ACTION:` / `ARGUMENTS:` 两段，正则解析（`:182-250`）后分三支：`SearchSemanticScholar`（调 `tool.use_tool` 文献查重）、`FinalizeIdea`（产出 7 字段想法 JSON）、非法动作。系统提示末尾强制"finalize 前至少检索一次文献"（`:96`），对应论文"结合文献综述避免重复"。
- **输出**：想法数组写入 `ideas/<name>.json`（`:260-266`，追加到 archive 再整体覆写，支持断点续跑）。每个想法 7 字段：`Name`（后用作实验目录名）、`Title`、`Short Hypothesis`、`Related Work`、`Abstract`（约 250 词会议格式）、`Experiments`、`Risk Factors and Limitations`。这正是论文说的"grant-proposal 式高层提案"。

### 阶段衔接：launch 编排（`launch_scientist_bfts.py:182-256`）

读 `ideas[idea_idx]` → 建实验目录 `experiments/<date>_<Name>_attempt_<id>/`（`:198`）→ `idea_to_markdown` 把 7 字段转 `idea.md`（可选拼 `.py` 脚手架代码块，`bfts_utils.py:7-42`）→ `edit_bfts_config_file` 把全局 `bfts_config.yaml` 拷成 per-idea config，注入 `desc_file=idea.json` / `workspace_dir` / `data_dir`（空目录，数据运行时从 HF 拉）/ `log_dir`（`:250`，`bfts_utils.py:45-76`）→ 调 `perform_experiments_bfts(idea_config_path)`（`:256`）。

注意一个桥接细节：`desc_file` 指向的是 `idea_path_json`（`launch:253`）而非 markdown，所以 `load_task_desc`（`config.py:193-194`）实际读 idea.json 原始文本作为任务目标。

### 阶段 1：树搜索实验（§2 详述）

`perform_experiments_bfts`（`perform_experiments_bfts_with_agentmanager.py`）：`load_cfg` → `prep_agent_workspace` → 建 `AgentManager` → rich Live UI → `manager.run(...)` → pickle manager 到 `logs/manager.pkl`（`:213`）→ `overall_summarize`（`log_summarization.py`）把 4 个 stage 的 good leaf nodes 聚成 4 份结构化 JSON：`baseline_summary.json` / `research_summary.json` / `ablation_summary.json`（+ draft，`:234`）。实验产物"真相源"在 `logs/<run>/experiment_results/experiment_<id>/`：`experiment_data.npy` + `*.png` + `experiment_code.py` + `plotting_code.py`。

### 阶段 2：画图聚合 / VLM（§3 + 见下）

回到 launch：拷贝 `experiment_results` 到 idea 目录（`:257`）→ `aggregate_plots(idea_dir)`（`perform_plotting.py:136-254`）。
- 输入：`load_idea_text` + `load_exp_summaries`（固定加载 baseline/research/ablation 三摘要）+ `filter_experiment_summaries(step_name="plot_aggregation")` 裁剪出 `overall_plan/plot_code/plot_analyses/vlm_feedback_summary/exp_results_npy_files`（`perform_icbinb_writeup.py:708-717`）。
- 机制：让 reasoning model（默认 `o3-mini`，`launch:88`）生成**一个独立聚合脚本** `auto_plot_aggregator.py`，强制从 `.npy` 读真实数据禁止 hallucinate（`perform_plotting.py:29,64`），`fig, ax = plt.subplots(1, 3)` 合成 compound figure（`:42,77`），上限 `MAX_FIGURES=12`，只有 `figures/` 目录的图进最终论文。
- reflection 循环（默认 5 轮，`:191-254`）：统计实际图数 + 回灌 stdout/stderr，"I am done" 早退。

VLM 在两处介入：(a) 树搜索阶段每个节点的图由 `_analyze_plots_with_vlm` 批判并设 `is_buggy_plots`（§2）；(b) 成稿阶段对 `figures/` 每张 PNG 调 `generate_vlm_img_review` 拿图描述喂写作（见下）。

### 阶段 3：写作（§见下 + 模型分工 §6）

`gather_citations`（Semantic Scholar，带断点续跑）→ `perform_writeup`（ICML 版）或 `perform_icbinb_writeup`（ICBINB workshop 版）。big model（默认 `o1-preview`，`launch:94`）一次性生成整篇 LaTeX；多轮 reflection 集成 chktex linter + 图引用一致性 + 页数检测 + VLM 图-caption 对齐。详见 §4 写作部分与下文。

### 阶段 4：评审

`perform_review`（默认 `gpt-4o`）+ `perform_imgs_cap_ref_review`（VLM 图-caption-引用对齐）→ `review_*.json`（`launch:311-318`）→ `save_token_tracker` + psutil 杀残留进程 + `sys.exit`。

---

### v2 端到端数据流图

```
[主题 .md] ──perform_ideation_temp_free.py──> [ideas/*.json 7字段想法]
                  └ ReAct: SearchSemanticScholar(查重) / FinalizeIdea
                                  │ --load_ideas --idea_idx
                                  ▼
launch_scientist_bfts.py: idea_to_markdown → edit_bfts_config_file(desc_file=idea.json)
                                  ▼
perform_experiments_bfts → AgentManager.run (4主阶段×动态子阶段)
   每子阶段 new ParallelAgent + new Journal:
     while not stage_complete:
        _select_parallel_nodes(best-first 选 N 点) → ProcessPool →
          worker: draft/debug/improve/hyperparam/ablation
            → Interpreter.run(code) [≤3600s] → parse(is_buggy)
            → LLM metric-parse → MetricValue → 绘图 → VLM(is_buggy_plots)
            → to_dict() → journal.append
        _check_stage/substage_completion
     stage完成 → multi-seed(3) → plot_aggregation(seed_agg_node)
                                  ▼
overall_summarize → baseline/research/ablation_summary.json
                                  ▼
aggregate_plots(o3-mini) → auto_plot_aggregator.py → figures/*.png (compound)
                                  ▼
gather_citations(S2) → perform_(icbinb_)writeup(o1) + chktex + VLM图核对 → PDF
                                  ▼
perform_review(gpt-4o) + perform_imgs_cap_ref_review(VLM) → review_*.json
```

---

## 2. Agentic 树搜索机制详解（核心）

整个树搜索核心位于 `ai_scientist/treesearch/`。映射：论文 **Experiment progress manager** = `agent_manager.py:AgentManager`（外层主/子阶段循环）；论文 **Parallelized agentic tree search** = `parallel_agent.py:ParallelAgent`（每 stage 一次 best-first 树搜索 + 进程池并行）；论文 **experimental journal** = `journal.py`。

### 2.1 数据结构：Node / Journal

**`Node`（`journal.py:43-128`）= 一次实验**。关键字段：
- 代码/计划：`plan` / `overall_plan` / `code` / `plot_code` / `plot_plan`（`:48-52`）。
- 树结构：`parent` / `children:set` / `step` / `id`（uuid hex，`:55-59`）；`__post_init__` 自动把自己加进 `parent.children`（`:120-126`）。
- 执行结果：`_term_out` / `exec_time` / `exc_type/exc_info/exc_stack`（`:62-67`）。
- 评价：`analysis` / `metric:MetricValue` / `is_buggy` / `is_buggy_plots`（`:87-92`）。
- 绘图与 VLM：`plots`/`plot_paths` / `plot_analyses` / `vlm_feedback_summary` / `datasets_successfully_tested`（`:97-105`）。

**`Journal`（`journal.py:361-613`）= 解树**：`nodes:list[Node]`，提供 `draft_nodes`（`parent is None` 即树的根集合）、`buggy_nodes`、`good_nodes` 视图。**一个 Journal 里可以有多棵树**（多个 draft 根），best-first 在多树间平衡探索。

### 2.2 节点类型（论文术语 ↔ 代码编码）

类型不是单一枚举字段，而是**由结构 + 标志位组合推断**：

| 论文术语 | 代码编码 | 位置 |
|---|---|---|
| draft（初稿/replication 起点） | `parent is None` → `Node.stage_name == "draft"` | `journal.py:158-168` |
| debug | `parent.is_buggy` → `"debug"` | 同上 |
| improve | 否则 → `"improve"` | 同上 |
| hyperparameter | `hyperparam_name` 非空 | `journal.py:114`；生成于 `parallel_agent.py:557-603` |
| ablation | `ablation_name` 非空 | `journal.py:111`；生成于 `parallel_agent.py:605-656` |
| replication（多种子复跑） | `is_seed_node` | `journal.py:117` |
| aggregation（聚合节点） | `is_seed_agg_node` | `journal.py:118` |

`debug_depth`（`journal.py:202-212`）递归数连续 debug 步数，用于限深。**注意**：论文只讲 hyperparameter/ablation/replication/aggregation 四类，但代码底层搜索原语其实是 AIDE 的 draft/debug/improve 三态；hyperparam/ablation 只是 improve 在特定 stage 的特化；replication = draft 根或 `is_seed_node`。

### 2.3 buggy / non-buggy 的双通道编码（代码比论文细）

有**两条独立 buggy 通道**：
1. `is_buggy`（`parallel_agent.py:713`）：`node.is_buggy = response["is_bug"] or node.exc_type is not None`，即"LLM reviewer 判 bug"或"执行抛异常"；metric 解析失败也强制 `is_buggy=True`（`:1642-1652`），此时 metric 设为 `WorstMetricValue()`。
2. `is_buggy_plots`（`_analyze_plots_with_vlm`，`:1020-1023`）：VLM 若判 `valid_plots_received=False`（空图/无意义/误导可视化）就标 True。**这正是论文 "VLM marks unclear labels/misleading viz as buggy"**。

`Journal.good_nodes`（`journal.py:405-407`）要求**两者都为 False**：`n.is_buggy is False and n.is_buggy_plots is False`。

### 2.4 best-first 的选择 / 扩展 / 评估

每个 stage 内的一次树搜索 = 反复调 `ParallelAgent.step()`。

**(a) 选择 `_select_parallel_nodes`（`parallel_agent.py:1931-2051`，主进程）** 一次选出 `num_workers` 个待扩展节点，循环直到选满：
1. **drafting**：`len(draft_nodes) < num_drafts`（默认 3，`bfts_config.yaml:69`）→ append `None`（让 worker 新建一个 draft 根）。
2. **debug（带概率）**：以 `search.debug_prob=0.5`（`:75`）概率，从 `buggy_nodes` 里挑 `is_leaf and debug_depth <= max_debug_depth(=3)` 的叶子随机选（`:1964-2004`）。
3. **Stage 特判**：stage_name 以 `"4_"` 开头 → append `best_stage3_node`（长 ablation）；以 `"2_"` 开头 → append `best_stage1_node`（长 hyperparam）（`:2009-2015`）。
4. **Stage 1/3 正常 best-first（exploitation）**：取 `journal.get_best_node()` 为父；用 `processed_trees` 集合保证不在同一棵树重复扩展（除非所有 viable 树都处理过），否则按 metric 降序找下一个 good node（`:2016-2049`）。

> 论文说"用 GPT-4o 做 best-first search 选最有希望节点"——**真正的"哪个最有希望"由 `get_best_node` 的 LLM 决定**，`_select_parallel_nodes` 只做并行/多树调度。

**(b) 扩展 `_process_node_wrapper`（`parallel_agent.py:1409-1796`，worker 进程）**：根据父节点状态选算子（`:1486-1522`）——`None`→`_draft()`；`parent.is_buggy`→`_debug()`；否则按 stage 选 hyperparam/ablation/`_improve()`。同一 worker 内完成完整 evaluate 流水线：`interpreter.run(code)` → `parse_exec_result`（is_buggy）→ 若有 `.npy` 则 LLM 生成并跑 metric 解析代码 → 若 non-buggy 则生成绘图代码（最多重试 3 次）→ 移图到 `experiment_results/` → `_analyze_plots_with_vlm`（is_buggy_plots）→ `node.to_dict()` 返回主进程。

**(c) 评估 `Journal.get_best_node`（`journal.py:420-502`）**：候选 = `good_nodes`；只 1 个直接返回；否则构造 prompt 列出每候选的 `metric/analysis/vlm_feedback_summary`，调 LLM（默认 `gpt-4o`，temp 0.3，`:471`）用 `node_selection_spec` 返回 `selected_id`；LLM 失败 fallback 到 `max(nodes, key=lambda n: n.metric)`。

> **关键细节（论文未细讲）**：best node 选择不是纯标量 argmax，而是带视觉/语义判断的 **LLM 软决策**，prompt 明确告诫"别只看 validation loss，要综合 plots 质量、多 dataset 表现"（`journal.py:442-448`）。`MetricValue.__gt__`（`metric.py:171-189`）按多 metric×多 dataset 的均值比较，`WorstMetricValue`（value=None）恒最差，保证 buggy 节点永远排最后。

> **metric 是二段式 LLM 提取**：先让 code-model 写一段解析脚本 print 指标（`parallel_agent.py:1554-1594`），跑完再让 feedback-model 用 `metric_parse_spec` 把 stdout 解析成结构化 `MetricValue`（`:1619-1646`）。

### 2.5 4 阶段管理器（Experiment progress manager）

**阶段定义（`agent_manager.py:143-167`）**：
```
{1: initial_implementation, 2: baseline_tuning, 3: creative_research, 4: ablation_studies}
```
论文术语 ↔ 代码：preliminary investigation→stage1；hyperparameter tuning→stage2；research agenda execution→stage3；ablation studies→stage4。

**`run()` 是双层循环（`:692-829`）**：外层遍历主阶段，内层遍历**动态生成的子阶段**（论文没展开）；每个子阶段 `with self._create_agent_for_stage(...) as agent`（`:703`）新建一个 `ParallelAgent` + 独立 `Journal`，即**每个子阶段各跑一次独立树搜索**。

**阶段间最优节点传递（`_create_agent_for_stage`，`:274-329`）**：stage2 用 stage1 best 作 `best_stage1_node`（hyperparam 起点）；stage3 继承 stage2 best 的 plotting code；stage4 用 stage3 best 作 `best_stage3_node`（ablation base）。每进入新子阶段把上一子阶段 best node deepcopy 当种子 append 进新 Journal（`:704-718`）。

**停止准则（关键）**：

`_check_stage_completion`（`:410-536`），每个 `step()` 后调（`:722-732`）：
- 通用上限：`len(journal.nodes) >= stage.max_iterations`（stage1=20, stage2=12, stage3=12, stage4=18，`bfts_config.yaml:38-41`）。
- **stage1**：`len(good_nodes) > 0` 即完成；到 max_iters 仍无 good node → `current_stage=None` **终止整个实验**（`:419-429`，对应论文"preliminary investigation 失败则放弃"）。
- **stage2**：best 必须 ≠ base（有改进，`:448-452`），再用 LLM（`stage_completion_eval_spec`）基于 VLM 图分析 + `datasets_successfully_tested` 判"收敛 / 至少两 dataset / 无明显不稳定"（`:455-498`）。
- **stage3**：best≠base；额外检查**执行时长**——若过半 max_iters 且 `best_node.exec_time < timeout/2`（跑太快 <30min），写 `exec_time_feedback` 要求 scale up（`:500-530`，论文未提的启发式）。
- **stage4**：无额外准则，跑满 max_iters。

`_check_substage_completion`（`:343-408`）：拿 best node 的 VLM feedback，LLM 对照子阶段 goals 判 `is_complete`；完成则 `_create_next_substage`（LLM 生成下一子阶段名+goals），主阶段完成则 `_create_next_main_stage`（stage4 后返回 None 全部结束）。

**每主阶段收尾（`:736-766`）**：对 best node 跑 `_run_multi_seed_evaluation`（默认 3 seed 并行复跑，`parallel_agent.py:1261-1330`）→ `_run_plot_aggregation`（LLM 生成带均值/标准误的聚合图，产 `is_seed_agg_node`，`:1332-1407`）。对应论文 replication+aggregation 节点 + 多种子统计显著性。

### 2.6 并行执行与超时

- **进程池 / GPU 分配**：`ProcessPoolExecutor(max_workers=num_workers)`（默认 8，`bfts_config.yaml:36`）。检测到 GPU（nvidia-smi，`parallel_agent.py:1120-1139`）则 `num_workers=min(num_workers, num_gpus)`，`GPUManager` 给每 worker 独占一卡（`CUDA_VISIBLE_DEVICES`，`:1443-1448`）；无 GPU 则 CPU-only。
- **step 并发流程（`:2053-2190`）**：选 N 点 → `to_dict()` 序列化（`_safe_pickle_test` 先验，`:2064`）→ 生成共享 `memory_summary`（`journal.generate_summary`，`journal.py:504-548`）→ **stage2/4 在主进程预生成 idea**（对 non-buggy 节点调 `_generate_hyperparam/ablation_idea`，立即加入 `tried_hyperparams`/`completed_ablations`，`:2096-2116`，**目的是避免并行 worker 跑重复 idea**）→ `executor.submit` 提交 → `future.result(timeout=self.timeout)` 收结果 → `Node.from_dict` 用 journal 重建 parent-child → `journal.append`。
- **两层超时（论文"每节点≤1h"落地）**：未来层 `future.result(timeout=3600)`（`:2152`）；解释器层是硬超时——`interpreter.py:run`（`:213-313`）子进程跑 code，每 1s poll，`running_time > timeout` 发 `SIGINT`（`:284`，KeyboardInterrupt 重写为 TimeoutError），再过 60s 不退 SIGTERM→SIGKILL（`:286-293`）。每段 code（实验/metric/绘图）各受此约束。
- **资源清理**：`ParallelAgent.__exit__→cleanup`（`:2333-2368`）释放 GPU + `executor.shutdown(cancel_futures=True)`；launch 最外层（`:321-359`）psutil 扫描 kill 含 python/torch/bfts/experiment 关键字的残留进程。

### 2.7 代码瑕疵（论文未提）

- `_check_substage_completion`（`agent_manager.py:371-395`）try 块里无论真假都已 `return`，其后 max-iter 检查（`:397-405`）与 `return False`（`:408`）是**死代码**。
- `_check_stage_completion` 实际返回 `(bool, str)` 元组，但 docstring 写 `-> bool`，签名不一致。
- `_create_stage_analysis_prompt` 里 `stage_number` 未定义（`:888`），该函数在 run() 主流程未被调用，属遗留代码。

---

## 3. Automated Reviewer 自动评审

### 3.1 v2 版（`AI-Scientist-v2/ai_scientist/perform_llm_review.py`）

核心入口 `perform_review`（`:125-233`）：
- **系统角色三档**：中性 `_base`、悲观 `_neg`（默认）、乐观 `_pos`（`:13-24`）。
- **评审表 `neurips_form`（`:64-122`）**= NeurIPS 官方指南全文。输出 JSON 字段（`:42-61`）：`Summary, Strengths, Weaknesses, Originality(1-4), Quality(1-4), Clarity(1-4), Significance(1-4), Questions, Limitations, Ethical Concerns(bool), Soundness(1-4), Presentation(1-4), Contribution(1-4), Overall(1-10), Confidence(1-5), Decision(Accept/Reject)`。论文点名的 soundness/presentation/contribution、优缺点、accept/reject 全在此。
- **few-shot**：`get_review_fewshot_examples`（`:312-340`）默认 1 个真实 paper+review（可选 3 个）。
- **5-review ensemble + meta-review（`:150-202`）**：`num_reviews_ensemble>1` 时用 `get_batch_responses_from_llm` 一次取 N 份（temp 0.75）→ `get_meta_review`（`:349-369`）用 area-chair 系统提示聚合 → **9 个数值分项用各 review 合法分的 np.mean 四舍五入覆盖**（`:171-188`，meta-review 文字来自 area chair，分数来自 ensemble 平均）。
- **reflection**：`num_reflections>1` 用 `reviewer_reflection_prompt`（`:236-254`）复核，"I am done" 早退。
- **PDF 读取**：`load_paper`（`:257-288`）三级 fallback `pymupdf4llm → pymupdf → pypdf`。

> **重要落差**：论文说 Reviewer 用 **o4-mini + 5-ensemble + meta-review**，但 `launch_scientist_bfts.py:311` 用全默认参数调 `perform_review`（即 `num_reviews_ensemble=1, num_reflections=1`），且 `--model_review` 默认 `gpt-4o-2024-11-20`（`:118`）。也就是说，**开箱默认是单条 review、无 ensemble、用 gpt-4o**；论文描述的 o4-mini + 5-ensemble 是代码暴露的**能力**而非默认。评审表用 NeurIPS 指南（非 ICLR），与人评对比在 ICLR OpenReview 数据上做（Table 1）。

### 3.2 workshop 复现版（`AI-Scientist-ICLR2025-Workshop-Experiment/ai-reviewing/perform_review.py`）

这是 **v1 的 Reviewer 实现**被直接拿来给 v2 生成的三篇论文打分（连 `perform_improvement` 用 Aider `coder` 都还留着，`:386-396`；`llm.py:116-118` 注明 reviewer pipeline 仅对 GPT-4 系列完整支持）。核心骨架与 v2 一致：`perform_review`@126、`neurips_form`@65、ensemble+meta@151-209、few-shot@310-352（池为 `132_automated_relational` / `attention`(Transformer 原论文) / `2_carpe_diem`）、reflection@246-264。

**两个 notebook 跑同一配置**（`gpt-4o-2024-05-13`, `num_reflections=5`, `num_fs_examples=1`, `num_reviews_ensemble=5`, `temperature=0.1`，即论文的 5-ensemble+meta-review）：
- `reviews_conference.ipynb`：默认 negative prompt，按**顶会 full-paper 高标准**评，产 `_conference_track.txt`（严苛基线）。
- `reviews_workshop.ipynb`：cell-1 换 **ICBINB workshop 专用 prompt**（贴出四要素征稿标准 + 五条评估维度），产 `_workshop_track.txt`（与真实人类分可比）。

### 3.3 与人类对比的核心结论

**关键观察 A — AI Reviewer 在 workshop 轨区分度不足**：三篇 workshop_track **全部** `Overall=7, Accept`（子分 3/3/3，意见模板化），无法把真正被接收的论文 1 与被拒的 2、3 区分开。conference 轨则把三篇全部 Reject（Overall 3/3/4）——说明 AI 自评对 rubric 敏感，但 workshop 阈值偏宽松。

**关键观察 B — 人类区分度明显更高**：ICLR 真人三评清楚拉开差距（6.33 vs 4.67 vs 3.0）；Sakana 内部人类也只把论文 1 评为 borderline accept。

**关键观察 C — AI 对优点判断与人类一致，对严重性判断不同**：AI 能正确识别"诚实报告负面结果/问题有现实意义"，但**系统性漏掉"声明与代码/图表不符"这类需查实现的硬伤**（论文 1 正则化实际作用在 embedding 而非论文所称 hidden state；论文 2 domain adaptation 代码里失败未实现；论文 3 temperature scaling 写了从未调用、Figure 3 引用了图中不存在的 ECE）——这些都靠 Sakana 团队 `sakana_review.pdf` 的 Code Review 查源码才发现。

> **结论审慎化**：Nature 摘要"Automated Reviewer 性能与人类评审相当"在这 3 个样本上需谨慎——AI 自评在 workshop 阈值偏宽松、区分能力明显弱于人类。v2 相对此 v1 实现的演进（o4-mini 推理模型 + VLM 视觉核查）正是针对"看不到图、靠多轮 prompt 模拟思考"这两个短板，但 ensemble+meta-review 骨架 v1/v2 一致。

---

## 4. v1 模板系统 与 v2 架构对比

### v1 三阶段串联（`AI-Scientist/launch_scientist.py`）

主入口 `:321-420` 把三阶段串成流水线。单个 idea 的生命周期 `do_idea`（`:154-318`）：
1. `copytree(base_dir, destination_dir)`（`:171`）把模板整目录克隆成独立沙盒。
2. 读 `run_0/final_info.json` 的 baseline `means`（`:172-176`），写 `notes.txt`。
3. **阶段2 实验**：创建 Aider Coder 挂载 `[experiment.py, plot.py, notes.txt]`（`:197-216`，`edit_format="diff"`, `use_git=False`）→ `perform_experiments`（`:221`）。
4. **阶段3 写作**：重建 Coder 挂载 `[experiment.py, latex/template.tex, notes.txt]` → `perform_writeup`（`:254`）。
5. **评审**：`perform_review(... gpt-4o-2024-05-13, num_reviews_ensemble=5, num_reflections=5 ...)`（`:267-276`，强制 OpenAI gpt-4o）。
6. 可选 `--improvement`：`perform_improvement` → `_improved.pdf` → 再评审。

idea 间用 `multiprocessing.Queue` 粗粒度并行（一 idea 一进程一 GPU，`:133,390`）——**注意这不是 v2 那种单实验内部的树搜索并行**。

**Aider 调用与 debug 循环（`perform_experiments.py`）**：`MAX_ITERS=4`（自动 debug 最多 4 轮，对应论文）、`MAX_RUNS=5`、单实验 `timeout=7200`（2h，对应论文"single experiment 7200s"）、plot `timeout=600`、latex 编译 30s/命令、chktex 修复 5 轮。主循环（`:116-166`）：`coder.run` 改 `experiment.py` → `run_experiment` 真跑 → 成功则反馈 means 进下一 run（debug 计数清零）；失败则把 stderr（截尾 1500 字符）喂回触发下一轮 debug，连续失败 4 次放弃整个 idea。

**模板结构（12 个模板，以 nanoGPT/grokking 为例）**：`experiment.py`（必须支持 `--out_dir=run_i`、产出 `final_info.json[*]["means"]` + `all_results.npy`）、`plot.py`、`prompt.json`（system+task_description）、`seed_ideas.json`（人写种子）、`latex/template.tex`（占位符 + 内嵌 `filecontents` bib + ICLR2024 样式）、`run_0/`。换模板只需提供这套文件、无需改 pipeline——这是 template-based 可扩展性的来源。

**idea archive 迭代（`generate_ideas.py:76-174`）**：seed_ideas 初始化 archive → 外层 `--num-ideas 50` 轮生成（每轮注入历史避免重复）→ 内层 `NUM_REFLECTIONS=3` 轮反思（"I am done" 早退）。**novelty check（`:405-492`）**：每 idea 最多 10 轮"搜索-判断"循环，LLM 用 query 调 `search_for_papers`（Semantic Scholar 或 OpenAlex），读 top-10 摘要判 `Decision made: novel/not novel`，只有 `novel==True` 进 `do_idea`。

### v1 vs v2 全面对比

| 维度 | v1 (Template-based) | v2 (Template-free) |
|---|---|---|
| **代码起点** | 依赖人写 `templates/<exp>/`，克隆后 Aider 原地 diff 改写 | 不依赖模板，从高层提案动态生成 |
| **代码生成** | **Aider**（`Coder.create`, `edit_format="diff"`, SEARCH/REPLACE 补丁） | **不用 Aider**，模型直接生成完整代码 |
| **探索结构** | **顺序流程**：单线 while，最多 5 run、每失败最多 debug 4 轮；idea 间粗粒度进程并行 | **Parallelized agentic tree search**：节点=实验(≤1h)，buggy/non-buggy，GPT-4o best-first 选点，进程池并行，journal 记录 |
| **阶段组织** | 单一线性阶段，无阶段管理器 | **Experiment progress manager** 协调 4 阶段，各阶段独立树搜索 + 停止准则 |
| **数据集** | 固定在模板里（idea prompt 明确"no additional datasets"，`generate_ideas.py:26`） | **Generalized dataset access**：动态查 HuggingFace Hub |
| **视觉** | 无 VLM，仅程序化检查图存在性/重复性 | **VLM integration**：GPT-4o 批图标 buggy + 图-caption 对齐 |
| **写作** | Aider 逐节填 LaTeX + chktex | reasoning model(o1) + reflection 直接生成 + 聚合 compound figures + VLM review |
| **超时粒度** | 单实验 7200s、plot 600s、latex 30s | 每树节点 ≤ 3600s |
| **评审** | gpt-4o，NeurIPS 表，5-ensemble+meta+5 reflection（开箱即用） | 同骨架，论文说换 o4-mini+VLM，但**代码默认 gpt-4o 单 review** |

**Sakana 自己的说法（README）**：workshop 仓库 README 开头明确："A paper produced by The AI Scientist passed a peer-review process at a workshop in a top machine learning conference. To our knowledge, this is the first fully AI-generated paper that has passed the same peer-review process that human scientists go through."（`README.md:12`）。

**一句话总结**：v1 = "人写模板 + Aider 顺序补丁 + 线性 debug 循环"，可靠性来自模板的 I/O 契约约束；v2 = "无模板 + 多模型分工 + 并行 agentic 树搜索 + VLM + 阶段管理器"，探索能力来自树搜索取代单线顺序执行。

---

## 5. Workshop 实验实物：3 篇生成论文

三篇全自动生成论文投到 ICLR 2025 ICBINB workshop（"I Can't Believe It's Not Better: Challenges in Applied Deep Learning"，主题=负面结果/意外障碍）。

**需先纠正一个常见误读**：README 表头 **"ICLR Workshop Scores"** 才是决定接收的 ICLR 三位匿名人类评审分；`sakana_review.pdf` 是 Sakana 内部团队人工复核（另一套人）；`ai_reviews/` 是 Automated Reviewer 自评。三者独立。

| 论文 | 主题 | ICLR 人类三评 (README) | 平均 | Sakana 内部人类 | AI Reviewer workshop | AI Reviewer conference |
|---|---|---|---|---|---|---|
| **1. Compositional Regularization** | LSTM 组合泛化的组合正则化失败（隐藏状态相邻时间步差异惩罚） | **6, 7, 6** | **6.33 接收** | Workshop 5/10 borderline accept | **7 → Accept** | 3 → Reject |
| **2. Pest Detection** | DL 病虫害检测真实场景泛化失败 (自创 ERS 指标) | 3, 7, 4 | 4.67 拒 | Workshop 3/10 Reject | 7 → Accept | 3 → Reject |
| **3. Label Noise** | 标签噪声对模型校准 (ECE) 的影响 | 3, 3, 3 | 3.0 拒 | Workshop 3/10 Reject | 7 → Accept | 4 → Reject |

**被接收的是论文 1（Compositional Regularization）**，6.33 超过接收阈值，成为论文主张的"首篇通过同行评审的全 AI 生成论文"。注意它即便被接收也只是**勉强过线**（Sakana 内部 5/10 borderline），且 `annotated_paper.pdf` 带 Sakana 事后红框标注的明确缺陷（正则化项实际作用在 embedding 而非论文所称 hidden state、引用错误、训练/测试集约 57% 重叠）。

**实物如何佐证主张**：(1) 接收事实可核验（README + `annotated_paper.pdf` 实物）；(2) 流程真实（ICLR leadership + workshop organizers 配合 + UBC IRB 批准，双盲评审，`README.md:14,16`）；(3) 透明（三篇都带 Sakana 红框标注 + Code Review，不回避缺陷）；(4) 对照组佐证边界——另两篇被拒证明"首篇通过"是 3 投 1 中的结果，而非系统稳定达标，反而让主张更可信。

引用文件：`compositional-regularization/ai_reviews/20250310-104944_review_0_workshop_track.txt`（7/Accept）vs `..._conference_track.txt`（3/Reject）。

---

## 6. 模型分工与成本

| 论文角色 | 论文模型 | 代码实际落点（本仓库快照） |
|---|---|---|
| **Ideation** | OpenAI **o3** | `perform_ideation_temp_free.py --model` 默认 `gpt-4o-2024-05-13`（o3-mini 在 `AVAILABLE_LLMS` 清单内） |
| **代码生成** | Claude **Sonnet 4** | `bfts_config.yaml:57` 实际 `anthropic.claude-3-5-sonnet-20241022-v2:0`（Sonnet 3.5 v2，走 Bedrock）；`max_tokens:12000`（`:59`） |
| **VLM 视觉** | **GPT-4o** | `bfts_config.yaml` `vlm_feedback.model: gpt-4o-2024-11-20`（`:69`）；与论文一致 |
| **best-node 选择 / metric 解析 / is_bug / stage 完成判定** | GPT-4o | `journal.py:471` 默认 gpt-4o；`feedback.model: gpt-4o`（`:63`） |
| **画图聚合** | reasoning model (o1) | `--model_agg_plots` 默认 `o3-mini-2025-01-31`（`launch:88`） |
| **写作** | reasoning model (o1) | `--model_writeup` 默认 `o1-preview-2024-09-12`（`launch:94`） |
| **引文收集** | small model | `--model_citation` 默认 `gpt-4o-2024-11-20` |
| **Review 推理** | **o4-mini** | `--model_review` 默认 `gpt-4o-2024-11-20`（`launch:118`），非 o4-mini |

**一个硬性工程约束（论文未点明）**：`backend_anthropic.py:34-36` 直接 `NotImplementedError` function calling。因此所有需结构化输出的环节（is_bug 判定、VLM 分析、metric 解析、best-node 选择、stage 完成评估）**必须走 OpenAI 系模型**，Claude 只负责自由文本的 plan+code 生成。o1 在 backend 被特判 `reasoning_effort=high` + `max_completion_tokens=100000`（`backend/__init__.py:76-77`）。

**模型版本说明**：本仓库快照早于论文定稿的模型矩阵（Sonnet 4 / o4-mini 尚未出现），由 3.5-Sonnet / o3-mini / o1 / gpt-4o 占位，但**角色分工与论文一致**。应以查询到的实际配置值为准。

**成本**：源码未直接给出每篇论文美元成本（论文正文给的 ~$15-20/paper 量级数字不在本次源码范围内）。可观测的成本控制点：单实验 1h 上限、各 stage max_iters（20/12/12/18）、`num_workers=8`、`save_token_tracker`（launch `:269`）统计 token。

---

## 7. 如何上手运行

### v1（Template-based，`AI-Scientist/`）

```bash
python launch_scientist.py \
  --model "<主模型>" \
  --experiment nanoGPT \
  --num-ideas 50 \
  [--skip-novelty-check] [--improvement] [--parallel N]
```
- 模板：`templates/<experiment>/`（12 个可选：nanoGPT/grokking/2d_diffusion/...）。换/加模板只需提供 `experiment.py`(支持 `--out_dir=run_i`)+`plot.py`+`prompt.json`+`seed_ideas.json`+`latex/template.tex`。
- API key：主模型 key + **`OPENAI_API_KEY`（评审强制 gpt-4o）** + **`S2_API_KEY`（Semantic Scholar 查新/引用，缺失降级限流）**。
- 环境：实验真跑 `subprocess python experiment.py`，需对应 GPU/依赖；`--parallel N` 一 idea 一 GPU。
- 耗时：单实验上限 2h，每 idea 最多 5 run；50 ideas 串行可达数十小时。

### v2（Template-free，`AI-Scientist-v2/`）

两个入口（先生成想法，再跑实验）：
```bash
# 1) 想法生成
python ai_scientist/perform_ideation_temp_free.py \
  --workshop-file ai_scientist/ideas/i_cant_believe_its_not_better.md \
  --model <ideation模型> --max-num-generations N --num-reflections 5

# 2) 树搜索 + 写作 + 评审
python launch_scientist_bfts.py \
  --load_ideas ai_scientist/ideas/i_cant_believe_its_not_better.json \
  --idea_idx 0 \
  [--add_dataset_ref] [--writeup-type icbinb|normal] \
  [--model_writeup ... --model_review ... --model_agg_plots ...]
```
- **关键配置 `bfts_config.yaml`**：`num_workers:8`（`:36`）、`exec.timeout:3600`（每节点 1h，`:18`）、各 stage `max_iters`（20/12/12/18，`:38-41`）、`search.num_drafts:3`（`:69`）/`debug_prob:0.5`（`:68`）/`max_debug_depth:3`（`:67`）、`num_seeds:3`（`:45`）、角色模型 `agent.code/feedback/vlm_feedback`（`:51,57,62`）。launch 会把它拷成 per-idea config 并注入路径。
- API key：**`OPENAI_API_KEY`（VLM/feedback/best-node/metric/review/写作-推理，必需）** + **`ANTHROPIC_API_KEY` 或 AWS Bedrock 凭证（code 生成走 AnthropicBedrock）** + **`S2_API_KEY`（引用）**。
- 环境：树搜索 worker 在子进程 `exec()` 真跑训练代码，**强烈建议 sandbox/容器隔离**（README 通常警告会执行 LLM 生成的任意代码）；有 GPU 则每 worker 独占一卡（`num_workers` 自动收敛到 GPU 数）；数据运行时从 HuggingFace 动态拉取（`data/` 初始为空）。
- 耗时：4 主阶段 × 动态子阶段 × 各 max_iters × 单节点最长 1h；一篇论文从想法到 PDF 通常数小时到一天量级（取决于 GPU 数与 scale-up 程度）。

---

## 8. 值得深入/魔改的切入点

按"改动收益 × 改动成本"给出建议入口：

1. **搜索策略（最核心）**：`parallel_agent.py:_select_parallel_nodes`（`:1931-2051`）控制 exploration/exploitation 平衡——可改 `processed_trees` 多树调度、`debug_prob`、`num_drafts`，或把 best-first 换成 UCB/MCTS。配合 `journal.py:get_best_node`（`:420-502`）改"最有希望节点"的 LLM 软决策 prompt（目前 `:442-448` 告诫别只看 val loss）。

2. **节点评价 / buggy 判定**：`metric.py:MetricValue.__gt__`（`:171-189`）改多 metric 聚合方式；`parallel_agent.py:_analyze_plots_with_vlm`（`:1020-1023`）改 VLM 把图标 buggy 的标准；`:713` 改 is_buggy 通道逻辑。

3. **阶段管理器**：`agent_manager.py:143-167`（阶段定义）+ 停止准则 `_check_stage_completion`（`:410-536`，含 stage3 的 scale-up 启发式 `:512-530`）——若做不同领域，这里是定制实验流程骨架的入口。注意先修 `:371-408` 的死代码。

4. **想法生成 / 工具扩展**：`perform_ideation_temp_free.py:128-266`（双层循环 + ReAct 协议）；新增工具实现 `tools/base_tool.py:BaseTool`（`:5-28`）并加进 `tools` 列表即可（手写 ReAct 不依赖 function-calling，跨供应商通用）。

5. **写作 / VLM 闭环**：`perform_icbinb_writeup.py` 才是论文 "VLM review during writing" 落地最完整的实现（reflection 里集成 `perform_imgs_cap_ref_review` + `detect_duplicate_figures` + 按页数预算筛图）；`perform_plotting.py:build_aggregator_prompt`（`:52-86`）改 compound figure 生成逻辑。

6. **评审增强**：`perform_llm_review.py` 已有完整 ensemble+meta-review 能力但默认未启用——在 `launch_scientist_bfts.py:311` 把 `perform_review(...)` 显式传 `num_reviews_ensemble=5, num_reflections=5` 并换推理模型，即可复现论文配置。结合 workshop 实物（§3.3、§5）做"AI 评审 vs 人类"对齐研究是现成的评测基准。

7. **模型后端**：`backend/__init__.py` + `backend_anthropic.py`（注意 function calling 的 `NotImplementedError` 约束 `:33-36`）——若要让 Claude 承担结构化输出环节，需在此实现 tool 支持，否则结构化任务被锁死在 OpenAI 系。

8. **代码-论文落差（不确定处）**：有三处需注意——(a) 代码模型版本早于论文（Sonnet 3.5 vs 论文 Sonnet 4）；(b) review 默认 gpt-4o 单 review vs 论文 o4-mini 5-ensemble；(c) workshop 仓库的 reviewer 是 v1 实现且不含 v2 `perform_llm_review` 源码。做复现时务必先查阅当前 `bfts_config.yaml` 与 launch 默认值确认实际配置，不要依赖论文描述。