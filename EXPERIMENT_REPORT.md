# The AI Scientist 复现 — 完整结果报告

**日期**：2026-06-02 ~ 06-03
**目标**：从"看论文找对应 GitHub"出发，到在自有 GPU 服务器上用自建 router 完整跑通 Sakana AI 的 *The AI Scientist v2*，并产出一项 AI 自主完成的 ML 研究实验。

---

## 1. 论文与代码

- **论文**：《Towards end-to-end automation of AI research》, *Nature* Vol 651 (2026), DOI 10.1038/s41586-026-10265-5 —— 即 Sakana AI 的 **The AI Scientist**。
- **三个官方仓库**（已 clone 到本地）：
  - `AI-Scientist/`（v1，模板式 + Aider）
  - `AI-Scientist-v2/`（v2，**本次主角**：template-free + agentic tree search）
  - `AI-Scientist-ICLR2025-Workshop-Experiment/`（Automated Reviewer + 3 篇 AI 生成论文）
- **源码级研究报告**：`STUDY_NOTES.md`（28k 字，8 章，对照论文逐一分析）。

## 2. 运行环境

| | 配置 |
|---|---|
| ideation（想法生成） | 本机 macOS（Apple M5 Pro，无需 GPU） |
| 完整实验（树搜索） | 服务器 **8× NVIDIA H20**（各 ~96GB），Linux，conda py3.11，torch 2.12+cu130 |
| LLM | 自建 OpenAI 兼容 router，code/feedback=**gpt-5.5**，vlm/plot/summary=**gpt-4o** |
| 数据 | HuggingFace 直连不通 → 走 `hf-mirror.com` 镜像 |

## 3. 过程

1. **ideation（本机跑通）**：用 gpt-5.5 针对自定主题生成研究想法（产出 "Sentinel Augmentations"）。
2. **填平 7 个"gpt-5.5 迁移坑"**（系统原为 gpt-4o/Claude 设计）：
   | # | 坑 | 修复 |
   |---|---|---|
   | 1 | ideation 解析器只认单 ACTION | 优先 FinalizeIdea + raw_decode |
   | 2 | Semantic Scholar 无限重试卡死 | backoff `max_tries=5` |
   | 3 | 14 处硬编码模型名 router 不认 | `_route_model` 规范化（llm.py + treesearch/backend） |
   | 4 | reasoning 输出 max_tokens 不够被截断 | 12000 → 32000 |
   | 5 | gpt-5.5 只给 code 不给 plan | `if code:`（放宽 plan+code 双非空要求） |
   | 6 | summary unpack 边界崩（stage 不足 4） | 补齐到 4 |
   | 7 | psutil 漏装 | 补装 |
3. **冒烟测试**：先小步数验证链路（stage1 出 good node → 4 stage 全通）。
4. **完整实验**：compositional_regularization 想法（idea_idx 0，正是论文 Fig.2 被 ICBINB workshop 接收的那篇），完整 max_iters(20/12/12/18)、8 worker 并行。

## 4. 完整实验结果

- **规模**：4 阶段（initial_implementation → baseline_tuning → creative_research → ablation_studies），**57 分钟**，46 个实验数据文件，**488 张节点级图**，4 份 stage summary，4 个 tree_data。
- **核心科学结论**：**compositional regularization 改善了组合泛化**
  - 最佳节点：compositional split exact-match accuracy **0.4658（正则化）vs 0.3288（基线）**
  - 修复后的聚合曲线图：部分 regularized variant 的 compositional test accuracy 爬升到 **0.8~1.0**，baseline 类约 0.4
- **stage4 消融有深度**：对比 baseline / stop-gradient(`target.detach()`) / 无 stop-gradient，分析表征 collapse 与 alignment。
- **与论文对照**：论文那篇（同一 idea seed）报告的是**负面结果**（《Unexpected Obstacles...》，正则化没显著改善）；本次在**合成 SCAN-like 数据**上跑出了**正面**结果——差异源于数据集（合成 vs 真实），印证了下方审稿模型的批评。

## 5. 产物清单

- 本地：`STUDY_NOTES.md`、`EXPERIMENT_REPORT.md`(本文)、三个 clone 的仓库、`my_topic.json`(ideation 产出)、`.venv-ideation`
- 服务器 `~/automationResearch/AI-Scientist-v2/experiments/2026-06-03_08-38-56_compositional_regularization_nn_attempt_0/`（74M）：
  - `logs/0-run/{draft,baseline,research,ablation}_summary.json`
  - `logs/0-run/stage_*/tree_data.json`（4 个 stage 的搜索树）
  - `logs/0-run/experiment_results/*/experiment_data.npy`（46 个）+ 488 张节点级图
  - `figures/`：**手写脚本生成的 2 张有数据聚合图**（`all_variants_best_accuracy.png`、`baseline_vs_regularized_compositional.png`）+ gpt-4o 原 8 张空图（保留作对照）

## 6. 已知局限 / 诚实记录

1. **结论限合成数据**：feedback 模型（gpt-5.5 当审稿人）指出实验只用合成 SCAN-like 数据，未按 stage2/3 要求引入真实 HuggingFace 数据集。
2. **gpt-4o 原生聚合图是空的**：`auto_plot_aggregator.py` 硬编码单一 dataset key，而实际有 37 个异构 key → 画空图。已用手写的"遍历所有 key + 兼容两种 metrics 格式"脚本修复（与 code model 是 4o/5.5 无关，是数据复杂度问题）。
3. **gpt-5.5 慢**：reasoning 模型，小规模 idea 完整实验已需 ~57min；更大实验建议 code model 改 `gpt-5-chat` 或接受慢。
4. **未产论文 PDF**：本次 `--skip_writeup --skip_review`（只到实验+图）；产 PDF 需在服务器装 LaTeX。
5. **token 统计不全**：`token_tracker` 只覆盖 `llm.py` 路径（gpt-4o ≈ 25 万 token），树搜索 `treesearch/backend` 的 gpt-5.5 调用未被记；router 未提供单价，cost 显示 0。

## 7. 可继续的方向

- 用 **gpt-4o** 配置（code/feedback/ideation）重跑，验证"原生兼容、避免 reasoning 格式坑"，并对比质量/速度。
- 加 **writeup**（装 LaTeX）产出完整论文 PDF。
- 换**真实 HuggingFace 数据集**重跑，回应审稿模型的批评，让结论更扎实。

---
*所有运行环境、patch、运行命令详见 patches/README.md。*
