# automationResearch — The AI Scientist 复现与评测工作区

> 围绕 Sakana AI《The AI Scientist》(论文 *Towards end-to-end automation of AI research*, **Nature 651, 2026**, DOI [10.1038/s41586-026-10265-5](https://doi.org/10.1038/s41586-026-10265-5);第二代见 arXiv:2504.08066)的研究、复现与系统评测工作区。
>
> **这里不是一个新框架**,而是把这套"自动化 AI 科研"系统完整跑通、逐层拆解源码、并用一套为**诚实性**设计的协议重新评测之后,产出的报告、复现实验、评测工具与数据。

被研究的系统能从一句研究主题出发,自动完成**想法生成 → 写实验代码 → 跑实验 → 画图 → 写论文 → 自动评审**的全链路。本工作区回答三个问题:它在非官方环境下**能不能跑通**、产出的论文**能不能用**、以及如何**可信地度量**它的产出。

> **本仓库不含官方代码与论文 PDF。** 三个官方 clone(`AI-Scientist` / `AI-Scientist-v2` / ICLR2025 workshop)各有自己的 git 与公开仓库,Nature 论文 PDF 有版权,均按 `.gitignore` 排除。对 v2 的适配改动以 diff 形式存于 [patches/](patches/)(用 `git apply` 打到官方 clone 上)。

---

## 目录

- [30 秒导航](#30-秒导航)
- [核心发现 (TL;DR)](#核心发现-tldr)
- [两套被研究的系统](#两套被研究的系统)
- [仓库结构](#仓库结构)
- [复现:快速开始](#复现快速开始)
- [评测方法(本工作区的增量)](#评测方法本工作区的增量)
- [出处与边界](#出处与边界)

---

## 30 秒导航

按"想了解什么"选起点,每篇都可独立阅读:

| 想了解 | 读这个 | 是什么 |
|---|---|---|
| 系统怎么运作(源码级) | [STUDY_NOTES.md](STUDY_NOTES.md) | 28k 字,论文术语 ↔ 三仓库 `文件:行号` 逐一对照 |
| 方法论拆解 | [METHODOLOGY_ANALYSIS.md](METHODOLOGY_ANALYSIS.md) | 想法算力分级 / 树搜索决策机制 / token / 三层超时 / 跨平台 |
| 一次完整复现长什么样 | [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) | 从"看论文找代码"到自有 GPU 跑通 v2、产出一篇 AI 自主完成的 ML 论文 |
| 不同 LLM 当"科学家"差多少 | [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) | v1/v2 × 多模型三层 benchmark(写作 / 树搜索 / 想法生成)+ 质量 gap 归因 |
| 这套东西能不能用、值多少 | [FEASIBILITY_REPORT.md](FEASIBILITY_REPORT.md) | 资源账本、照搬可行性、能力边界、论文可用性(含放大配置档试点) |
| 评测协议为何这样设计 | [PREREGISTRATION.md](PREREGISTRATION.md) · [COMPREHENSIVE_PROTOCOL.md](COMPREHENSIVE_PROTOCOL.md) | 数据产出前冻结的预注册:选择规则、声明层级、审计校准承诺 |

新读者建议顺序:**本文 → FEASIBILITY(结论) → BENCHMARK(证据) → STUDY_NOTES(机理)**。

---

## 核心发现 (TL;DR)

把这套系统跑到底之后,最值得记的几条:

- **"端到端很脆"有一半是框架 bug 的假象。** 定位 4 个会杀掉整个 run 或毁掉产物的工程缺陷(2 个修复、2 个流程规避);仅修复其中一个"工作节点异常被重新抛出杀全 run"的 bug,就把全流程成功率从 **2/5 提到 4/5**。
- **自动写作会系统性编造没做的实验。** 首次测得论文的**原始 grounded 率**(声明能在运行日志里找到支撑的比例)中位约 **30%**,主体落在 14–40%;官方流程写作后直接评审,没有任何"声明 ↔ 日志"核对环节。
- **外挂一个"零造假 gate"代价很小、收益是质变。** 经 gate 认证的诚实版相对原始版的"诚实税"均值仅 **−0.279**(28 实例口径,8 篇零税;28 = 26 收盘冻结实例 + 2 个放大配置档试点),过 gate 后每条结论都能逐条溯源到日志。
- **绝对评分没有区分度。** 5 票 ensemble 把所有机器论文压在 1.8–2.6 的窄带里;改用**成对盲评锦标赛**后,诚实版对官方原版胜率 **84.4%** [0.756, 0.905](三家族评委、双向、预注册)。
- **瓶颈是 LLM 往返,不是 GPU。** 小预算档 LLM 调用占墙钟 85–95%,GPU 利用率常年个位数;吞吐约 **1.6 篇/卡·天**,token 边际成本 **$20–30/篇** 量级。给这套系统升级 GPU 没有意义,买 token 吞吐才有意义。

> 完整数字、置信区间、威胁有效性的讨论见 [FEASIBILITY_REPORT.md](FEASIBILITY_REPORT.md) 与 [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md)。

---

## 两套被研究的系统

官方系统有两代,架构差异很大,复现路径也不同(两者均为官方 clone,不在本仓库,适配 diff 见 [patches/](patches/)):

| | v1 (`AI-Scientist`) | v2 (`AI-Scientist-v2`,论文主角) |
|---|---|---|
| 策略 | template-based:人写代码模板,迭代改写 | template-free:从想法**零起点**,agentic 树搜索 |
| 入口 | `launch_scientist.py` | `launch_scientist_bfts.py` |
| 流程 | 想法 → 改模板实验 → 写作 → 评审 | 想法 → 4 阶段树搜索(初始→基线→创新→消融)→ 聚合画图 → 写作 → 评审 |
| 结果稳定性 | 较稳(代码框架固定) | 方差大:每次自己重新设计实验,同一想法可能跑出相反结论 |

`AI-Scientist-ICLR2025-Workshop-Experiment` 是 Automated Reviewer 的复现 + 3 篇 AI 生成论文的实物。

---

## 仓库结构

```
automationResearch/
├── README.md
├── STUDY_NOTES.md              # 源码级研究报告(论文↔代码)
├── METHODOLOGY_ANALYSIS.md     # 方法论拆解
├── EXPERIMENT_REPORT.md        # 一次完整复现的结果
├── BENCHMARK_REPORT.md         # v1/v2 × 多模型 benchmark
├── FEASIBILITY_REPORT.md       # 可行性 / 资源 / 能力边界总评估
├── PREREGISTRATION.md          # 预注册评测协议(数据前冻结)
├── COMPREHENSIVE_PROTOCOL.md   # 评测协议细则
├── patches/                    # v2 适配 patch(git apply 到官方 clone)
├── tools/                      # 评测与编排工具链(见下)
├── selection/                  # 想法多评委筛选的留痕(ledger / shortlist)
├── papers/                     # 满预算生成的 AI 论文 PDF + 生成透明度附录
├── my_topic.md / my_topic.json # ideation 主题样本
└── .gitignore                  # 排除官方 clone + 版权 Nature PDF
```

[tools/](tools/) 是这个工作区**额外**搭的脚手架,不属于官方系统,包含:一键复现(`setup_ai_scientist.sh`)、token 计量、想法多评委 ensemble 选择(`select_ideas` / `merge_selection` / `recurate_8h`)、grounding 审计与定点改写、5 票 ensemble 评审、成对盲评锦标赛、审计器校准、批量编排与抢救链。

---

## 复现:快速开始

> **前置:** 一个 OpenAI 兼容的 LLM 端点(官方 OpenAI key 或任意兼容网关,key 经环境变量注入、不写进代码)+ 一块能跑 PyTorch+CUDA 的 NVIDIA GPU。官方栈只认 CUDA;本仓适配 patch 已新增 Apple Silicon (MPS) 支持,无 NVIDIA 时也可走 MPS 或纯 CPU 降级跑(慢)。本机想法生成是纯 API、不需 GPU。

```bash
# 1) 取官方 v2 并应用本仓适配 patch
git clone https://github.com/SakanaAI/AI-Scientist-v2.git
cd AI-Scientist-v2 && git apply ../patches/*.patch     # 具体文件见 patches/

# 2) 想法生成(纯 API,可在本机跑)
OPENAI_API_KEY=<key> OPENAI_BASE_URL=https://<your-openai-compatible-endpoint>/v1 \
python ai_scientist/perform_ideation_temp_free.py \
  --workshop-file ../my_topic.md --model gpt-5.5 --max-num-generations 1 --num-reflections 3

# 3) v2 树搜索(需 GPU;先按需改 bfts_config.yaml 的模型 / max_iters / num_workers)
OPENAI_API_KEY=<key> OPENAI_BASE_URL=https://<your-openai-compatible-endpoint>/v1 \
python launch_scientist_bfts.py \
  --load_ideas ai_scientist/ideas/<ideas>.json --idea_idx 0 \
  --skip_writeup --skip_review --model_agg_plots gpt-4o
```

> 用非官方端点时需要的适配清单(模型名路由规范化、引用搜索改用 OpenAlex、reasoning 模型输出解析鲁棒化、LaTeX 工具链用户态安装等)与动机见 [FEASIBILITY_REPORT.md](FEASIBILITY_REPORT.md) §2–3 与 [STUDY_NOTES.md](STUDY_NOTES.md);其中若干工程缺陷已整理为上游 issue。

---

## 评测方法(本工作区的增量)

官方流程在写作后直接评审。本工作区在中间加了一层为诚实性服务的协议,核心四件:

1. **零造假 gate** — 逐条抽取论文声明,与运行产物核对,把无支撑/矛盾的声明标出并定点重写,产出可逐条溯源的"诚实版"。
2. **审计器校准** — 用故障注入(真/假声明各半盲审)标定审计器的 recall / 假阳率,并跨模型家族双审,保证"诚实税"等结论不是审计器自身偏差。
3. **双仪器评分** — 绝对尺(5 票 ensemble)只判"同档",优越性主张挂在成对盲评锦标赛的胜率 + 置信区间上,评委覆盖多个模型家族。
4. **预注册** — 选择规则、声明层级、审计承诺在数据产出前冻结([PREREGISTRATION.md](PREREGISTRATION.md)),避免事后挑数据。

---

## 出处与边界

- **论文:** Sakana AI et al., *Towards end-to-end automation of AI research*, Nature 651 (2026)。官方代码仓库版权归原作者所有,本工作区只做 clone + 研究 + 适配。
- **结论边界:** 全部评测由 LLM 评委完成,与人类同行评审决策之间的效度鸿沟未闭合;实验域限于"代码即实验"的小数据 / 小模型 / 训练动力学诊断(T0–T1),更大规模(T2–T3)未跑通。详见 [FEASIBILITY_REPORT.md](FEASIBILITY_REPORT.md) §6–7。
- **可复现性提醒:** v2 每次重新设计实验(连数据集 / split / 代码都可能不同),同一想法多次运行方差很大;要复现某个结果需固化 best node 的代码与 seed,而非重跑树搜索。
