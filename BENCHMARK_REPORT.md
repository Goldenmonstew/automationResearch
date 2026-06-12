# The AI Scientist — v1/v2 × 多模型 Benchmark 对比报告

> 2026-06-03 在 8×H20 GPU 服务器上批量运行。LLM 走自建 OpenAI 兼容 router。
> 复用同一份已完成的 v2 实验(`compositional_regularization`)作为 L0 的固定输入,保证只变「被测模型」这一个变量。

## 0. TL;DR

- **deepseek-v3.2 是本次 benchmark 的黑马**:L0 论文质量最高(Overall **4**,唯一高于均值),L1 实验能力也最强(**3/8 good nodes**,gpt-4o 是 0/8)。
- **gpt-4o / gpt-5-chat 最稳**:原生兼容、不踩格式坑,L0 Overall 3,工程零额外适配。
- **reasoning 模型(gpt-5.5)写论文反而吃亏**:格式敏感(不裹 ```latex)、慢(16min vs gpt-4o 3min),L0 Overall 仅 2。
- **gemini-3.5-flash 两头不讨好**:L0 能写(Overall 2)但 L2 ideation 直接 0 产出(不遵循 ReAct ACTION 格式)、L1 树搜索挂死 50min。
- **所有 AI 写的论文都被自动评审 Reject**(Overall 2-4,阈值通常 6)——印证论文「端到端 AI 科研产出质量普遍不达标」的核心结论。
- **v1(template-based)vs v2(template-free)**:同为 gpt-4o,v1 Overall 3 = v2 gpt-4o 3,质量相当;但 v1 更稳(代码框架固定),v2 更灵活(从零设计实验)但方差大。
- **全流程从零做实验(5 模型并行,见 §7):stock v2 只有 2/5 跑通,修掉一个 framework bug 后变 4/5**——deepseek-v3.2 / gpt-5.5 / gpt-4o / kimi-k2.5 各产出完整论文,**全部 Overall 3 / Reject**(唯一失败的 minimax 是被 router 限流,非模型/框架问题)。初次 gpt-4o/kimi 死于同一个未处理异常 raise 的 framework bug(已定位并修复)。**两个核心结论:① 端到端"脆"主要是框架健壮性 bug 造成的假象,可修;② 能跑通的 4 个模型论文全收敛到 Overall 3——瓶颈是 v2 方法本身的天花板,不是模型,换谁都过不了评审线。**

## 1. 实验设计

| 层 | 测什么 | 怎么控变量 | 模型数 |
|---|---|---|---|
| **L0** writeup/review | 把同一份实验结果写成论文 + 自动评审的能力 | 固定同一个 v2 实验产物(idea+summaries+figures),VLM 图片反思固定 gpt-4o,只换写作模型 | 6 |
| **L1** tree search | 从 idea 零起点写实验代码 + 自调试的能力 | 同一 idea、同一降预算配置(stage 8/4/4/0, seed1, worker4, drafts2),只换 code/feedback 模型 | 4(3 可用) |
| **L2** ideation | 生成研究 idea 的能力 | 同一 workshop 主题(ICBINB 负面结果),max-gen 2 + 3 reflections | 8 |
| **v1** template-based | 对照组:人写模板 + Aider 迭代 | nanoGPT_lite 模板,gpt-4o 全流程 | 1 |

被测模型:gpt-4o, gpt-5-chat, gpt-5.5, gemini-3.5-flash, deepseek-v3.2, glm-5.1, (+L2: minimax-m2.5, kimi-k2.5)。

## 2. L0 — writeup/review(论文写作质量)

同一份实验,换模型写论文,再用 gpt-4o 自动评审(子分 1-5 / Overall 1-10,均 Reject):

| 模型 | Overall | Soundness | Presentation | Contribution | Confidence | PDF | 备注 |
|---|---|---|---|---|---|---|---|
| **deepseek-v3.2** | **4** | 3 | 4 | 2 | 4 | 496KB | 最高分;但引用了不存在的图(hallucination) |
| gpt-4o | 3 | 2 | 3 | 2 | 5 | 473KB | 原生兼容,最稳 |
| gpt-5-chat | 3 | 2 | 3 | 2 | 5 | 471KB | 与 gpt-4o 持平 |
| **v1 (gpt-4o, template)** | **3** | 2 | 2 | 2 | 4 | 209KB(6页) | 对照组,质量≈v2 gpt-4o |
| gpt-5.5 | 2 | 2 | 1 | 1 | 4 | 502KB | reasoning,Presentation 仅 1 |
| gemini-3.5-flash | 2 | 1 | 2 | 1 | 4 | 506KB | Soundness 最低 |
| glm-5.1 | 2 | 1 | 3 | 1 | 4 | 519KB | |

**结论**:deepseek-v3.2 > gpt-4o ≈ gpt-5-chat ≈ v1 > gpt-5.5 ≈ gemini ≈ glm。reasoning 模型(gpt-5.5)在格式密集的写作任务上**不如**普通 chat 模型。

## 3. L1 — tree search(实验/编码能力,降预算)

同一 idea 从零写实验代码 + 自调试,统计 good node 率(降预算 stage 8/4/4/0):

| 模型 | total nodes | good | buggy | 时长 | 主要错误 | 备注 |
|---|---|---|---|---|---|---|
| **deepseek-v3.2** | 8 | **3** | 5 | 834s(14min) | RuntimeError×2, KeyError | **唯一调出可运行实验** |
| gpt-4o | 8 | **0** | 8 | 186s(3min) | RuntimeError×3, IndexError, ModuleNotFound | 全 buggy,放弃快 |
| glm-5.1 | 0 | 0 | 0 | 5s(秒崩) | `400 messages 参数非法` | router 拒绝其在树搜索 backend 的消息格式,直接挂 |
| gemini-3.5-flash | 0 | 0 | 0 | 挂死 50min(timeout) | — | 树搜索卡死,无产出 |

**结论**:只有 deepseek-v3.2 和 gpt-4o 真正产出了节点数据。**deepseek-v3.2 显著强于 gpt-4o(3 good vs 0 good)**,且愿意花 4.5× 时间 debug;gpt-4o 3 分钟就放弃(全 buggy)。**gemini / glm 都没能进入有效搜索**——gemini 树搜索挂死 50min、glm 被 router 以 400「messages 参数非法」拒绝。这本身是强烈的工程兼容性信号:这两个模型不适配 v2 树搜索 backend。**caveat**:降预算偏小(为可控),gpt-4o 的 0/8 不代表它做不出——加大预算(论文级 stage 20/12/12/18)结果会不同;此处主要反映「相同小预算下的相对鲁棒性 + 接入兼容性」。多数失败节点在代码阶段就报错,没到 GPU 训练。

## 4. L2 — ideation(idea 生成)

同一 ICBINB 主题,每模型生成 idea(数 + 标题质量定性):

| 模型 | 成功生成 | 标题质量(定性) | 代表 idea |
|---|---|---|---|
| gpt-5.5 | 2 | 高,具体新颖 | "Shortcut Canaries: Predicting Real-World DL Failures", "Best Before Deployment: Estimating Expiration Dates" |
| kimi-k2.5 | 2 | 高 | "The Success-Failure Inversion", "The Determinism Paradox" |
| deepseek-v3.2 | 2 | 高 | "When More Data Hurts: Distributional Oversmoothing", "The Silent Saboteur" |
| gpt-5-chat | 2 | 中 | "Predictive Uncertainty Drift: Detecting Emerging Failure Modes" |
| minimax-m2.5 | 1 | 中 | "When Bigger Fails Faster: Inverse Scaling Degradation" |
| glm-5.1 | 1 | 中 | "The Late-Stage Shortcut Collapse" |
| gpt-4o | 2 | 偏模板化 | "Towards a Unified Framework for...", "Learning from Failure: A Systematic..." |
| **gemini-3.5-flash** | **0** | — | `Failed to parse LLM response (no ACTION)` — 不遵循 ReAct 格式 |

> 注:ideation schema 不含 Novelty/Feasibility 数值分,故按数量 + 标题创意定性比较。gpt-4o/gpt-5.5 的 json 含上一轮残留(append),已按本轮去重计数。

**结论**:gpt-5.5 / kimi / deepseek 的 idea 最具体新颖;gpt-4o 偏「Towards a Unified Framework」式套话;gemini-3.5-flash 因格式不兼容完全失败。

## 5. 工程兼容性(踩坑数 = 接入成本)

| 模型 | L0 | L1 | L2 | 需要的额外适配 |
|---|---|---|---|---|
| gpt-4o / gpt-5-chat | 可用 | 可用 | 可用 | 无(原生兼容,最省心) |
| gpt-5.5 | 可用 | n/a | 可用 | max_tokens 要够大 + 鲁棒 LaTeX 提取(reasoning 不裹 fence) |
| deepseek-v3.2 | 可用 | 可用(3/8) | 可用 | create_client / get_response_from_llm 路由到 router + 加白名单 |
| glm-5.1 | 可用 | 失败(400 messages非法) | 可用 | 同上;但树搜索 backend 消息格式被 router 拒 |
| gemini-3.5-flash | 可用 | 失败(挂死50min) | 失败(0产出) | 同上 + VLM 不能走 Google 原生端点;ideation ReAct 格式不兼容 |
| minimax / kimi | n/a | n/a | 可用 | 同 deepseek |

**为这套多模型 benchmark 新增的 patch(详见 patches/)**:
1. VLM 图片反思固定 gpt-4o(deepseek/glm 无视觉、gemini 要 GEMINI_API_KEY)
2. `llm.py create_client`:gemini 走 router、`else` 兜底走 router(不再 raise)
3. `llm.py get_response_from_llm`:第二个模型分发同样兜底 router + `content or ""` 防 None
4. `MAX_NUM_TOKENS` 4096→16384(整篇论文 + reasoning 留空间)
5. `_extract_latex_block()` 鲁棒 LaTeX 提取(4 处统一)
6. `AVAILABLE_LLMS` 白名单 +8 个 router 模型名(否则 argparse 入口直接拒)
7. v1:texlive2026 补装 `units`(nicefrac)+ `titlesec`(titletoc)宏包

## 6. 核心结论

1. **deepseek-v3.2 综合最强**:写论文质量最高(L0 Overall 4)+ 实验能力最强(L1 3/8 good)。性价比之选。
2. **gpt-4o / gpt-5-chat 最稳省心**:零额外适配,L0 Overall 3。要稳定就选它们。
3. **reasoning 模型(gpt-5.5)别用在格式密集环节**:慢 5×、格式敏感、L0 反而最低分之一。
4. **国产/gemini 模型在 v2 树搜索(L1)上兼容性差**:gemini 挂死、glm 被 router 以 400 拒;只有 deepseek-v3.2 能在树搜索里跑通(且最强)。gemini-3.5-flash 接入坑最多(VLM/ideation/树搜索三处都要特殊处理或直接失败)。
5. **AI 写的论文全部 Reject**(Overall 2-4):自动评审都没过,印证「端到端 AI 科研当前产出不达发表标准」。
6. **v1 vs v2 质量相当(gpt-4o 均 Overall 3)**,但 v1 框架固定更稳、v2 从零设计更灵活但方差大、坑多。

## 7. 全流程「从零做实验→写论文」对比(5 模型并行,最忠实复现方向)

这是最接近论文真实主张的一轮:每个模型跑**完整 pipeline**——共享一个 fresh idea「When More Data Hurts: Distributional Oversmoothing in Continual Learning」(deepseek 在 L2 生成的),独立工作副本 + GPU 分区,树搜索做实验 → writeup(**OpenAlex 真引用**,非 `(?)`)→ review(裁判统一 gpt-4o)。预算 stage 15/8/8/0、seed 1。

| 模型 | 全流程结果 | Overall | 论文 | 说明 |
|---|---|---|---|---|
| **deepseek-v3.2** | 完整论文 | **3** (S2 P2 C2 Conf4) | 275KB | 三轮(L0/L1/全流程)唯一全跑通 |
| **gpt-5.5** | 完整论文 | **3** (S2 P2 C2 Conf4) | 1605KB | 做了最多树搜索工作量(37k 行日志);writeup 慢但成 |
| **gpt-4o** | 完整论文(修 bug 后) | **3** | — | 初次死于下方 framework bug;**修复后重跑跑通 Overall 3**——证明初次"失败"是框架 bug 而非模型不行 |
| **kimi-k2.5** | 完整论文(修 bug 后) | **3** (S2 P3 C2 Conf4) | 400KB | 初次同样崩;修复后跑通(虽 `function_call` 偶尔空被跳过),Overall 3 |
| minimax-m2.5 | 废 | — | — | router 限流到只能 backoff(50min 仅 310 行),已终止 |
| gemini-3.1-pro-preview | 仅 L0 | (L0=1) | — | router 上 gemini 不支持强制 function-calling,做不了树搜索;只参与了 L0 写作(Overall 1) |

**关键发现:**
1. **端到端"脆"主要是 framework bug,不是模型**:stock v2 下初次只有 **2/5** 跑通;**修掉一个未处理崩溃 bug 后,gpt-4o 和 kimi 重跑都跑通 → 4/5**(deepseek/gpt-5.5/gpt-4o/kimi 全成,唯一失败的 minimax 是被 router 限流而非模型/框架问题)。这强烈说明:之前以为的"模型做不出实验"很大程度是框架健壮性 bug 造成的假象。
2. **4 篇成功论文全部 Overall 3 / Reject**——auto-reviewer 给的分和"哪个模型写的"几乎无关(同一 idea 下全收敛到 3)。**瓶颈是方法(v2 端到端流程)本身,不是模型**:换更强/更弱的 code 模型,最终论文质量都卡在 Overall 3、过不了评审线。这是本次最有信息量的结论。
3. **gpt-4o 和 kimi 初次死于同一个 framework bug**:`parallel_agent.py` 里 `except Exception: ... raise`——某个 worker 节点抛异常被捕获后**又 raise 重新抛出**,层层上抛直接杀掉整个 run。改成「记录+跳过该坏节点、继续搜索」(去掉 raise)后,gpt-4o 重跑直接跑通,kimi 也跑通(虽它自身 function_call 偶尔空被跳过)。这是 v2 一个真实的、会非确定性杀 run 的健壮性缺陷,**值得上游修**。
4. **deepseek-v3.2 再次是最稳的**——L0 写作冠军、L1 实验冠军、全流程一次跑通(初次就没崩),三轮全靠前。
5. **gpt-5.5 的反转**:它 L0 单独写作只有 2(reasoning 模型格式敏感),但作为「实验大脑 + 完整流程」反而成了(Overall 3),因为它肯花最多时间做树搜索。说明**单层能力 ≠ 端到端能力**。
6. gemini/minimax 出局原因不同:早先认为 gemini 全系不支持强制 fc、进不了实验阶段,**后查清是 max_tokens 探针假象(见 §10),gemini 其实能做实验**;minimax 被 router 限流到废。**能进实验阶段的模型,最终论文全是 Overall 3——模型差异在端到端层面被"方法天花板"抹平了。**

## 10. 订正:gemini 其实能做实验(2026-06-04)

§7/§9 里"gemini 全系不支持强制 function-calling、做不了树搜索"的结论**是错的**,根因:
- **真根因**:gemini 2.5/3.x 是 thinking 模型,吐 functionCall 前先烧 output token 思考(同一 trivial 调用实测 **130~3855 token,非确定性、1/5 概率爆 3855**),且 thinking 计入 maxOutputTokens。**max_tokens 太小 → 思考没完就 length 截断 → 空 tool_calls**。早先的预检脚本恰好传了 60-100 的小 max_tokens,造成"gemini 不支持 fc"的假象。
- **实证证伪**:① 不传 max_tokens(真实框架路径)时,三个 gemini 全部正常返回 tool_call(上游默认够大);② **gemini-2.5-pro 真实树搜索实测:0 个 `function_call is empty` 错误、0 截断、Stage1 找到 working implementation**——确实能做实验。
- **两处修复**:① router 侧对 gemini+tools 请求把 max_tokens 下限抬到 4096(env 可调,实测 21/21 通过,且因 gemini "答完即停" 不增 token 成本);② 框架侧 max_tokens 不够时本就走 None→上游大默认。
- **结论**:gemini(至少 2.5-pro)应纳入"能做实验"的模型;之前的排除是工具链 bug,不是模型能力。**教训:预检要贴近真实调用参数,小 max_tokens 会把 thinking 模型误判成"不会 fc"。**

## 8. Caveats / 局限

- L0 是「同一实验换模型写」——隔离了写作能力,但没测「不同模型设计的实验本身好坏」(那是 L1)。
- L1 用了**降预算**,good-node 率偏低且方差大,只反映「相同小预算下的相对鲁棒性 + 接入兼容性」,非各模型上限。4 模型里只有 gpt-4o(0/8)+ deepseek(3/8)真正产出节点,gemini 挂死、glm 被 router 400 拒。
- 自动评审用 gpt-4o,本身可能有偏好(对 gpt 系产出未必更宽容——实测 deepseek 反而最高)。
- L2 无数值 novelty 分,定性比较为主。
- 单次运行,未做多 seed 方差(时间限制)。

## 9. 结果可信度验证(hallucination 审计 + 评审器校准)

为了确认上面的结论站不站得住,做了两项独立验证。

### 9.1 评审器校准:Overall 3 是真信号,不是"拍平"
用同一个 gpt-4o 评审器审 3 篇**真·已发表论文**,看它会不会区分:

| 论文 | Overall | Decision |
|---|---|---|
| Attention Is All You Need(landmark) | **10** | Accept |
| 132_automated_relational(真 ICLR,边缘) | 5 | Reject |
| 2_carpe_diem(真 ICLR,边缘) | 4 | Reject |
| 本文 4 篇 AI 全流程论文 | **全 3** | Reject |

→ 评审器**会区分**(Transformer 给 10)。AI 论文(3)比边缘真论文(4-5)还低、远低于 landmark(10)。**"全收敛到 Overall 3 = 方法天花板"成立**,不是评审器对啥都给 3。

### 9.2 Hallucination 审计:论文普遍捏造 headline 结果(最重要的警示)
对四篇论文分别做了文本↔实际跑出日志结果的逐条比对:

| 模型 | 判定 | 关键问题 |
|---|---|---|
| **gpt-4o** | **重度捏造** | 实际只跑合成 MNIST(0-4/5-9),论文**凭空编造 CIFAR-10/100 + DomainNet 实验 + EWC/Replay 对比表**——数据集和 baseline 全是编的 |
| deepseek-v3.2 | 轻度漂移 | 核心忠实、多数数字精确,但 joint-training 结果**编反**(称 74.5% 实为 68.5% 退化,Fig 1 靠假数字);oversmoothing 指标数值+符号都错 |
| kimi-k2.5 | 轻度漂移 | 核心忠实,但**捏造 joint-training baseline**(Fig 5 数字日志里没有),且**漏报**一个真跑过的 EWC ablation |
| gpt-5.5 | 轻度漂移 | 数据集忠实,但 headline 的 DSI 表全编(日志全 0)、CIFAR "完美 1.0" 与日志 0.765 矛盾 |

**结论(本次最重要的警示)**:这些 Overall 3 的论文**不只是"质量低",而是含捏造结果**。共同模式是——**实验设置 + 大部分数字忠实,但支撑论点的 headline 结果/对比表/图常被编造或与日志矛盾**;最差的(gpt-4o)直接虚构整套数据集和 baseline。**一个相信论文的人会被误导。**这把"AI 端到端科研不可靠"从抽象担忧坐实成了具体的、可量化的造假行为——也正是它们该被 Reject 的根本原因。

> 方法:论文文本 + 实际 summary 日志逐条比对;评审器校准用 v2 自带 fewshot 真论文。

## 11. 质量 gap 归因:为什么本文论文"看起来"比官方 showcase 差(两边实物对比)

把**本文生成的 PDF**(deepseek/gpt4o 满预算)和**官方 showcase 的 3 篇 AI 论文**(`AI-Scientist-ICLR2025-Workshop-Experiment/` 的 annotated_paper + sakana_review)**逐页对读**后,结论有反转:

### 11.0 反转结论:差距大半是错觉,两边其实同一档

> "看起来差很多"是在比 **封面 vs 草稿**:官方 showcase 是 **43 篇里手工精选 3 篇、其中仅 1 篇过审**的尾部最佳;本文看的是**未经筛选的单次产出**。论文原文自白:"We manually filtered the most promising outputs",筛选门槛(切题 / 代码能跑 / 格式正确)恰好会把本文的破相项全刷掉。团队内审也承认 3 篇"**none met the higher bar for a main ICLR conference**"。

### 11.1 实证:官方过审那篇也是玩具+退化结果

| | 官方 comp-reg(**唯一过审** 6/7/6) | 官方 pest(3/7/4,没过) | 官方 label-noise(3/3/3,没过) | 本文 deepseek(4)/gpt4o(3)/gemini(3) |
|---|---|---|---|---|
| 模型/数据 | 单层 LSTM + **合成单数字算术** | ResNet-18 + 真实 Crop Pest 等(最实在) | ResNet-18 + CIFAR/MNIST | SimpleNN + **make_blobs 合成** |
| 最佳结果 | **acc=100/0 退化** | mAP(但 domain-adapt 造假) | ECE(但温度缩放写了没跑) | **test acc=100 退化** |
| 本质缺陷 | 玩具+退化+57%训练集泄漏 | 结论造假(多数据集联训实际没做) | hallucinate 实验+重复图 | 玩具+退化+缺图+(?)引用 |

→ **官方过审那篇的退化(acc=100/0)和本文 deepseek 的 test-acc=100 同根同源**;本文的 3-4 分**正好和官方那 2 篇没过的论文同档**。sakana_review 实证:comp-reg 自评 Conference 仅 4/10(borderline reject),pest/label-noise 都是 Reject。

### 11.2 真实差距只有三块(且都不是"模型烂")

1. **Claude Sonnet 4 当代码大脑(最致命)**:论文树搜索默认 Sonnet 4 写代码,**本文全程无 Claude**。实测(§3 L1):同一 idea,gpt-4o **0/8 好节点**、deepseek 才 3/8。代码大脑强弱直接决定实验能否落成"非玩具"。
2. **节点预算(Fig 3c 实锤)**:论文 compute↔质量曲线显示 **5-16 节点平台期 ~3.3、30 节点才跳到 ~3.95**。本文 `num_workers=2`、有时 stage4=0,有效节点落在 5-16 平台区 → **Overall 3 是曲线低预算端的"应得值",不是失败**。官方跑曲线右端 30+ 节点。
3. **模型栈整体降配**:ideation o3 / 写作 o1 / 评审 o4-mini×5 ensemble vs 本文 router 代理 + gpt-4o 单评审。

### 11.3 要扣除的"假失败"(纯环境/工具链,非方法或模型)

- 参考文献空 + 正文 `(?)`:无 S2 key(原版有),已用 OpenAlex 绕过。
- figure 空白/缺图:plot 聚合 bug(硬编码 dataset key)+ router 模型不裹 ```latex。
- gemini/glm 接入坑:router 兼容性,原版用官方端点没有。

→ 这些让 PDF "一眼坏",但**与科学质量无关**;扣掉后剩的真实缺陷(玩具/退化/hallucination)官方那几篇**一篇不少**(论文 Limitations 自列 hallucinations / inaccurate citations / duplicating figures / underdeveloped ideas)。

**§11 结论**:差距 = **survivorship(43→3→1)+ 无 Claude Sonnet 4 代码大脑 + 节点预算只到曲线平台区**,外加一层 router/无-key 的工具链破相。**不是"模型烂到只配 3 分"**——补上前两块(尤其 Claude 代码大脑 + 30+ 节点),产出会和官方 showcase 落同一档;而官方 showcase 本身按真实顶会标准也是弱论文。

## 12. 满预算复跑:预算到底能不能破 Overall-3 天花板

§7 的全流程是降预算(stage 15/8/8/0)。为隔离"预算"这个变量,4 个能跑通树搜索的模型各做一次**满预算**全流程(stage 20/12/12/18、num_seeds 3、exec.timeout 1h,同一 idea「distributional oversmoothing」,writeup/review 统一 gpt-4o 兜底/裁判),结果:

| 模型 | 满预算 Overall | 节点(good 率) | 实验数据 | 备注 |
|---|---|---|---|---|
| **deepseek-v3.2** | **4** | 64(41%) | 合成(SimpleNN+make_blobs) | 唯一 >3;退化 100% 结果包装成阴性发现 |
| gpt-4o | 3 | 63(68%) | 合成 | 论文 hallucinate 成 CIFAR/DomainNet(造假) |
| gemini-2.5-pro | 3 | 60(50%) | 真实 | 部分图缺失 |
| **gpt-5.5** | 3(S2 P2 C2) | 63(**66%**) | **真实 MNIST/Fashion/CIFAR** | 代码大脑强,但 writeup 弱拖 Presentation=2 |

**核心结论(预算 vs 天花板)**:
1. **满预算只把 deepseek 从 3 抬到 4,其余三个仍卡 Overall 3**。对照降预算(§7 全 3),预算从 stage15/8/8/0 → 20/12/12/18 **只带来边际提升**,远没破评审线(6)。
2. **真实数据集 ≠ 高分**:gpt-5.5 自发用**真实 MNIST/Fashion/CIFAR**(非合成退化)+ 66% good 节点(代码大脑最强之一),**仍只 Overall 3**——因为 ① writeup 弱(Presentation 2)② auto-reviewer 对 AI 论文的压缩标尺。这反驳了"gpt-5.5 做大脑不够"的早先印象(它是强大脑),并实锤"**真实实验本身突破不了 Overall-3 天花板**"。
3. **由此推出 Phase 2 的方向**:要破 3,光靠满预算(20/12/12/18)和真实数据不够,需要 **Fig3c 右端的更大节点预算(scaled 40/30/30/40,目标 100+ 有效节点)+ ensemble 评审 + grounding 零造假验证**——即综合复现协议 v3(`COMPREHENSIVE_PROTOCOL.md`)。生成透明度附录见 `papers/*_GENERATION_APPENDIX.md`,token 实测 deepseek 满预算 ≈ 4.3M/篇。

## 13. 48h 冲刺:零造假 gate + 头对头官方 showcase(2026-06-10/11)

§12 指出的方向(更大节点预算)在实跑中被证伪了一半:scaled 档(stage 40/30/30/40、num_seeds 10、exec.timeout 8h、每路 1 worker)8 个 idea 跑满 72h 墙钟被全部强杀,0 篇论文产出——根因是 num_workers 被代码夹到 min(workers, 可见 GPU 数)、8h 节点超时把单节点实验撑到数小时且把 128 核 CPU 打爆(load 289+)、num_seeds=10 在单 worker 下等于 40 次串行重跑。结论:**预算的正确花法不是把单棵树撑大,而是按官方预算多跑树 + 把省下的算力投给质量与诚实性的后处理**。据此改为 48h 冲刺,评测规则在数据完成前预注册(`PREREGISTRATION.md`,commit 63f26f2)。

### 13.1 冲刺管线

- **生成**:4 路并行(各 2 GPU),官方预算(20/12/12/18、seed 3、drafts 3、exec 1h),code/feedback=gpt-5.5,writeup/citation/review/agg=gpt-4o;每 idea 10h 硬上限;idea 队列按 5 评委共识分排序(`selection/`)。单 idea 墙钟终版实测 2.2-10h(中位 ~6h,5/20 触 10h 上限)。
- **抢救链**(`tools/rescue_writeup.py`):被强杀的树离线重建 stage summaries(journal 每步落盘,summary 仅在全阶段结束才写——框架缺陷)→ 聚合 → writeup → review。实测被杀树的论文与原生完整 run 同分数带。
- **强制重聚合**(`tools/fix_aggregation.py`):官方聚合机制让 LLM 猜数据结构,0-1 图且 reflection 5 轮修不动;把 npy 真实嵌套结构 dump 给模型重写聚合脚本,一次命中 8-9 图。
- **5 票评审尺**(`tools/ensemble_review.py`):gpt-5.5×3 + gpt-4o×2 + meta(NeurIPS 表单),对我方论文、官方 showcase、校准集(Attention=9.0、边缘真 ICLR=3.2)同尺适用。
- **grounding gate**(`tools/grounding_audit.py` + `tools/reground_rewrite.py`):逐条声明对照 journal/实验数据/图,判 grounded/无支撑/矛盾;定点 find/replace 改写(矛盾→改成日志真值,无支撑→删除或改为诚实局限)→ 重编译 → 重审计,循环至 ≥95% grounded;含编译修复前置(writeup 末轮 reflection 常留坏 tex)。
- **诚实重写**(`tools/honest_rewrite.py`,writeup-v2):从 grounded 材料整篇重写干净叙事(非手术删改),输入仅限机器产物,产物强制重过 gate。
- **审计器校准**(`tools/audit_calibration.py`):fault-injection,20 条已知捏造 + 20 条已验真声明盲审。

### 13.2 首发测量量(本工作真正的原创贡献)

1. **AI 论文 writeup 层系统性造假率**:原始论文逐条声明审计,grounded 比率仅 **14%-40%**,且与实验完整度无关(完整 4 阶段树的论文同样只有 17-36%)——造假是写作环节的系统行为,不是实验不足的症状。官方 3 篇(干净原版)纯文本一致性审计也有 **14-32% 声明内部自相矛盾**。
2. **诚实税**:强制全部声明可溯源后,5 票分平均下降约 **-0.25**(区间 0 至 -0.6);**7 篇论文零税或仅 -0.2 守住 2.4**——实验真有信号时诚实几乎不掉分,印证"G1×G2 同时成立的支点是实验扎实"。
3. **gate 收敛性**:15+ 篇全部在 ≤3 轮内收敛到 95%-100% grounded(多数 100%),每篇产出"声明→日志"溯源证书。
4. **审计器可信度**:注入造假 recall **100%**(20/20)、误伤真声明 FPR **5%**;跨家族双审(deepseek-v3.2)Cohen's κ=**0.849**。
5. **分母漂移警示**:claim 重抽会使同一论文跨轮 grounded% 漂移(实测 95.3%→93.9% 纯因重抽),跨轮比较须冻结分母。

### 13.3 头对头官方 showcase

官方 3 篇基线用其 arXiv 报告(2504.08066)附录 C.1-C.3 切出的**无标注原版**(GitHub 的 annotated 版会压低评审分 0.2-0.4,作废)。

- **绝对尺(5 票)**:所有机器论文(我方 raw 2.4-2.6 / 我方诚实版 1.8-2.4 / 官方 2.0-2.6 / 四篇早期满预算论文 2.4-2.6)压缩在同一分数带;该尺粒度 0.2=单评委单票、gpt-5.5 评委对 2-3 档 mode-collapse,**只能支持"同档"结论**。
- **成对盲评锦标赛(预注册的优越性仪器)**:我方 5 篇诚实认证版 vs 官方 3 篇原版,3 评委家族 × 双向呈现 × 90 场:**总胜率 84.4%,Wilson 95% CI [0.756, 0.905]**;分评委 gpt-5.5 100%、deepseek-v3.2(对双方栈均中立)93.3% [0.787, 0.982]、gpt-4o 60% [0.423, 0.754](唯一跨 50%,如实披露);对官方逐篇胜率 90%/86.7%/76.7%,CI 下界全部高于 50%。注意让步条件:我方出战的是删尽无支撑内容的诚实版,对方是含造假的原版。
- **G2 头对头**:我方 gate 通过率 100%(15+ 篇,每篇带证书);官方 0/3——无公开实验日志使其根本无法接受声明级审计,而其文本级弱审计也不过(上文内部矛盾率),且其造假已由官方自己的标注版 PDF 与报告 Limitations 承认。两层审计深度不对称是数据可得性的硬约束,两侧数字不可同列比较。

### 13.4 双口径 survivorship

我方公开**全部** N 篇分数分布(原始+诚实+重写三版谱系,被杀/抢救树打 flag 单列,见 `sprint/scoreboard.json`),从中按预注册规则(诚实分 top-5)选 showcase;官方为 43 选 3(7%)且其余 40 篇分布未公开。

### 13.5 Threats to validity

评委面板无 Anthropic 家族模型(官方论文由 Claude/o1 栈生成,我方由 gpt 栈生成,deepseek 评委对双方中立是主要缓解);官方样本 n=3;绝对尺在 2-3 档无校准锚;审计员与改写循环存在 in-loop 耦合(由 fault-injection 校准 + 跨家族双审缓解,held-out 终审为后续工作);树搜索逐 run 不可复现(v2 固有);gpt-4o 评委票与 writeup 模型同源。

### 13.6 新增框架缺陷清单(复现者注意)

聚合器猜数据结构产 0 图且 reflection 修不动;writeup 末轮 reflection 覆盖写坏 template.tex(样式名写错/Lonely \item,而 PDF 是早期好版本编译的,2/8 命中);writeup 重试开场删旧 reflection PDF(与并行读 PDF 的工具竞态);`find_pdf_path_for_review` 在无 PDF 时 UnboundLocalError 使整个 run 以 rc=1 崩溃;stage summary 延迟落盘(全阶段结束才写)使被杀树必须离线重建;psutil 清理按关键词杀全机 python 进程(并行多 run 时第一个完成者会误杀其余,本文已中和)。

### 13.7 收盘终版(2026-06-12,全部数字以本节为准)

冲刺 20 棵树全部消化(14 个 rc=0、5 个触 10h 上限、1 个评审环节崩,后两类经离线补链全部成文),加 6 棵抢救树共 **26 个论文实例(20 个独立 idea)走完 gate 链**:

- **gate**:25/26 达 ≥95% grounded,**17/26 达 100% CLEAN**;1 篇(warmup_weight_decay)止步 93%,按预注册规则标记、不进 showcase。原始版 grounded 中位 ~30%(17 篇在 14-40%、2 篇 ~53%);2 篇抢救树"原始版"即达 95/98%——抢救链 writeup 的输入是从日志重建的真实 summary,**输入真实时编造骤降,造假率主要由 writeup 输入质量决定**。
- **诚实税终版**:26 篇均值 **-0.285**(区间 0 至 -0.6),7 篇零税;诚实版分布 2.4×9 / 2.2×7 / 2.0×7 / 1.8×3;原始版带宽 2.4-2.8。
- **showcase top-5**(预注册规则,9 篇并列 2.4 按策展共识分破平;同 idea 双版本取最优版并披露):label_noise_diagnostic(17.0,触上限树)、augmentation_noise_interaction(16.75,抢救+诚实重写版)、confidence_maximizing_augmentation(16.75,原生)、bn_momentum_small_data(16.33,原生)、compressibility_early_stopping(16.25,评审崩树完整)。并列落选(共识 16.0):test_time_simplicity_gap、class_learning_order_imbalance、kernel_placement_small_data。打包(PDF+证书+评审)见 `sprint/showcase/`。
- **Presentation 维度实测**(67 篇次 ensemble):机器论文全体 1.0-2.0(我方原始均值 1.48、诚实版 1.33;官方干净版 1.27-1.33),人类校准论文 2.6-3.8——机器/人类区分度最大的维度。
- **资源账本**:全战役 token 中心估计 ~1.4 亿(树搜索 1810 节点 × ~67K/节点 ≈ 1.2 亿 + writeup/评审硬记账 816 万 + 评测链估算 1000-1500 万);GPU 预留 ~1140 GPU·h(util 0-12%、显存 0.5-2GB/96GB,严重过剩);折合 ~540 万 token/篇、按 gpt-4o 牌价粗折 ~$20/篇量级(v1 预印本宣称 "less than $15 per paper",量级相当;Nature 版未报任何算力/成本)。吞吐 ~1.6 篇/卡·天(20 篇 ÷ 37.9h×8 卡)。
- 资源/可行性/能力边界的完整分析见 `FEASIBILITY_REPORT.md`。
