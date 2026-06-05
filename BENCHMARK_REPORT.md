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
