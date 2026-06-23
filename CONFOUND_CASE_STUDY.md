# 自主科研产出的有效性:一个可证伪的混淆案例研究(8 个发现 / 2 批 convenience sample)

> 2026-06-23 初稿;2026-06-24 经多轮独立对抗复核后**全面软化措辞**(所有"总体率/定律/系统级"表述降级为便利样本上的假设生成性观察)。本文是对早期 `PROJECT_REPORT.md` / `GAP_ANALYSIS.md` 中"质量天花板 = 科学有效性"这一**未严格证明**断言的严谨化复核。方法上刻意**不依赖任何 AI 评委判定有效性**(规避"用同源 AI 判 AI"的循环——该路径已被 11 样本 FPR=1.0 证伪)。
>
> **定位(重要)**:这是**一组事后汇集的便利样本(convenience sample)案例系列**,不是 rate study,也不是受控对比。结论是**假设生成性**的:它能客观证伪"被审查的具体发现",但**不能**估计总体混淆率、不能证明"混淆率随主张锐度变化"。

---

## 0. 一句话

对自主系统(The AI Scientist)在本工作区产出的 8 个头条"发现"做案例审查:

- **3 个锐利因果机制式发现(gen3_mechanistic 批)都站不住,但方式不同**:2 个是**效应被泄漏伪造**(移除泄漏后效应消失——客观消融证明),1 个是**循环设计且被其自身数据证伪**。
- **5 个多主题发现(诊断/对比/经验类)**:1 个有**剂量混淆**(归因失效——效应未必凭空伪造,只是不能归因到所称机制),4 个**未检出致命混淆**(作者代码审计判定,非盲审/外部裁决)。
- **评审检测**:在 3 个案例里,评审对混淆的反应与其在 writeup 中的显性程度一致(suggestive,n=3,评审设置不统一)。

**两批是事后拼出来、不可直接比较的便利样本**(生成时间/模型/想法池/筛选都不同,分组也是事后划的),所以"锐利因果主张更易出严重设计问题"只是**待检验的提示**,不是 base rate,也没证明随主张锐度变化。实务提示:**仅靠论文文本的评审不足以发现部分代码级混淆;代码审查 + 针对性消融是更强、更直接的补充**(未与人类 code review / 预注册 checklist / 外部裁决比较,故不声称"唯一可靠")。

D1(发现是否站得住)用客观消融 / 构造论证,**全程零 AI 评委**;D3(评审检测)用既有评审产物。

---

## 1. 被检验的声明与旧证据的硬伤

**声明(旧文档,未严格证明)**:自主科研质量封顶 ~3/10,根因是科学有效性(设计有混淆、系统识别不了),非写作/算力/效应大小。

**旧证据不可识别**:"论文是坏科学"(Gap C)与"评委量不出有效性"(Gap B)纠缠,方向相反;且旧证据用 AI 评委/AI 混淆检测器判有效性,而这把尺自身已被证不可靠(FPR=1.0)。

**破法**:D1 = 客观消融(移除被疑混淆,看招牌效应是否消失,可证伪);D3 = 检验评审栈能否识别 D1 已证实的混淆。

---

## 2. D1-A:3 个锐利因果机制发现都站不住(方式各异)

| # | finding | 问题类型 | 证据 | 工具 |
|---|---|---|---|---|
| idx0 | gradient_alignment_grokking | **效应被泄漏伪造** | teacher 训在含 val 全表;移除泄漏 → 效应消失 | 5-seed 消融(决定性) |
| idx2 | noise_alignment_double_descent | **效应被泄漏伪造**(干净监督注入) | 打乱 anchor 标签 → 方向效应消失 | 3-seed 消融(决定性) |
| idx1 | symmetry_acquisition_grokking | **循环设计 + 自身数据证伪** | 指标 by-construction 置零;控制组无对称也泛化 | 既有数据 + 构造论证 |

- **idx0**:"aligned" 方向 = 朝 `ref_state` 推,而 ref_state 是**训练在含 val 全表**的 teacher(已 100% 解 val)。5-seed 消融(只换 ref 来源):REPLICATE(含 val)5/5 grok@2750;DECOY_PERM(置换输出)5/5@12750(慢 4.6×);DECOY_RANDOM/TRAIN_ONLY/iso/anti/none 全 0/5。→ **头条量级主要由"朝已解任务 teacher 蒸馏 + 泄漏 val 答案"解释;当前消融不支持原论文所称的"噪声对齐"机制,但 DECOY_PERM 仍慢 grok,故不排除更弱的特征蒸馏/对齐效应。**
- **idx2**:"aligned 噪声" = 干净 anchor 梯度注入 30% 标签噪声训练。消融(digits,3 seeds):CLEAN/anti 0.22(远差于 baseline 0.076);**SHUFFLED/RANDOMDIR 臂全部塌回 baseline** → 方向效应 = 干净监督注入,非噪声对齐。注:原始效应本就弱(aligned≈baseline、无真 peak shift)。
- **idx1**:`symmetry_enforced` 在平移 orbit 取均值=硬编码不变性,RSV by-construction 置零(acq_epoch=1 是机械的)。其**自身控制组证伪头条**:baseline/weight_decay_control 不获对称也泛化、symmetry_prevented 没被阻断(val 0.987/0.89/0.71)、且根本非 grokking(图像分类)。对循环混淆,re-run decoy 非正确工具——既有数据 + 构造即足。

注:idx0 与 idx2 复用同一"干预朝答案推"模板(teacher / clean-anchor),故二者的**独立信息量比表面 2 例更小**。

---

## 3. D1-B:5 个多主题发现 —— 1 剂量混淆 + 4 未检出致命混淆

对不同主题分布的 5 个发现(跨 noise/正则/增强/校准/组合泛化;含 diagnostic 与 interventional)做**作者代码审计**(独立 agent 初分 + 作者逐个对代码核验)。**注意:这是作者判定,未经盲审/外部裁决/inter-rater;"未检出致命混淆"≠"结论可靠"。**

| finding | 主张类型 | 判定 | 依据(已核代码) |
|---|---|---|---|
| **regularization_interference** | causal-mechanism | **剂量混淆(强结构嫌疑)** | 配对=两正则**全强度相加并集**(行 439-453),无 dose-match → "组合更差"分不清是"信号冲突"还是"总剂量更大"。**注:效应未必伪造,是归因失效;是否算致命混淆待 dose-matched 消融或外部复核确认。** |
| label_noise_diagnostic | diagnostic | 未检出致命混淆 | 噪声仅训练集、selection 仅验证集;招牌 pooled-rho 被数据集难度抬高=质量缺陷非泄漏 |
| augmentation_noise_interaction | interaction | 未检出致命混淆 | 噪声仅训练集、干净标签评估、(aug,noise)vs(no-aug,noise)公平对照 |
| confidence_maximizing_augmentation | intervention | 未检出致命混淆 | 选择仅用模型自身置信度 argmax、不碰标签;max-of-k 序统计偏差**反**向不利校准主张=非作弊 |
| compositional_regularization | causal | 未检出致命混淆 | 持出新组合真隔离;regularizer 只用训练集 source 端、不碰 label/eval;metric≠loss 项 |

- 唯一被判有问题的(reg_interference)是 5 个里最 causal-mechanism-framed 的,且其混淆 method-visible。
- 4 个"未检出"都带质量硬伤(非混淆):pooled 头条被数据集难度抬高、缺 rate=0 / generic-reg 对照、单 seed、把 synthetic toy 命名成真 benchmark(SCAN/GeoQuery 实为手搓玩具语法)。

**逐案小结(避免不稳的分子定义)**:8 个里 **2 个效应被泄漏伪造**(idx0/idx2)、**1 个循环+自证伪**(idx1)、**1 个剂量混淆/归因失效**(reg_interference)、**4 个未检出致命混淆**。把这些混成单一"致命混淆率"会掩盖类型差异——按类型逐案看更可靠。

---

## 4. D3:评审检测与文本可见度(n=3,suggestive)

| finding | 混淆在 writeup 的可见度 | 评审栈反应 |
|---|---|---|
| idx1 symmetry | 完全可见(对称化架构写进方法) | **抓住、判致命**:"heavily confounded ... does not isolate grokking" |
| idx2 dd | 半可见(提"从 held-out set 估计 direction") | **部分**:5 票里 1 个提 "possible test/validation leakage",仅作 limitations 建议 |
| idx0 grok2 | 藏在代码(writeup 不提 teacher 训 val) | **完全漏判**:6 份评审 + grounding(100%)无一提及;拒稿理由全是表层 |

**在这 3 个案例里,评审对混淆的反应与其在 writeup 中的显性程度一致,提示文本可见度可能影响检出。** 但 n=3、单系统、事后可见度标注、评审设置不统一(回收既有产物,非固定 reviewer/prompt/预算),**未把可见度从其它差异中隔离**,故不能声称"单调依赖"或"系统性漏掉"。grounding(声明 vs 数据)对这些都无效——数据真的显示了效应。

---

## 5. 合成(克制)

- **Gap C(生成器产混淆)**:这些案例提供**支持性证据**,并提示可能有结构(锐利因果机制式主张在本样本中更易暴露严重设计问题);是否一般成立需更大、预先定义的样本。D1 的逐案证伪不依赖任何 AI 评委。
- **Gap B(评估器量不出有效性)**:本样本中评审能罚表层、能抓写在纸面的混淆,但漏掉藏在代码、未披露的混淆;这是**支持性证据**,非系统级真值。
- **二者互锁(提示)**:制造表层扣分的"欠描述",往往也是藏住有效性缺陷的东西。

**对旧断言的修正**:"天花板 = 科学有效性"过于笼统且未证。更克制版:**在这组便利样本里,自主系统在锐利因果主张处更易出现严重设计问题(含 2 例可证伪的泄漏伪造);而仅靠论文文本的评审 + grounding 对代码级混淆检出不足。** 不外推为总体率或定律。

---

## 6. 含义

- **对"用 AI 评判 AI 有效性"(trust-verify 设想的第二层)**:仅从**文本**做混淆检测,对代码级混淆不足;更直接的手段是审实验**代码** + 落到**可证伪消融**(本研究即走此路)。未与人类 code review / 预注册 checklist 比较,故不主张它是唯一手段。
- **对"无人干预自主科研"**:有效性这一环在文本层难以自我纠正——本样本里生成器在锐利因果主张处产严重设计问题、文本评审按可见度漏判。是否一般成立未证。

---

## 7. 范围与对有效性的威胁(据实,优先列最致命的)

- **便利样本,非随机/非预注册**:8 个发现是事后汇集;"锐利因果机制批/多主题批"是**事后分组**;两批**不可直接比较**(生成时间、模型、prompt、想法池、筛选都不同)。故 3/3 与 1/5 只是描述性差异,**不是 base rate**。
- **证据强度严重不齐**:2 个做完整 re-run 消融(idx0 5-seed、idx2 3-seed)、1 个构造+既有数据证伪(idx1)、5 个仅作者代码审计;**无 blinded external adjudication、无 inter-rater reliability、无置信区间**。"未检出致命混淆"是作者判断,非客观事实。
- **分子定义不稳**:把"泄漏伪造效应"与"剂量归因失效/循环"混成单一"致命混淆"会误导;本文已改为按类型逐案陈述。
- **D3**:n=3、单系统、事后可见度标注、评审设置不统一;只是提示,不能声称单调/系统。
- **reg_interference**:剂量混淆是强结构嫌疑,但未做 dose-matched 消融;效应是否真实/能否归因到"冲突"未定。
- **同一系统 / 同一模型代际**(The AI Scientist,gpt-5.x 级);跨系统、更强代际未测。任务族:代码即实验、小模型小数据。
- **强处(据实)**:idx0/idx2/idx1 的逐案证伪是**可证伪的实验/结构事实、不依赖任何 AI 评委**,对"这三个具体发现"是决定性的——这部分免疫早期报告的循环论证。

---

## 附录:产物与可复现指针

- 消融脚本:`tools/d1_confound_ablation.py`(idx0)、`tools/d1b_feature_vs_answer.py`(idx0 特征 vs 答案)、`tools/d1c_dd_ablation.py`(idx2);混淆检测器原型 `tools/confound_detector.py`(注:基于文本,已知对代码级混淆不足)。
- 结果:`tools/track_b_results/{d1_results,d1b_results,d1c_results}.json`(随本仓库归档,可复核 seed 级数字)。
- 标的 run(服务器,best node):idx0 `sprint/auto/grok2/...gradient_alignment_grokking...`(38142299);idx1 `sprint/gen3/lane1/...symmetry_acquisition_grokking...`(0cfbc03b);idx2 `sprint/auto/demo_dd/...noise_alignment_double_descent...`(48216db9);D1-B 5 例在 phase2/slot* 与 sprint/lane* 及主仓 experiments 下。这些 run 目录与既有评审 JSON 在服务器,**未随仓库归档,文中相关数字属作者报告**。
- 所有消融脚本仅改"参考来源/对照臂",model/optimizer/hp/噪声注入/判据逐行保持原节点代码。
