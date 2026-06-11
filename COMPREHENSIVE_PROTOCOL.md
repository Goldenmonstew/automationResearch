# 综合复现协议 v3 —— Scaled to Beat + 零造假

> 目标:无 Claude(代码大脑 **gpt5.5**,已实测 62% good + 自发用真实数据集)、token 不限、时间/资源充足的条件下,**在两条线上都超过官方 Sakana showcase**:
>
> - **G1 超过官方(同标尺)**:在**同一套评审仪器**下不弱于/优于官方 showcase。⚠️ 修正(2026-06-11):官方 6.33 是人类 workshop 尺,与任何 AI 评审尺不可比;G1 的合法形态 = 绝对尺(5 票 ensemble)宣称同档 + **成对盲评锦标赛胜率 + 置信区间**宣称优越(见 PREREGISTRATION.md)。最终结果:胜率 84.4% [0.756, 0.905]。
> - **G2 真实零造假(他们做不到的差异化)**:我们的 showcase 论文**逐条声明都能追溯到实验日志**,通过严格 grounding 审计;而实测官方 3 篇全 hallucinate headline 结果。
>
> 约束:单 8×H20 箱,run 基本串行,每个 scaled run ~6–15h;整轮 ≈ 1–2 周墙钟(token/算力不是瓶颈)。

---

## 核心张力(必须设计进去)

**造假能虚高分数,删假会掉分** → G1 和 G2 天然冲突,**除非实验本身有真发现**。所以本协议的命门是:**用真实数据 + 放大算力 + 精选 idea,让实验产出【真的、够强的】结果,这样诚实写的论文同时也是高分论文。** 实验扎实是 G1×G2 同时成立的唯一支点;无脑堆节点修不了造假,垃圾实验也无法既诚实又高分。

---

## 阶段 0:修"假失败"(~半天,不抢 GPU,可现在起)

1. **plot 聚合 bug**(硬编码 dataset key→空图,真实数据必触发):改成遍历所有 key + 兼容两种 metrics 格式。
2. **真引用**:确认 OpenAlex drop-in 生效(非 `(?)`);可选申请真 S2 key。
3. **LaTeX 缺图 / gpt5.5 ```latex 不稳**:`_extract_latex_block` patch 已在;writeup 不稳则该步回退 gpt-4o。
- **验收**:拿现成实验目录跑一遍 writeup,图不空、引用真、PDF 完整。

## 阶段 1:idea 大池子(~半天)

模仿官方"43 选 3",**池子要大**:
- gpt5.5 ideation 批量生成 **~50 个** ICBINB 负面结果/失败模式 idea。
- 自动 + 人工 curate 到 **~20–24 个**跑(门槛:绑真实小数据集 MNIST/CIFAR/digits 等、效应能显现有区分度、T0–T1 塞得进 1h、避开 make_blobs 退化)。
- 产出 `ideas/scaled_pool.json`,措辞钉死真实数据集 + 要求真算目标指标。

## 阶段 2:Scaled 批量满预算(主体,~1–2 周)

### 2A. 主轴 — gpt5.5 跑全部 ~20–24 idea(放大到官方曲线右端外)
- 模型:code/feedback **gpt5.5**;writeup gpt5.5(不稳回退 gpt-4o);citation/agg gpt-4o。
- **放大预算(关键,探官方没测过的 30+ 节点无人区)**:`num_workers` **8**(零限流、GPU 全闲)、stage iters **30/20/20/30**(官方 20/12/12/18 的 ~1.6×)、`num_seeds` **5**、`num_drafts` **5**(更多样初始草稿)。目标**有效 good 节点 100+**(官方 Fig3c 最多测到 ~30)。
- 串行 setsid + loop 监控;每出一篇推送 + 拉 PDF。

### 2B. 对照轴 — 代码大脑公平比 + 方差
- 1–2 个 anchor idea(真实数据)× {gpt5.5, deepseek, gpt-4o, gemini} × 2 run-seed,同 scaled 预算。回答 gpt5.5 vs 其它的 good 率/扎实度/分数差 + run 间方差。

## 阶段 2.5:★ Hallucination 验证 Gate(G2 核心,官方没有)

每篇 writeup 产出后,**强制过 grounding 审计**(做成 pipeline 一关,不是事后):
1. **抽取**论文里每个经验声明:数字、表、图、数据集名、对比结论、headline claim。
2. **逐条核对** `experiment_data.npy` / stage summaries / 日志,判定:grounded / 无支撑 / 与日志矛盾。
3. **有无支撑或矛盾 → 自动改写 writeup**(删掉或改正该 claim)→ 重编译 → 重审,**循环到全 grounded**(上限 N 轮;实在删不掉的标"未验证"不进 showcase)。
4. 产出 **grounding 证书**:每条 claim → 对应日志行的 provenance 表。
- **效果**:论文要么诚实地强(实验真有结果),要么诚实地弱(没结果就不吹)——绝不造假。这是官方那 3 篇过不了的关。

## 阶段 3:严谨评测(对齐甚至超过官方)

- **ensemble reviewer**:每篇 **5 次独立评审(gpt5.5×3 + gpt-4o×2)+ meta-review** 汇总,报 mean±std + 分歧。对齐官方 o4-mini×5。
- **多 seed 方差**:2B run-seed + 节点级 seed5 → Overall 的 mean±std。
- **校准集**:Attention + 边缘 ICLR + 官方 3 篇 showcase 全放进同一把尺。

## 阶段 4:survivorship 精选 + 双口径

- 从 ~20+ 篇里,**必须先过阶段 2.5 零造假 gate**,再按 ensemble 分 + 切题/格式门槛,**精选 3 篇** showcase。
- **两口径都报**:① 精选最佳 3 篇 ② 全部 ~20 篇 Overall 分布(mean±std);明写"**跑了 N → 选了 3**"对标官方 43→3。

## 阶段 5:★ 与官方头对头(同标尺,证明两条线都赢)

- 把**官方 3 篇 showcase 论文**也喂进**我们的 ensemble reviewer + grounding 审计**。
- 出对照表:
  - **G1**(2026-06-11 修正口径):绝对尺只报"同档";优越性 = 成对盲评胜率 + CI(实测 84.4% [0.756, 0.905],逐篇 CI 下界全 >50%)。6.33 为人类尺,不进任何对比。
  - **G2**:我们的 grounding 通过率(实测 100%,15+ 篇 ≥95% grounded,审计器 recall 100%/FPR 5%)vs 官方 0/3(无日志不可受声明级审计,文本级 14-32% 内部矛盾)。
- 结论形态:"**盲评显著优于官方 + 唯一能出具声明级溯源证书的端到端系统;两层审计深度不对称如实分列**"。

## 交付物

- `ideas/scaled_pool.json`(~24 真实数据 idea)
- 每篇:PDF + 生成透明度附录 + **grounding 证书** + ensemble 评审 + token/算力账本
- survivorship 报告(N→3,双口径)+ 代码大脑对照表 + 校准表 + **头对头对照表(G1/G2)**
- `BENCHMARK_REPORT.md` 新章

## 诚实的天花板(免得盲投)

1. **30 节点以上是外推**——可能续涨也可能收益递减;填这个空白本身就是超过官方的贡献,但别假设线性。
2. **算力修不了造假**:G2 完全靠阶段 2.5 的验证 gate,不靠堆节点。
3. **auto-reviewer 分高 ≠ 真同行评审接收**:官方 headline 投的是最宽容 workshop;要在那点超过得真投 venue(涉 AI 署名伦理),属另一种工程,本协议不含。
4. **G1×G2 同时成立的前提是实验真有结果**——所以阶段 1(真实数据+可显现 idea)和阶段 2(放大算力出真信号)是上限决定项,比阶段 5 的对比更重要。

## 规模旋钮 —— ★已选「更猛」档

| 旋钮 | 当前设 | **★更猛(已选)** | 更省 |
|---|---|---|---|
| idea 池子 | 生成 50 跑 24 | **生成 100 跑 40** | 生成 30 跑 12 |
| stage iters | 30/20/20/30 | **40/30/30/40** | 20/12/12/18(官方原值) |
| num_seeds | 5 | **10** | 3 |
| 对照轴 2B | 1–2 anchor×4 模型 | **全模型矩阵** | 跳过 |
| showcase | 3 篇 | **5 篇** | 1 篇 |

> 已定更猛:**生成 100 idea 跑 40、stage 40/30/30/40、seed 10、全模型矩阵、选 5 篇**。墙钟约 2–3 周,token/算力不限。

## ★ 2026-06 关键更新(实跑后定稿)

- **idea 池子定稿**:ideation 出 76 → **5 个异构 LLM 评委 ensemble + 对抗复核**去重打分 → 1h-aware 共识 35 + 8h 后捞回重负载 6 → **最终 Phase 2 池 = 41 个**(`selection/phase2_ideas.json`)。全程留痕:`selection/{MASTER_LEDGER,SHORTLIST_8H,FINAL_POOL_8H}.md` + `ledger.json`;选择 pipeline:`tools/{select_ideas,merge_selection,recurate_8h}.py`(可复用)。
- **exec.timeout 3600(1h)→ 28800(8h)**:用户定"破 1h 无所谓、效果优先、token/时间/GPU 管够"。调高它**同时**解除框架 prompt 的"做轻量实验"引导(`complete within {timeout}` 自动派生)→ 实验可真训练到收敛、用更大网络/更大数据,直接服务"真实发现→不空洞"。**重节点拖长但轻节点不受影响。**
- **不简化重负载 idea**:5(DistilBERT)、23(Tiny-ImageNet)及 6 个 add-back(架构/梯度对齐类)**保留全量**,用 8h 跑。
- **degenerate 仍裁**(29/64/71/26):效应本身 < seed 方差,是实验设计问题,调 timeout 救不了。
