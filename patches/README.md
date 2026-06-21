# AI-Scientist-v2 适配 patch

针对官方 `SakanaAI/AI-Scientist-v2`(本仓库未纳入)做的适配,使其在自建 OpenAI 兼容 router +
多 router 模型(deepseek/gemini/gpt-5.5/gpt-4o 等)上跑通,并新增 Mac MPS 跨平台、OpenAlex 真引用、
framework 崩溃修复等。详见 ../METHODOLOGY_ANALYSIS.md。

## 应用
```
git clone https://github.com/SakanaAI/AI-Scientist-v2.git
cd AI-Scientist-v2 && git apply ../patches/AI-Scientist-v2.patch
```

## 覆盖
- llm.py: `_route_model` 模型名规范化 + router 路由 + MAX_NUM_TOKENS 16384
- treesearch/backend/__init__.py: `_route_model`
- treesearch/parallel_agent.py: framework 崩溃修复(worker 异常不再 raise 杀 run)+ GPUManager 尊重 CUDA_VISIBLE_DEVICES + **Mac MPS 跨平台** + plan_and_code 放宽 `if code` + **去除实验"轻量化"引导**:把 code 模型的运行时提示从「必须在 1h 内完成」改为「软算力预算 + 鼓励用真实数据集和足够规模、避免指标饱和(~100% acc)导致目标现象观测不到」(解除框架对玩具实验的诱导) + **worker 大结果 stdout 截断**:返回结果前把 `_term_out`/`parse_term_out`/`plot_term_out` 超 20KB 的部分截断(保留首尾),消除巨型结果(如多 run 超参扫描把全部 metric 堆进 stdout → 数百 KB)经 multiprocessing 管道回传时 `future.result()` 卡死的"大对象管道死锁"(metric 单独存 `result_data["metric"]`,截断无损)
- treesearch/interpreter.py: **输出感知的超时策略**——原逻辑纯看墙钟超时即 SIGINT(作者自留 `# TODO`);改为子进程仍在产出 stdout/stderr(经 `result_outq.qsize()` 增长判定,如训练还在刷 loss)时宽限续跑,只有「超时且静默 ≥120s」或「超过硬上限 ≈2× timeout」才杀;`qsize()` 不可用平台(macOS)自动退化为原始硬超时行为
- bfts_config.yaml: `exec.timeout` 3600 → 7200(软预算,给足够规模实验留出现象浮现的空间;配合 interpreter.py 的产出宽限)
- tools/openalex_search.py(新增): OpenAlex 真引用 drop-in(替 arXiv/S2)
- tools/semantic_scholar.py: `@backoff max_tries` 防 429 卡死
- perform_icbinb_writeup.py: VLM 固定 gpt-4o + `_extract_latex_block` 鲁棒提取 + **`filter_experiment_summaries` 对部分-run summary 健壮**(崩溃/死锁中断的 run 只有部分 stage,baseline/research/ablation summary 是空 list 非 dict;加 isinstance 守卫优雅跳过,使离线恢复链对部分 run 不再崩) + **写作机械修复 P0-P4**(胜出 tex 在主目录重编而非 copy 旧 PDF 留 `??`/破 cite;writeup 前校验 figures 非空否则补产图;`invalid_figs` 确定性 fuzzy-map/删除而非靠 LLM 自觉;reflection 以 LaTeX 编译错误为硬退出门;`n_writeup_reflections` 3→4)
- perform_ideation_temp_free.py: ReAct 解析鲁棒 / vlm.py + bfts_config.yaml: router 模型名
