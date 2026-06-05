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
- treesearch/parallel_agent.py: framework 崩溃修复(worker 异常不再 raise 杀 run)+ GPUManager 尊重 CUDA_VISIBLE_DEVICES + **Mac MPS 跨平台** + plan_and_code 放宽 `if code`
- tools/openalex_search.py(新增): OpenAlex 真引用 drop-in(替 arXiv/S2)
- tools/semantic_scholar.py: `@backoff max_tries` 防 429 卡死
- perform_icbinb_writeup.py: VLM 固定 gpt-4o + `_extract_latex_block` 鲁棒提取
- perform_ideation_temp_free.py: ReAct 解析鲁棒 / vlm.py + bfts_config.yaml: router 模型名
