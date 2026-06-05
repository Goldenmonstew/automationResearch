# automationResearch — The AI Scientist 研究/复现工作区

研究并运行 Sakana AI《Towards end-to-end automation of AI research》(Nature 651, 2026)
的工作区产物。官方三个 clone(AI-Scientist / v2 / workshop)与版权 Nature PDF 未纳入本仓库
(见 `.gitignore`);对 v2 的适配改动以 diff 形式存于 `patches/`。

## 内容
- **METHODOLOGY_ANALYSIS.md** — 方法论分析(idea GPU 分级 / 树搜索决策机制 / token / 三层超时 / 跨平台)
- **BENCHMARK_REPORT.md** — v1/v2 × 多模型 benchmark + 质量 gap 归因(两边实物对比)
- **STUDY_NOTES.md** — 源码级研究报告 / **EXPERIMENT_REPORT.md** — 一次完整复现实验
- **tools/** — `gen_gen_appendix.py`(生成透明度附录)/ `measure_tokens.py` / `setup_ai_scientist.sh`(一键复现)
- **papers/** — 满预算生成的 AI 论文 PDF + 各自的树搜索生成透明度附录
- **patches/** — AI-Scientist-v2 适配 patch(`git apply`)
