#!/usr/bin/env bash
# =============================================================================
# setup_ai_scientist.sh — 一键复现环境(回答"重头再来要不要手动重打 patch")
#
# 用法:
#   PY=/path/to/python V2_DIR=./AI-Scientist-v2 bash tools/setup_ai_scientist.sh
#
# 做三件事:
#   1) 装这套适配漏装/必需的 Python 依赖
#   2) 写一份可 source 的 router/镜像 env 模板(key 不写进文件,运行时 export)
#   3) 验证关键适配 patch 是否都在 repo 里(部署到未打补丁的官方 clone 会逐条报缺)
#
# 设计:幂等、只读式验证、绝不把 API key 落盘。CUDA/Linux 与 Mac/MPS 通用。
# =============================================================================
set -uo pipefail

PY="${PY:-python3}"
V2_DIR="${V2_DIR:-$(cd "$(dirname "$0")/.." && pwd)/AI-Scientist-v2}"
ROUTER_BASE="${ROUTER_BASE:-https://<your-router-host>/v1}"

echo "== AI Scientist v2 一键 setup =="
echo "   PYTHON : $PY"
echo "   V2_DIR : $V2_DIR"
echo

if [ ! -d "$V2_DIR" ]; then
  echo "[X] 找不到 V2_DIR=$V2_DIR;先把仓库放到位再跑。"; exit 1
fi
if ! "$PY" -c 'import sys' 2>/dev/null; then
  echo "[X] PYTHON=$PY 不可用。"; exit 1
fi

# ---------------------------------------------------------------------------
# 1) 依赖(含这套适配里反复踩的漏装:psutil 清理阶段、scikit-learn 大量 idea import)
# ---------------------------------------------------------------------------
echo "== [1/3] 安装/校验 Python 依赖 =="
DEPS=(openai anthropic backoff requests tiktoken psutil scikit-learn pyyaml \
      matplotlib numpy pandas rich coolname humanize dataclasses-json)
"$PY" -m pip install --quiet --upgrade "${DEPS[@]}" \
  && echo "   [OK] 依赖就绪(${#DEPS[@]} 个)" \
  || echo "   [!] 依赖安装有报错,检查上面的 pip 输出"
# torch 不在此装:CUDA 机用 conda 装对应 cu 版;Mac 装 CPU/MPS 版(pip install torch)
echo "   [i] torch 请按平台单独装:CUDA 机用 conda 对应 cuXXX;Mac 用 'pip install torch'(自带 MPS)"
echo

# ---------------------------------------------------------------------------
# 2) env 模板(key 不落盘)
# ---------------------------------------------------------------------------
echo "== [2/3] 写 router/镜像 env 模板(不含 key)=="
ENV_FILE="$(cd "$V2_DIR/.." && pwd)/.env.router"
cat > "$ENV_FILE" <<EOF
# source 本文件后再 export 你的 key:  export OPENAI_API_KEY=<router-key>
# litellm/aider 读 OPENAI_API_BASE;openai SDK 读 OPENAI_BASE_URL —— 两个都设
export OPENAI_API_BASE=$ROUTER_BASE
export OPENAI_BASE_URL=$ROUTER_BASE
export HF_ENDPOINT=https://hf-mirror.com          # HuggingFace 直连不通,走镜像
export OPENALEX_MAILTO=you@example.com            # OpenAlex 礼貌头(免 key 真引用)
# export PATH=\$HOME/texlive2026/bin/x86_64-linux:\$PATH   # writeup 编译 PDF 需 LaTeX
# export OPENAI_API_KEY=<router-key>              # ← 运行时再填,别写死
EOF
echo "   [OK] 写到 $ENV_FILE — 用法:source .env.router && export OPENAI_API_KEY=<key>"
echo

# ---------------------------------------------------------------------------
# 3) 验证关键适配 patch 是否齐全(逐条 grep,部署到未打补丁的 clone 会报缺)
# ---------------------------------------------------------------------------
echo "== [3/3] 验证适配 patch 是否齐全 =="
miss=0
check() {  # check <说明> <文件> <grep正则>
  if grep -qE "$3" "$V2_DIR/$2" 2>/dev/null; then
    printf "   [OK] %s\n" "$1"
  else
    printf "   [X] 缺: %s  (%s)\n" "$1" "$2"; miss=$((miss+1))
  fi
}
check "router 模型名规范化 _route_model (llm)"        "ai_scientist/llm.py"                          "_route_model"
check "gemini/else 路由到 router + content or '' (llm)" "ai_scientist/llm.py"                          "_route_model|content or"
check "MAX_NUM_TOKENS 调大 (llm)"                       "ai_scientist/llm.py"                          "MAX_NUM_TOKENS\s*=\s*(16384|32768)"
check "backend _route_model"                            "ai_scientist/treesearch/backend/__init__.py"  "_route_model"
# 注:VLM 路径始终用 gpt-4o(router 原生),无需 vlm.py 改 _route_model;
# 真正的 VLM 适配是 writeup 里固定 create_vlm_client("gpt-4o")(见下条)。
check "VLM 固定 gpt-4o (writeup)"                        "ai_scientist/perform_icbinb_writeup.py"        "create_vlm_client\(.gpt-4o"
check "OpenAlex 真引用 drop-in"                          "ai_scientist/tools/openalex_search.py"        "search_for_papers"
check "writeup 用 OpenAlex + _extract_latex_block"      "ai_scientist/perform_icbinb_writeup.py"        "openalex_search|_extract_latex_block"
check "S2 backoff max_tries (防 429 卡死)"              "ai_scientist/tools/semantic_scholar.py"        "max_tries"
check "framework crash fix (worker 异常不再 raise 杀 run)" "ai_scientist/treesearch/parallel_agent.py"  "isinstance\(e, BrokenProcessPool\)|记录.*跳过|continue"
check "GPUManager 尊重 CUDA_VISIBLE_DEVICES 分区"       "ai_scientist/treesearch/parallel_agent.py"    "Respect CUDA_VISIBLE_DEVICES|only manage those ids"
check "跨平台:MPS 探测 (Mac)"                           "ai_scientist/treesearch/parallel_agent.py"    "def mps_available"
check "跨平台:device 模板含 mps 分支"                   "ai_scientist/treesearch/parallel_agent.py"    "torch.backends, 'mps'"
check "ReAct 解析鲁棒 (ideation, reasoning 模型)"       "ai_scientist/perform_ideation_temp_free.py"   "ACTION|json|reflection"
check "plan_and_code 放宽 if code (reasoning code-only)" "ai_scientist/treesearch/parallel_agent.py"    "if code"

echo
if [ "$miss" -eq 0 ]; then
  echo "[done] 全部 patch 就位 —— 这就是'重头再来 = 一条命令'。"
else
  echo "[!] 有 $miss 项缺失(多半是部署到了未打补丁的官方 clone)。"
  echo "  对照 patches/README.md 补齐。"
fi
echo
echo "== 跑实验(先 source env + 填 key)=="
echo "  source .env.router && export OPENAI_API_KEY=<router-key>"
echo "  cd $V2_DIR && \$PY launch_scientist_bfts.py --load_ideas ai_scientist/ideas/<x>.json --idea_idx 0 ..."
echo "  (Mac:不设 CUDA_VISIBLE_DEVICES,自动走 MPS;bfts_config.yaml 把 num_workers 调到 2-3)"
