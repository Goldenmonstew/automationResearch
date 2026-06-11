#!/bin/bash
# ============================================================================
# Phase 2 批量复现编排:41 个 idea × scaled(gpt5.5 代码大脑, 8h 节点 timeout,
#   stage 40/30/30/40, num_seeds 10, num_drafts 5)。
#
# 并行 SLOTS 路,每路 GPU 分区(靠 parallel_agent.py 的 GPUManager 尊重
# CUDA_VISIBLE_DEVICES 补丁,各路只用自己那几张卡,不互相 double-book),
# 每路串行跑分配给它的 idea。GPU 基本闲(瓶颈是 router LLM 往返),所以
# 并行主要换 router 吞吐 → SLOTS 越大越快但越吃 router 配额。
#
# 用法(launch 前 export key + base url):
#   SLOTS=4 OPENAI_API_KEY=<k> OPENAI_API_BASE=<url> OPENAI_BASE_URL=<url> \
#     setsid bash tools/phase2_run.sh
# ============================================================================
set -u
SLOTS=${SLOTS:-4}                          # 并行路数(8 卡 / SLOTS = 每路卡数)
TOTAL_GPUS=${TOTAL_GPUS:-8}
GPUS_PER=$(( TOTAL_GPUS / SLOTS ))         # 每路 GPU 数 = 该路 num_workers
WRITEUP_MODEL=${WRITEUP_MODEL:-gpt-5.5}    # 不稳可设 gpt-4o(格式更稳)
SRC=$HOME/automationResearch/AI-Scientist-v2
ROOT=$HOME/automationResearch/phase2
POOL=$ROOT/phase2_ideas.json
PY=${CONDA_PY:-python}   # conda env python (set CONDA_PY for your deployment)
: "${OPENAI_API_KEY:?need OPENAI_API_KEY}"
mkdir -p $ROOT
N=$( $PY -c "import json;print(len(json.load(open('$POOL'))))" )

patch_cfg () {   # $1=config路径 $2=workers
  $PY - "$1" "$2" "$WRITEUP_MODEL" <<'PYED'
import re, sys
cfg, workers, wm = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(cfg).read()
s = re.sub(r'(\n  code:\n    model: )\S+',     r'\g<1>gpt-5.5', s)
s = re.sub(r'(\n  feedback:\n    model: )\S+', r'\g<1>gpt-5.5', s)
s = re.sub(r'stage1_max_iters: \d+', 'stage1_max_iters: 40', s)
s = re.sub(r'stage2_max_iters: \d+', 'stage2_max_iters: 30', s)
s = re.sub(r'stage3_max_iters: \d+', 'stage3_max_iters: 30', s)
s = re.sub(r'stage4_max_iters: \d+', 'stage4_max_iters: 40', s)
s = re.sub(r'num_seeds: \d+',   'num_seeds: 10', s)
s = re.sub(r'num_workers: \d+', f'num_workers: {workers}', s)
s = re.sub(r'num_drafts: \d+',  'num_drafts: 5', s)
s = re.sub(r'(\n  timeout: )\d+', r'\g<1>28800', s)   # exec.timeout 8h/节点
open(cfg, 'w').write(s)
print(f'scaled: 40/30/30/40 seed10 workers{workers} 8h, code/feedback=gpt-5.5, writeup={wm}')
PYED
}

run_slot () {    # $1=slot id
  local s=$1
  local lo=$(( s * GPUS_PER ))
  local cvd=$( seq -s, $lo $(( lo + GPUS_PER - 1 )) )
  local DST=$ROOT/slot$s/AI-Scientist-v2
  rm -rf $ROOT/slot$s; mkdir -p $DST
  rsync -a --exclude=experiments --exclude=workspaces --exclude=__pycache__ \
        --exclude='*.pyc' --exclude='.git' $SRC/ $DST/
  cp $POOL $DST/pool.json
  patch_cfg "$DST/bfts_config.yaml" $GPUS_PER >> $ROOT/progress.log
  for (( i=s; i<N; i+=SLOTS )); do
    echo "=== [P2 slot$s idea$i gpu=$cvd] start $(date +%F_%T) ===" >> $ROOT/progress.log
    ( cd $DST && CUDA_VISIBLE_DEVICES=$cvd \
        PATH=$HOME/texlive2026/bin/x86_64-linux:$(dirname $PY):$PATH \
        OPENAI_API_KEY=$OPENAI_API_KEY OPENAI_API_BASE=$OPENAI_API_BASE OPENAI_BASE_URL=$OPENAI_BASE_URL \
        HF_ENDPOINT=https://hf-mirror.com OPENALEX_MAILTO=ai-scientist@example.com \
        HF_HUB_DOWNLOAD_TIMEOUT=30 HF_HUB_ETAG_TIMEOUT=30 HF_HUB_DISABLE_TELEMETRY=1 \
        timeout 259200 $PY -u launch_scientist_bfts.py --load_ideas pool.json --idea_idx $i \
          --add_dataset_ref --writeup-type icbinb \
          --model_writeup "$WRITEUP_MODEL" --model_writeup_small "$WRITEUP_MODEL" \
          --model_citation gpt-4o --model_review gpt-4o --model_agg_plots gpt-4o \
          --num_cite_rounds 10 > $ROOT/idea_${i}.log 2>&1 )
    local pdf=$( find $DST/experiments -name '*reflection*.pdf' 2>/dev/null | wc -l )
    echo "[P2 slot$s idea$i] done $(date +%F_%T) (slot累计PDF=$pdf)" >> $ROOT/progress.log
  done
  echo "=== [P2 slot$s] ALL DONE $(date +%F_%T) ===" >> $ROOT/progress.log
}

echo "=== PHASE2 START $(date) : $N ideas, SLOTS=$SLOTS, ${GPUS_PER}gpu/slot, writeup=$WRITEUP_MODEL ===" >> $ROOT/progress.log
for (( s=0; s<SLOTS; s++ )); do run_slot $s & done
wait
echo "=== PHASE2 ALL DONE $(date) ===" >> $ROOT/progress.log
