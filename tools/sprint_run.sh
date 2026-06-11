#!/bin/bash
# ============================================================================
# 48h SPRINT: official-budget batch runs over the curated idea pool.
#   Config per run: stage 20/12/12/18, num_seeds 3, num_drafts 3,
#   num_workers 2, exec.timeout 3600 (the empirically validated full-budget
#   setup: ~4.5-7h wall per idea incl. writeup+review).
#   4 lanes x 2 GPUs, each lane runs its consensus-ranked queue serially.
#   Per-idea hard cap 10h; no new idea starts after T+29h (leaves eval tail).
#
# Usage (env in ~/.sprint_env: OPENAI_API_KEY/OPENAI_API_BASE/OPENAI_BASE_URL/
#        HF_ENDPOINT/OPENALEX_MAILTO):
#   setsid bash sprint_run.sh
# ============================================================================
set -u
SRC=$HOME/automationResearch/AI-Scientist-v2
ROOT=$HOME/automationResearch/sprint
POOL_SRC=$HOME/automationResearch/phase2/phase2_ideas.json
PY=${CONDA_PY:-python}   # conda env python (set CONDA_PY for your deployment)
LANES=4
IDEA_TIMEOUT=36000                         # 10h hard cap per idea
CUTOFF_TS=$(( $(date +%s) + 29*3600 ))     # no new idea after T+29h
set -a; . "$HOME/.sprint_env"; set +a
: "${OPENAI_API_KEY:?need OPENAI_API_KEY}"
mkdir -p "$ROOT"

# Interleaved consensus top-20 (pool_idx into phase2_ideas.json).
# pool 9 swapped out for pool 7: ledger requires stripping its DistilBERT/SST-2
# leg, pool 7 is the designated first substitute (same score tier, light).
LANE_QUEUE_0="0 5 7 15 21"
LANE_QUEUE_1="1 3 11 16 17"
LANE_QUEUE_2="2 6 12 18 24"
LANE_QUEUE_3="4 8 13 19 25"

patch_cfg () {   # $1 = bfts_config.yaml path
  $PY - "$1" <<'PYED'
import re, sys
cfg = sys.argv[1]
s = open(cfg).read()
def sub(p, r):
    global s
    s, n = re.subn(p, r, s)
    assert n >= 1, f"pattern not found: {p}"
sub(r'(\n  code:\n    model: )\S+',     r'\g<1>gpt-5.5')
sub(r'(\n  feedback:\n    model: )\S+', r'\g<1>gpt-5.5')
sub(r'stage1_max_iters: \d+', 'stage1_max_iters: 20')
sub(r'stage2_max_iters: \d+', 'stage2_max_iters: 12')
sub(r'stage3_max_iters: \d+', 'stage3_max_iters: 12')
sub(r'stage4_max_iters: \d+', 'stage4_max_iters: 18')
sub(r'num_seeds: \d+',   'num_seeds: 3')
sub(r'num_workers: \d+', 'num_workers: 2')
sub(r'num_drafts: \d+',  'num_drafts: 3')
sub(r'(\n  timeout: )\d+', r'\g<1>3600')
open(cfg, 'w').write(s)
print('sprint cfg: 20/12/12/18 seed3 workers2 drafts3 exec3600 code/feedback=gpt-5.5')
PYED
}

defuse_cleanup () {   # $1 = launch_scientist_bfts.py path
  # The end-of-run cleanup kills EVERY process on the machine whose cmdline
  # contains "python"/"torch"/... With parallel lanes the first finisher would
  # murder all other lanes. Empty the keyword list (child cleanup stays).
  $PY - "$1" <<'PYED'
import sys
p = sys.argv[1]
s = open(p).read()
old = 'keywords = ["python", "torch", "mp", "bfts", "experiment"]'
assert old in s, 'cleanup keywords line not found'
s = s.replace(old, 'keywords = []  # sprint: machine-wide kill disabled (parallel lanes)')
open(p, 'w').write(s)
print('machine-wide cleanup defused')
PYED
}

run_lane () {   # $1 = lane id
  local s=$1
  local lo=$(( s * 2 ))
  local cvd="$lo,$(( lo + 1 ))"
  local DST=$ROOT/lane$s/AI-Scientist-v2
  rm -rf "$ROOT/lane$s"; mkdir -p "$DST"
  rsync -a --exclude=experiments --exclude=workspaces --exclude=__pycache__ \
        --exclude='*.pyc' --exclude='.git' "$SRC/" "$DST/"
  cp "$POOL_SRC" "$DST/pool.json"
  patch_cfg "$DST/bfts_config.yaml" >> "$ROOT/progress.log" 2>&1
  defuse_cleanup "$DST/launch_scientist_bfts.py" >> "$ROOT/progress.log" 2>&1
  sleep $(( s * 600 ))   # stagger lanes 10 min apart (desync writeup bursts)
  local q="LANE_QUEUE_$s"
  for i in ${!q}; do
    if [ "$(date +%s)" -ge "$CUTOFF_TS" ]; then
      echo "[lane$s] cutoff reached, skip idea$i $(date +%F_%T)" >> "$ROOT/progress.log"
      break
    fi
    echo "=== [lane$s idea$i gpu=$cvd] start $(date +%F_%T) ===" >> "$ROOT/progress.log"
    ( cd "$DST" && CUDA_VISIBLE_DEVICES=$cvd \
        PATH=$HOME/texlive2026/bin/x86_64-linux:$(dirname $PY):$PATH \
        HF_HUB_DOWNLOAD_TIMEOUT=30 HF_HUB_ETAG_TIMEOUT=30 HF_HUB_DISABLE_TELEMETRY=1 \
        timeout --signal=TERM -k 120 $IDEA_TIMEOUT \
        $PY -u launch_scientist_bfts.py --load_ideas pool.json --idea_idx "$i" \
          --add_dataset_ref --writeup-type icbinb \
          --model_writeup gpt-4o --model_writeup_small gpt-4o \
          --model_citation gpt-4o --model_review gpt-4o --model_agg_plots gpt-4o \
          --num_cite_rounds 5 > "$ROOT/idea_${i}.log" 2>&1 )
    local rc=$?
    # reap any stragglers of THIS idea only (idea_idx is unique across lanes)
    pkill -9 -f "[i]dea_idx $i " 2>/dev/null
    local pdf; pdf=$( find "$DST/experiments" -name '*reflection*.pdf' 2>/dev/null | wc -l )
    local rev; rev=$( find "$DST/experiments" -name 'review_text.txt' 2>/dev/null | wc -l )
    echo "[lane$s idea$i] done rc=$rc $(date +%F_%T) (lane cumulative: reflectionPDF=$pdf review=$rev)" >> "$ROOT/progress.log"
  done
  echo "=== [lane$s] QUEUE DONE $(date +%F_%T) ===" >> "$ROOT/progress.log"
}

echo "=== SPRINT START $(date) : ${LANES} lanes x 2gpu, official budget, new-idea cutoff +29h ===" >> "$ROOT/progress.log"
for (( s=0; s<LANES; s++ )); do run_lane "$s" & done
wait
echo "=== SPRINT ALL DONE $(date) ===" >> "$ROOT/progress.log"
