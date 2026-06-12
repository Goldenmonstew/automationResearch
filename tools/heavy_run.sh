#!/bin/bash
# ============================================================================
# HEAVY-TIER PILOT (T2): 2 lanes x 4 GPUs, one heavy idea each.
#   lane0 = pool 9  (training_stats_early_stopping: ResNet-18/CIFAR-10 +
#                    DistilBERT/SST-2)
#   lane1 = pool 33 (weight_decay_scaling_law: ResNet-18 x 9 subset sizes
#                    x 200 epochs)
#   Config: official stage iters 20/12/12/18, seed 3, drafts 3,
#           num_workers 3 (4 visible GPUs -> NOT clamped to 1, the phase2
#           failure mode), exec.timeout 14400 (4h/node), 15h hard cap.
#   Responses-API priority backend enabled via USE_RESPONSES_API=1 (env) +
#   .use_responses_api sentinel in each copy.
# Usage:  setsid bash heavy_run.sh
# ============================================================================
set -u
SRC=$HOME/automationResearch/AI-Scientist-v2
ROOT=$HOME/automationResearch/heavy
POOL_SRC=$HOME/automationResearch/phase2/phase2_ideas.json
PY=${CONDA_PY:-python}
IDEA_TIMEOUT=54000                       # 15h hard cap per idea
set -a; . "$HOME/.sprint_env"; set +a
: "${OPENAI_API_KEY:?need OPENAI_API_KEY}"
mkdir -p "$ROOT"

LANE_QUEUE_0="9"
LANE_QUEUE_1="33"

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
sub(r'num_workers: \d+', 'num_workers: 3')
sub(r'num_drafts: \d+',  'num_drafts: 3')
sub(r'(\n  timeout: )\d+', r'\g<1>14400')
open(cfg, 'w').write(s)
print('heavy cfg: 20/12/12/18 seed3 workers3 drafts3 exec14400 code/feedback=gpt-5.5')
PYED
}

defuse_cleanup () {   # $1 = launch_scientist_bfts.py path
  $PY - "$1" <<'PYED'
import sys
p = sys.argv[1]
s = open(p).read()
old = 'keywords = ["python", "torch", "mp", "bfts", "experiment"]'
assert old in s, 'cleanup keywords line not found'
s = s.replace(old, 'keywords = []  # heavy pilot: machine-wide kill disabled')
open(p, 'w').write(s)
print('machine-wide cleanup defused')
PYED
}

run_lane () {   # $1 = lane id
  local s=$1
  local lo=$(( s * 4 ))
  local cvd="$lo,$(( lo + 1 )),$(( lo + 2 )),$(( lo + 3 ))"
  local DST=$ROOT/lane$s/AI-Scientist-v2
  rm -rf "$ROOT/lane$s"; mkdir -p "$DST"
  rsync -a --exclude=experiments --exclude=workspaces --exclude=__pycache__ \
        --exclude='*.pyc' --exclude='.git' "$SRC/" "$DST/"
  cp "$POOL_SRC" "$DST/pool.json"
  touch "$DST/.use_responses_api"
  patch_cfg "$DST/bfts_config.yaml" >> "$ROOT/progress.log" 2>&1
  defuse_cleanup "$DST/launch_scientist_bfts.py" >> "$ROOT/progress.log" 2>&1
  sleep $(( s * 300 ))   # stagger lanes 5 min
  local q="LANE_QUEUE_$s"
  for i in ${!q}; do
    echo "=== [heavy lane$s idea$i gpu=$cvd] start $(date +%F_%T) ===" >> "$ROOT/progress.log"
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
    pkill -9 -f "[i]dea_idx $i " 2>/dev/null
    echo "[heavy lane$s idea$i] done rc=$rc $(date +%F_%T)" >> "$ROOT/progress.log"
  done
  echo "=== [heavy lane$s] QUEUE DONE $(date +%F_%T) ===" >> "$ROOT/progress.log"
}

echo "=== HEAVY PILOT START $(date) : 2 lanes x 4gpu, exec 4h, cap 15h ===" >> "$ROOT/progress.log"
for (( s=0; s<2; s++ )); do run_lane "$s" & done
wait
echo "=== HEAVY PILOT ALL DONE $(date) ===" >> "$ROOT/progress.log"
