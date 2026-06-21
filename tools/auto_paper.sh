#!/bin/bash
# Autonomous paper generation — one command, no human intervention.
#
# Launches an isolated tree-search lane plus a self-healing supervisor daemon
# that (a) auto-recovers the ProcessPoolExecutor futex deadlock (the dominant
# unattended failure), and (b) fires the writeup+review chain on completion.
# Produces a reviewed PDF with zero babysitting.
#
# Usage:
#   auto_paper.sh <ideas_json> <idea_idx> [lane_name] [interval_secs]
# Example:
#   auto_paper.sh ai_scientist/ideas/gpt54pro_ideas.json 3 demo1
#
# Prereqs (this server): ~/.sprint_env or ~/.env_llm_api_key for the router key;
# tools/sprint_supervisor.py + sprint/process_{completed,killed}.sh deployed.

IDEAS=${1:?need ideas json path relative to AI-Scientist-v2 repo (ai_scientist/ideas/x.json)}
IDX=${2:?need idea index}
LANE=${3:-auto_$(date +%Y%m%d_%H%M%S)}
INTERVAL=${4:-600}

H=$HOME
SRC=$H/automationResearch/AI-Scientist-v2
SP=$H/automationResearch/sprint
BASE=$SP/auto
DST=$BASE/$LANE/AI-Scientist-v2

mkdir -p "$BASE/$LANE"

# 1. Router/env. Prefer ~/.sprint_env; fall back to the bare key file.
set -a
[ -f "$H/.sprint_env" ] && . "$H/.sprint_env"
[ -z "${OPENAI_API_KEY:-}" ] && [ -f "$H/.env_llm_api_key" ] && OPENAI_API_KEY=$(cat "$H/.env_llm_api_key")
set +a
# Resolve python AFTER sourcing ~/.sprint_env so CONDA_PY (set there) takes effect.
PY=${CONDA_PY:-python}   # conda env python (set CONDA_PY for your deployment)
# Router base URL + key come from ~/.sprint_env at runtime (not committed).
export OPENAI_API_BASE=${OPENAI_BASE_URL:-${OPENAI_API_BASE:-}}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export DATASET_CACHE_DIR=${DATASET_CACHE_DIR:-$H/automationResearch/.data_cache}
export PATH=$H/texlive2026/bin/x86_64-linux:$(dirname "$PY"):$PATH
if [ -z "${OPENAI_API_KEY:-}" ]; then echo "FATAL: no OPENAI_API_KEY (set ~/.sprint_env or ~/.env_llm_api_key)"; exit 1; fi

# 2. Isolated lane copy — inherits the interpreter dataset-symlink patch, the
#    latex iclr2025_conference.sty fix, and the framework deadlock/parse patches.
echo "[auto_paper] rsync $SRC -> $DST"
rsync -a --exclude experiments --exclude '__pycache__' --exclude '*.bak_*' "$SRC/" "$DST/" || { echo "FATAL: rsync failed"; exit 1; }

# 3. Robust node timeout (8h) so a long node is never mistaken for / killed as a stall.
cfg=$DST/bfts_config.yaml
[ -f "$cfg" ] && sed -i 's/^\([[:space:]]*timeout:\).*/\1 28800/' "$cfg"

# 4. Launch tree search unattended (supervisor owns writeup -> skip here).
cd "$DST" || exit 1
LOG=$BASE/$LANE.tree.log
nohup env OPENAI_API_KEY="$OPENAI_API_KEY" OPENAI_BASE_URL="$OPENAI_BASE_URL" \
  OPENAI_API_BASE="$OPENAI_API_BASE" HF_ENDPOINT="$HF_ENDPOINT" \
  DATASET_CACHE_DIR="$DATASET_CACHE_DIR" PYTHONPATH="$DST" PATH="$PATH" \
  "$PY" -u launch_scientist_bfts.py --load_ideas "$IDEAS" --idea_idx "$IDX" \
  --skip_writeup --skip_review --model_agg_plots gpt-4o \
  > "$LOG" 2>&1 &
TREE_PID=$!
echo "[auto_paper] tree search PID $TREE_PID -> $LOG"

# 5. Self-healing supervisor daemon: auto-recovers deadlocks + fires writeup chain.
SUPLOG=$BASE/$LANE.supervisor.log
nohup env OPENAI_API_KEY="$OPENAI_API_KEY" OPENAI_BASE_URL="$OPENAI_BASE_URL" PATH="$PATH" \
  "$PY" -u "$H/automationResearch/tools/sprint_supervisor.py" \
  --roots "$DST" --sprint_dir "$SP" --interval "$INTERVAL" \
  > "$SUPLOG" 2>&1 &
SUP_PID=$!
echo "[auto_paper] supervisor PID $SUP_PID -> $SUPLOG"

echo "[auto_paper] '$LANE' running unattended."
echo "  tree log:       $LOG"
echo "  supervisor log: $SUPLOG"
echo "  paper + review will land under: $DST/experiments/ and $SP/reviews/"
echo "  stop: kill $TREE_PID $SUP_PID"
