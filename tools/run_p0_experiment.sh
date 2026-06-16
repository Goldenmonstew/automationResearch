#!/usr/bin/env bash
# =============================================================================
# P0 experiment runner — compare standard vs critique-enhanced writeup pipeline
#
# Prerequisites:
#   - cwd = AI-Scientist-v2 repo root (with patches applied)
#   - OPENAI_API_KEY and OPENAI_BASE_URL set
#   - tools/ directory accessible (symlink or rsync from automationResearch)
#
# Usage:
#   # Single experiment comparison
#   bash tools/run_p0_experiment.sh experiments/<run>
#
#   # Evidence distillation only (quick test)
#   bash tools/run_p0_experiment.sh experiments/<run> distill_only
#
#   # Critique pipeline only (skip standard for speed)
#   bash tools/run_p0_experiment.sh experiments/<run> critique_only
# =============================================================================
set -uo pipefail

IDEA_DIR="${1:?usage: $0 <idea_dir> [distill_only|critique_only|full]}"
MODE="${2:-full}"
PY="${PY:-python}"
MODEL="${MODEL:-gpt-4o}"
MODEL_CRITIQUE="${MODEL_CRITIQUE:-gpt-5.5}"
TOOLS_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${IDEA_DIR}/p0_experiment_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "== P0 Pipeline Experiment =="
echo "   IDEA_DIR       : $IDEA_DIR"
echo "   MODE           : $MODE"
echo "   MODEL          : $MODEL"
echo "   MODEL_CRITIQUE : $MODEL_CRITIQUE"
echo "   LOG_DIR        : $LOG_DIR"
echo "   TIMESTAMP      : $TIMESTAMP"
echo

# ---------- distill_only: just run evidence distiller and inspect ----------
if [ "$MODE" = "distill_only" ]; then
    echo "[exp] running evidence distiller only ..."
    $PY -u "$TOOLS_DIR/evidence_distiller.py" \
        --exp_dir "$IDEA_DIR" \
        --out "$LOG_DIR/evidence" \
        --model "$MODEL_CRITIQUE" \
        2>&1 | tee "$LOG_DIR/distill.log"
    echo "[exp] done. Check $LOG_DIR/evidence.json and .md"
    exit 0
fi

# ---------- critique_only: skip standard pipeline ----------
if [ "$MODE" = "critique_only" ]; then
    echo "[exp] running critique pipeline only ..."
    $PY -u "$TOOLS_DIR/critique_writeup.py" \
        --idea_dir "$IDEA_DIR" \
        --model_distill "$MODEL_CRITIQUE" \
        --model_writeup "$MODEL" \
        --model_critique "$MODEL_CRITIQUE" \
        --model_review "$MODEL" \
        --out_log "$LOG_DIR/critique_log.json" \
        2>&1 | tee "$LOG_DIR/critique.log"
    echo "[exp] done. Check $LOG_DIR/critique_log.json"
    exit 0
fi

# ---------- full: run both pipelines via comparison harness ----------
echo "[exp] running full comparison (standard + critique) ..."
$PY -u "$TOOLS_DIR/pipeline_comparison.py" \
    --idea_dir "$IDEA_DIR" \
    --model "$MODEL" \
    --model_critique "$MODEL_CRITIQUE" \
    --out "$LOG_DIR/comparison" \
    2>&1 | tee "$LOG_DIR/comparison.log"

echo
echo "[exp] DONE. Results:"
echo "  $LOG_DIR/comparison.json  (machine-readable)"
echo "  $LOG_DIR/comparison.md    (summary table)"
echo "  $LOG_DIR/comparison.log   (full output)"
