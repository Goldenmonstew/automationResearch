#!/bin/bash
# Presentation-pass chain (post-registration v3 lineage, facts frozen):
#   honest_rewrite --variant presentation -> grounding audit -> reground
#   (suffix rewrittenv3) -> ensemble (tag rewrittenv3)
# usage: hc_v3.sh <repo_root> <exp_name_grep>
set -a; . $HOME/.sprint_env; set +a
DST=$1; PAT=$2
SP=$HOME/automationResearch/sprint
PY=${CONDA_PY:-python}   # conda env python (set CONDA_PY for your deployment)
export PATH=$HOME/texlive2026/bin/x86_64-linux:$(dirname $(command -v $PY)):$PATH
export PYTHONPATH=$DST
cd $DST
EXPN=$(ls experiments/ | grep "$PAT" | head -1)
[ -n "$EXPN" ] || { echo "V3: no experiment matching $PAT"; exit 1; }
AUDIT=$(ls -t experiments/$EXPN/grounding_*.json 2>/dev/null | head -1)
[ -n "$AUDIT" ] || { echo "V3: no final audit json"; exit 1; }
echo "V3($PAT): exp=$EXPN audit=$AUDIT"
$PY -u $SP/honest_rewrite.py --exp_dir experiments/$EXPN --audit "$AUDIT" \
    --variant presentation || { echo "V3: rewrite failed"; exit 1; }
PDF0=experiments/$EXPN/${EXPN}_rewrittenv3_r0.pdf
[ -f "$PDF0" ] || { echo "V3: no r0 pdf"; exit 1; }
$PY -u $SP/grounding_audit.py --pdf $PDF0 --exp_dir experiments/$EXPN \
    --out $SP/audit_v3_$PAT || exit 1
$PY -u $SP/reground_rewrite.py --exp_dir experiments/$EXPN \
    --audit $SP/audit_v3_$PAT.json --audit_script $SP/grounding_audit.py \
    --suffix rewrittenv3
FINALPDF=$(ls -t experiments/$EXPN/${EXPN}_rewrittenv3_r*.pdf 2>/dev/null | head -1)
[ -n "$FINALPDF" ] || { echo "V3: no final pdf"; exit 1; }
( cd $HOME/automationResearch/AI-Scientist-v2 && PYTHONPATH=$PWD \
  $PY -u $SP/ensemble_review.py --out $SP/reviews --tag rewrittenv3 --pdf $DST/$FINALPDF )
echo "V3 CHAIN COMPLETE ($PAT)"
