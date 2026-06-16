#!/usr/bin/env python
"""P2: Progress Quality Filter (PQF) for tree search stage transitions.

ICR-inspired quality gate: before allowing progression from one tree search
stage to the next, evaluate whether the current stage output meets quality
thresholds.  Kill bad branches early to save compute.

Can be used:
1. As a standalone evaluator (CLI): check if a stage's output is good enough
2. As an importable module: call from agent_manager.py at stage boundaries

Quality dimensions checked:
  - buggy_rate:     fraction of nodes that failed (too high → stage didn't work)
  - best_metric:    best node's metric value (too low → no meaningful progress)
  - diversity:      number of distinct approaches tried (too low → stuck in local optimum)
  - code_quality:   LLM assessment of best node's code (optional, expensive)

Usage (standalone):
  python tools/pqf_stage_gate.py --journal_path <stage_journal.json> \
      [--stage_number 1] [--threshold strict|normal|lenient]
"""
import argparse
import json
import os
import os.path as osp
import sys
import time

# ---------------------------------------------------------------------------
# Quality thresholds per stage (tuned for AI Scientist v2)
# ---------------------------------------------------------------------------

# Each stage has different expectations:
# Stage 1 (draft): just needs ONE working implementation
# Stage 2 (baseline): needs measurable metric, stable training
# Stage 3 (creative): needs improvement over baseline
# Stage 4 (ablation): needs controlled comparisons

THRESHOLDS = {
    "strict": {
        1: {"max_buggy_rate": 0.90, "min_good_nodes": 1, "min_diversity": 1},
        2: {"max_buggy_rate": 0.70, "min_good_nodes": 2, "min_diversity": 2},
        3: {"max_buggy_rate": 0.60, "min_good_nodes": 3, "min_diversity": 2},
        4: {"max_buggy_rate": 0.70, "min_good_nodes": 2, "min_diversity": 2},
    },
    "normal": {
        1: {"max_buggy_rate": 0.95, "min_good_nodes": 1, "min_diversity": 1},
        2: {"max_buggy_rate": 0.80, "min_good_nodes": 1, "min_diversity": 1},
        3: {"max_buggy_rate": 0.75, "min_good_nodes": 2, "min_diversity": 1},
        4: {"max_buggy_rate": 0.80, "min_good_nodes": 1, "min_diversity": 1},
    },
    "lenient": {
        1: {"max_buggy_rate": 1.0, "min_good_nodes": 1, "min_diversity": 1},
        2: {"max_buggy_rate": 0.90, "min_good_nodes": 1, "min_diversity": 1},
        3: {"max_buggy_rate": 0.85, "min_good_nodes": 1, "min_diversity": 1},
        4: {"max_buggy_rate": 0.90, "min_good_nodes": 1, "min_diversity": 1},
    },
}


# ---------------------------------------------------------------------------
# Quality evaluation
# ---------------------------------------------------------------------------

def evaluate_stage_quality(journal_data, stage_number, threshold_level="normal"):
    """Evaluate a completed stage's quality.

    Args:
        journal_data: parsed journal JSON (with "nodes" list)
        stage_number: 1-4
        threshold_level: "strict", "normal", or "lenient"

    Returns:
        dict with: passed (bool), score (0-100), dimensions (per-check details),
                   recommendation ("proceed" | "retry" | "abort")
    """
    thresholds = THRESHOLDS[threshold_level].get(stage_number,
                                                  THRESHOLDS[threshold_level][1])
    nodes = journal_data.get("nodes", [])
    if not nodes:
        return {"passed": False, "score": 0,
                "recommendation": "abort",
                "reason": "no nodes in journal"}

    total = len(nodes)
    buggy = sum(1 for n in nodes if n.get("is_buggy", False))
    good = total - buggy
    buggy_rate = buggy / total if total > 0 else 1.0

    # Diversity: count distinct plan hashes (or unique first 100 chars of plan)
    plans = set()
    for n in nodes:
        plan = str(n.get("plan", ""))[:100]
        if plan:
            plans.add(plan)
    diversity = len(plans)

    # Best metric extraction
    best_metric = None
    best_metric_raw = None
    for n in nodes:
        if n.get("is_buggy"):
            continue
        m = n.get("metric")
        if m is None:
            continue
        if isinstance(m, dict):
            val = m.get("value")
            if isinstance(val, dict):
                nums = [v for v in val.values() if isinstance(v, (int, float))]
                val = max(nums) if nums else None
            elif not isinstance(val, (int, float)):
                val = None
        elif isinstance(m, (int, float)):
            val = m
        else:
            val = None
        if val is not None and isinstance(val, (int, float)):
            if best_metric is None or val > best_metric:
                best_metric = val
                best_metric_raw = m

    # Score each dimension (0-25 each, total 0-100)
    dimensions = {}

    # 1. Buggy rate (0-25)
    br_pass = buggy_rate <= thresholds["max_buggy_rate"]
    br_score = max(0, 25 * (1 - buggy_rate / max(thresholds["max_buggy_rate"], 0.01)))
    dimensions["buggy_rate"] = {
        "value": round(buggy_rate, 3),
        "threshold": thresholds["max_buggy_rate"],
        "passed": br_pass,
        "score": round(br_score, 1),
    }

    # 2. Good nodes count (0-25)
    gn_pass = good >= thresholds["min_good_nodes"]
    gn_score = min(25, 25 * good / max(thresholds["min_good_nodes"], 1))
    dimensions["good_nodes"] = {
        "value": good,
        "threshold": thresholds["min_good_nodes"],
        "passed": gn_pass,
        "score": round(gn_score, 1),
    }

    # 3. Diversity (0-25)
    div_pass = diversity >= thresholds["min_diversity"]
    div_score = min(25, 25 * diversity / max(thresholds["min_diversity"], 1))
    dimensions["diversity"] = {
        "value": diversity,
        "threshold": thresholds["min_diversity"],
        "passed": div_pass,
        "score": round(div_score, 1),
    }

    # 4. Metric quality (0-25) — harder to gate without knowing the task
    # Give full score if metric exists, half if not
    if best_metric is not None:
        met_score = 25.0
        met_pass = True
    elif good > 0:
        met_score = 12.5
        met_pass = True
    else:
        met_score = 0
        met_pass = False
    dimensions["metric_quality"] = {
        "value": best_metric,
        "raw": str(best_metric_raw)[:200] if best_metric_raw else None,
        "passed": met_pass,
        "score": round(met_score, 1),
    }

    total_score = sum(d["score"] for d in dimensions.values())
    all_passed = all(d["passed"] for d in dimensions.values())

    if all_passed and total_score >= 50:
        recommendation = "proceed"
    elif total_score >= 30:
        recommendation = "retry"
    else:
        recommendation = "abort"

    return {
        "passed": all_passed,
        "score": round(total_score, 1),
        "recommendation": recommendation,
        "stage_number": stage_number,
        "threshold_level": threshold_level,
        "total_nodes": total,
        "buggy_nodes": buggy,
        "good_nodes": good,
        "best_metric": best_metric,
        "dimensions": dimensions,
    }


# ---------------------------------------------------------------------------
# Hook for agent_manager integration
# ---------------------------------------------------------------------------

def check_stage_gate(journal, stage_number, threshold_level="normal"):
    """Convenience wrapper for agent_manager integration.

    Args:
        journal: Journal object (from agent_manager.journals[stage_name])
        stage_number: int (1-4)
        threshold_level: str

    Returns:
        (passed: bool, report: dict)
    """
    # Convert journal object to dict format
    nodes_data = []
    for node in getattr(journal, "nodes", []):
        nd = {
            "is_buggy": getattr(node, "is_buggy", False),
            "metric": None,
            "plan": getattr(node, "plan", ""),
        }
        metric = getattr(node, "metric", None)
        if metric is not None:
            if hasattr(metric, "value"):
                nd["metric"] = {"value": metric.value,
                                "name": getattr(metric, "name", "")}
            elif isinstance(metric, (int, float)):
                nd["metric"] = metric
        nodes_data.append(nd)

    journal_data = {"nodes": nodes_data}
    report = evaluate_stage_quality(journal_data, stage_number, threshold_level)

    if report["passed"]:
        print(f"[PQF] Stage {stage_number} PASSED (score={report['score']}/100, "
              f"{report['good_nodes']}/{report['total_nodes']} good, "
              f"rec={report['recommendation']})")
    else:
        failed = [k for k, v in report["dimensions"].items() if not v["passed"]]
        print(f"[PQF] Stage {stage_number} FAILED (score={report['score']}/100, "
              f"failed: {failed}, rec={report['recommendation']})")

    return report["passed"], report


# ---------------------------------------------------------------------------
# CLI for standalone evaluation
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--journal_path", required=True,
                   help="path to stage journal.json")
    p.add_argument("--stage_number", type=int, default=1)
    p.add_argument("--threshold", default="normal",
                   choices=["strict", "normal", "lenient"])
    p.add_argument("--out", default=None, help="output JSON path")
    args = p.parse_args()

    with open(args.journal_path) as f:
        journal_data = json.load(f)

    report = evaluate_stage_quality(
        journal_data, args.stage_number, args.threshold)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
