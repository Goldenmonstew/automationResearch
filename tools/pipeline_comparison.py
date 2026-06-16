#!/usr/bin/env python
"""Pipeline comparison: standard vs critique-enhanced writeup.

Runs BOTH pipelines on the same experiment directory and produces a
side-by-side comparison of grounding rates, quality scores, and cost.

This script is the experiment harness for validating P0 improvements.
It uses rsync to create an isolated copy for each pipeline run so they
don't interfere.

Usage (from AI-Scientist-v2 repo root):
  python tools/pipeline_comparison.py --idea_dir experiments/<run> \
      [--model gpt-4o] [--out comparison_results]

Output: <out>.json with per-pipeline metrics + <out>.md summary table.
"""
import argparse
import json
import os
import os.path as osp
import shutil
import subprocess
import sys
import time


def run_cmd(cmd, label, timeout=3600):
    """Run a command and return (returncode, duration_s, stdout_tail)."""
    print(f"\n{'='*60}\n[compare] {label}\n  {' '.join(cmd)}\n{'='*60}")
    t0 = time.time()
    r = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    dur = time.time() - t0
    tail = (r.stdout or "")[-3000:]
    if r.returncode != 0:
        tail += "\nSTDERR:\n" + (r.stderr or "")[-2000:]
    print(f"[compare] {label}: rc={r.returncode} in {dur:.0f}s")
    return r.returncode, round(dur, 1), tail


def copy_experiment(src, dst):
    """rsync the experiment directory to an isolated copy."""
    if osp.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)
    print(f"[compare] copied {src} → {dst}")


def run_standard_pipeline(idea_dir_copy, model, tools_dir, audit_script):
    """Standard pipeline: rescue_writeup → grounding_audit → reground_rewrite."""
    results = {"pipeline": "standard", "steps": []}
    py = sys.executable

    # 1. writeup
    rc, dur, out = run_cmd(
        [py, "-u", osp.join(tools_dir, "rescue_writeup.py"),
         "--idea_dir", idea_dir_copy,
         "--model_writeup", model, "--model_review", model,
         "--model_agg_plots", model, "--model_citation", model],
        "standard:writeup", timeout=2400)
    results["steps"].append({"step": "writeup", "rc": rc, "duration_s": dur})

    # find PDF
    pdf = None
    for f in os.listdir(idea_dir_copy):
        if f.endswith(".pdf") and "reflection" in f:
            pdf = osp.join(idea_dir_copy, f)
    if not pdf:
        for f in os.listdir(idea_dir_copy):
            if f.endswith(".pdf"):
                pdf = osp.join(idea_dir_copy, f)
    if not pdf:
        results["error"] = "no PDF produced"
        return results

    # 2. grounding audit
    audit_out = osp.join(idea_dir_copy, "std_audit")
    rc, dur, out = run_cmd(
        [py, "-u", audit_script, "--pdf", pdf,
         "--exp_dir", idea_dir_copy, "--out", audit_out, "--model", model],
        "standard:audit")
    results["steps"].append({"step": "audit", "rc": rc, "duration_s": dur})

    audit_json = audit_out + ".json"
    if osp.exists(audit_json):
        with open(audit_json) as f:
            audit = json.load(f)
        results["initial_grounded_ratio"] = audit["summary"]["grounded_ratio"]
        results["initial_verdicts"] = audit["summary"]["verdict_counts"]
    else:
        results["error"] = "audit failed"
        return results

    # 3. reground
    rc, dur, out = run_cmd(
        [py, "-u", osp.join(tools_dir, "reground_rewrite.py"),
         "--exp_dir", idea_dir_copy, "--audit", audit_json,
         "--audit_script", audit_script,
         "--model", model, "--max_rounds", "3"],
        "standard:reground", timeout=1800)
    results["steps"].append({"step": "reground", "rc": rc, "duration_s": dur})

    # collect final audit from reground output
    for rnd in range(3, 0, -1):
        fp = osp.join(idea_dir_copy, f"grounding_regrounded_r{rnd}.json")
        if osp.exists(fp):
            with open(fp) as f:
                final = json.load(f)
            results["final_grounded_ratio"] = final["summary"]["grounded_ratio"]
            results["final_verdicts"] = final["summary"]["verdict_counts"]
            results["reground_rounds"] = rnd
            break
    else:
        results["final_grounded_ratio"] = results.get("initial_grounded_ratio", 0)
        results["reground_rounds"] = 0

    results["total_duration_s"] = round(sum(s["duration_s"] for s in results["steps"]), 1)
    results["pdf"] = pdf
    return results


def run_critique_pipeline(idea_dir_copy, model_writeup, model_critique, tools_dir):
    """Enhanced pipeline: evidence_distill → constrained writeup → critique loop."""
    py = sys.executable
    log_path = osp.join(idea_dir_copy, "critique_log.json")

    rc, dur, out = run_cmd(
        [py, "-u", osp.join(tools_dir, "critique_writeup.py"),
         "--idea_dir", idea_dir_copy,
         "--model_distill", model_critique,
         "--model_writeup", model_writeup,
         "--model_critique", model_critique,
         "--model_review", model_writeup,
         "--out_log", log_path],
        "critique:full_pipeline", timeout=3600)

    results = {"pipeline": "critique", "rc": rc, "total_duration_s": dur}
    if osp.exists(log_path):
        with open(log_path) as f:
            clog = json.load(f)
        results["log"] = clog
        results["critique_rounds"] = clog.get("summary", {}).get("critique_rounds_used", 0)
        results["final_grounded_ratio"] = clog.get("summary", {}).get("final_grounded_ratio", 0)

        # extract initial grounded ratio (first critique round)
        for s in clog.get("steps", []):
            if s.get("step", "").startswith("critique_round"):
                results["initial_grounded_ratio"] = s.get("grounded_ratio", 0)
                break
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--idea_dir", required=True)
    p.add_argument("--model", default="gpt-4o",
                   help="model for standard pipeline + writeup in critique pipeline")
    p.add_argument("--model_critique", default="gpt-5.5",
                   help="model for distillation + critique in enhanced pipeline")
    p.add_argument("--out", default="comparison_results",
                   help="output prefix")
    p.add_argument("--skip_standard", action="store_true",
                   help="skip standard pipeline (only run critique)")
    p.add_argument("--skip_critique", action="store_true",
                   help="skip critique pipeline (only run standard)")
    args = p.parse_args()

    assert osp.isdir("ai_scientist"), "cwd must be an AI-Scientist-v2 repo root"
    idea_dir = args.idea_dir.rstrip("/")
    assert osp.isdir(idea_dir), f"missing {idea_dir}"
    tools_dir = osp.join(osp.dirname(osp.abspath(__file__)))
    audit_script = osp.join(tools_dir, "grounding_audit.py")

    comparison = {"experiment": idea_dir, "model": args.model,
                  "model_critique": args.model_critique,
                  "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    # Run standard pipeline on a copy
    if not args.skip_standard:
        std_copy = idea_dir + "_std_copy"
        copy_experiment(idea_dir, std_copy)
        try:
            comparison["standard"] = run_standard_pipeline(
                std_copy, args.model, tools_dir, audit_script)
        finally:
            pass  # keep copy for inspection

    # Run critique pipeline on a copy
    if not args.skip_critique:
        crit_copy = idea_dir + "_crit_copy"
        copy_experiment(idea_dir, crit_copy)
        try:
            comparison["critique"] = run_critique_pipeline(
                crit_copy, args.model, args.model_critique, tools_dir)
        finally:
            pass

    # Summary comparison table
    lines = ["# Pipeline Comparison",
             f"- experiment: {idea_dir}",
             f"- model (writeup): {args.model}",
             f"- model (critique): {args.model_critique}",
             f"- timestamp: {comparison['timestamp']}",
             "",
             "| Metric | Standard | Critique-Enhanced |",
             "|--------|----------|-------------------|"]

    def get(d, k, default="-"):
        return d.get(k, default) if d else default

    std = comparison.get("standard", {})
    crit = comparison.get("critique", {})
    lines.append(f"| Initial grounded ratio | "
                 f"{get(std, 'initial_grounded_ratio')} | "
                 f"{get(crit, 'initial_grounded_ratio')} |")
    lines.append(f"| Final grounded ratio | "
                 f"{get(std, 'final_grounded_ratio')} | "
                 f"{get(crit, 'final_grounded_ratio')} |")
    lines.append(f"| Repair rounds needed | "
                 f"{get(std, 'reground_rounds')} | "
                 f"{get(crit, 'critique_rounds')} |")
    lines.append(f"| Total time (s) | "
                 f"{get(std, 'total_duration_s')} | "
                 f"{get(crit, 'total_duration_s')} |")

    with open(args.out + ".json", "w") as f:
        json.dump(comparison, f, indent=2)
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(lines))

    print(f"\n{'='*60}")
    print("\n".join(lines))
    print(f"\n→ {args.out}.json / .md")


if __name__ == "__main__":
    main()
