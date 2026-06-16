#!/usr/bin/env python
"""P1: Multi-Strategy Writeup + Blind Judge.

ICR-inspired BFS over writeup strategies: generate N independent writeup
drafts with distinct framing strategies, run a quick critique pass on each,
then let a blind judge pick the best.

Strategies:
  A (precision): strict evidence grounding, conservative claims, exact numbers
  B (narrative):  compelling story arc, strong motivation → conclusion flow
  C (complete):   exhaustive coverage, failed approaches, limitations prominent

Architecture:
  1. evidence_distill (shared, reuse P0a)
  2. for each strategy:  inject strategy-specific constraints → writeup → 1-round critique
  3. blind_judge: sees all N papers anonymized, scores and picks winner
  4. winning paper → final output

Usage (from AI-Scientist-v2 repo root, PYTHONPATH includes it):
  python tools/multi_strategy_writeup.py --idea_dir experiments/<run> \
      [--model_writeup gpt-4o] [--model_judge gpt-5.5]
"""
import argparse
import json
import os
import os.path as osp
import re
import shutil
import sys
import time
import traceback

TOOLS_DIR = osp.dirname(osp.abspath(__file__))

# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

STRATEGIES = {
    "precision": {
        "label": "A",
        "description": "Precision-first: exact numbers, conservative claims, strict grounding",
        "constraint_prefix": """
WRITEUP STRATEGY: PRECISION-FIRST
- Every quantitative claim MUST cite the exact source (stage/node).
- Use hedging language ("our results suggest", "we observe") rather than
  strong causal claims unless the experiment design warrants it.
- If a result is based on a single seed, explicitly state this limitation.
- Prefer under-claiming to over-claiming — omit a result rather than overstate it.
- Report ALL exact numbers from validated findings, even small improvements.
""",
    },
    "narrative": {
        "label": "B",
        "description": "Narrative-first: compelling story, strong motivation, clear takeaway",
        "constraint_prefix": """
WRITEUP STRATEGY: NARRATIVE-FIRST
- Open with a clear, motivating problem statement grounded in the evidence.
- Build a logical arc: problem → hypothesis → method → key result → insight.
- Lead with the most impactful finding, not the first chronological one.
- Write the abstract and conclusion to be self-contained and compelling.
- Still grounded in evidence — but prioritize clarity and impact of presentation.
- Use active voice and direct statements where evidence supports them.
""",
    },
    "complete": {
        "label": "C",
        "description": "Completeness-first: cover everything, include failures and limitations",
        "constraint_prefix": """
WRITEUP STRATEGY: COMPLETENESS-FIRST
- Cover EVERY experiment that was run, including failed approaches.
- Dedicate a subsection to negative results and what they teach us.
- Include a detailed limitations section discussing all NOT_OBSERVED items.
- Report ablation results exhaustively, even when differences are small.
- Include methodology details sufficient for reproduction.
- Discuss what the experiment CANNOT conclude (explicit scope boundaries).
""",
    },
}

# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def llm_call(prompt, system, model):
    if model.startswith("gpt-5"):
        try:
            import openai
            client = openai.OpenAI()
            r = client.responses.create(
                model=model, instructions=system, input=prompt,
                service_tier="priority", max_output_tokens=32768)
            txt = getattr(r, "output_text", None)
            if txt and txt.strip():
                return txt
        except Exception as e:
            print(f"[msw] responses-API failed ({e}); chat fallback")
    from ai_scientist.llm import create_client, get_response_from_llm
    client, mname = create_client(model)
    resp, _ = get_response_from_llm(
        prompt, client=client, model=mname, system_message=system, print_debug=False)
    return resp


def parse_json_object(text):
    patterns = [
        re.findall(r"```json\s*(.*?)```", text or "", re.DOTALL),
        re.findall(r"```\s*(.*?)```", text or "", re.DOTALL),
        [text or ""],
    ]
    for candidates in patterns:
        for c in candidates:
            c = c.strip()
            start = c.find("{")
            if start == -1:
                continue
            dec = json.JSONDecoder()
            try:
                obj, _ = dec.raw_decode(c[start:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON object in response (len={len(text or '')})")


# ---------------------------------------------------------------------------
# Single strategy runner
# ---------------------------------------------------------------------------

def make_strategy_workdir(idea_dir, strategy_key):
    """Create a workdir for a strategy that preserves the original basename.

    perform_writeup derives PDF names from osp.basename(base_folder), so the
    workdir must have the same basename as the original experiment directory.
    We put it under a sibling 'strategy_<key>/' directory instead of renaming.
    """
    parent = osp.dirname(idea_dir)
    base = osp.basename(idea_dir)
    work_dir = osp.join(parent, f"strategy_{strategy_key}", base)
    return work_dir


def run_single_strategy(idea_dir, strategy_key, evidence, model_writeup,
                        model_critique, citations_text, work_dir):
    """Run writeup + 1-round critique for one strategy. Returns dict with metrics."""
    strat = STRATEGIES[strategy_key]
    result = {"strategy": strategy_key, "label": strat["label"]}
    t0 = time.time()

    idea_md = osp.join(work_dir, "idea.md")
    idea_backup = idea_md + f".pre_{strategy_key}"

    try:
        # Inject strategy + evidence constraints
        sys.path.insert(0, TOOLS_DIR)
        from critique_writeup import build_constraint_block
        constraint_block = build_constraint_block(evidence)
        strategy_block = strat["constraint_prefix"]

        if osp.exists(idea_md):
            shutil.copy(idea_md, idea_backup)
            with open(idea_md, "a") as f:
                f.write("\n" + strategy_block + "\n" + constraint_block)

        # Run writeup
        import ai_scientist.perform_icbinb_writeup as icb
        writeup_fn = getattr(icb, "perform_writeup", None) or \
                     getattr(icb, "perform_icbinb_writeup")

        success = False
        for attempt in range(2):
            try:
                success = writeup_fn(
                    base_folder=work_dir,
                    small_model=model_writeup,
                    big_model=model_writeup,
                    page_limit=4,
                    citations_text=citations_text,
                    n_writeup_reflections=2,
                )
            except Exception:
                traceback.print_exc()
            if success:
                break

        result["writeup_success"] = success
        result["writeup_time_s"] = round(time.time() - t0, 1)

        # Don't trust return value — perform_writeup often returns False
        # even when tex was produced. Check for actual tex instead.
        tex_path = osp.join(work_dir, "latex", "template.tex")
        if not success and not osp.exists(tex_path):
            result["error"] = "writeup failed — no tex produced"
            return result

        # Quick 1-round critique
        if osp.exists(tex_path):
            from critique_writeup import critique_tex, fix_tex
            with open(tex_path) as f:
                tex = f.read()

            claims, counts, ratio = critique_tex(tex, evidence, model_critique)
            result["pre_critique_grounded"] = ratio
            result["pre_critique_counts"] = counts

            bad = [c for c in claims
                   if c.get("audit_verdict") in ("unsupported", "contradicted")]
            from critique_writeup import MIN_FIXABLE_RATIO
            if bad and ratio < 0.90 and ratio >= MIN_FIXABLE_RATIO:
                new_tex, applied = fix_tex(tex, bad, evidence, model_critique)
                with open(tex_path, "w") as f:
                    f.write(new_tex)
                result["critique_edits"] = applied

                # Re-check after fix
                claims2, counts2, ratio2 = critique_tex(new_tex, evidence, model_critique)
                result["post_critique_grounded"] = ratio2
                result["post_critique_counts"] = counts2
            elif bad and ratio < MIN_FIXABLE_RATIO:
                print(f"[msw] grounding {ratio:.2f} < {MIN_FIXABLE_RATIO} — "
                      "skipping fix_tex to avoid over-hedging")
                result["post_critique_grounded"] = ratio
                result["critique_skipped"] = "grounding_too_low"
            else:
                result["post_critique_grounded"] = ratio

        # Compile PDF
        pdf_file = osp.join(work_dir,
                            f"{osp.basename(work_dir)}_{strategy_key}.pdf")
        icb.compile_latex(osp.join(work_dir, "latex"), pdf_file)
        result["pdf_exists"] = osp.exists(pdf_file)
        result["pdf_path"] = pdf_file if osp.exists(pdf_file) else None

        # Read final tex for judge
        if osp.exists(tex_path):
            with open(tex_path) as f:
                result["final_tex"] = f.read()

    finally:
        if osp.exists(idea_backup):
            shutil.copy(idea_backup, idea_md)

    result["total_time_s"] = round(time.time() - t0, 1)
    return result


# ---------------------------------------------------------------------------
# Blind Judge
# ---------------------------------------------------------------------------

JUDGE_SYS = """You are a senior ML conference reviewer conducting a BLIND comparison
of three paper drafts about the SAME experiment. You do not know which draft
was written by which strategy. Evaluate each purely on its merits."""

JUDGE_PROMPT = """Below are three drafts of a paper about the same experiment.
Your job: score each draft and pick the best one.

SCORING CRITERIA (1-10 each):
1. Scientific Accuracy: Are claims well-supported? Any hallucination red flags?
2. Clarity: Is the paper easy to follow? Good structure and flow?
3. Completeness: Does it cover the key results, methods, and limitations?
4. Overall Quality: Would this pass peer review at a workshop level?

{papers}

Return a JSON object inside a ```json fence:
{{
  "scores": {{
    "A": {{"accuracy": <int>, "clarity": <int>, "completeness": <int>, "overall": <int>}},
    "B": {{"accuracy": <int>, "clarity": <int>, "completeness": <int>, "overall": <int>}},
    "C": {{"accuracy": <int>, "clarity": <int>, "completeness": <int>, "overall": <int>}}
  }},
  "winner": "<A or B or C>",
  "reasoning": "<2-3 sentences explaining why the winner is best>"
}}"""


def blind_judge(strategy_results, model_judge):
    """Run blind evaluation of all strategy outputs. Returns judge verdict."""
    papers_text = []
    label_map = {}
    for sr in strategy_results:
        label = sr["label"]
        tex = sr.get("final_tex", "")
        if not tex:
            continue
        # Truncate to ~20k chars for judge context
        tex_truncated = tex[:20000]
        papers_text.append(f"=== PAPER {label} ===\n```latex\n{tex_truncated}\n```\n")
        label_map[label] = sr["strategy"]

    if len(papers_text) < 2:
        return {"error": "fewer than 2 papers to judge", "label_map": label_map}

    prompt = JUDGE_PROMPT.format(papers="\n".join(papers_text))
    resp = llm_call(prompt, JUDGE_SYS, model_judge)

    try:
        verdict = parse_json_object(resp)
        verdict["label_map"] = label_map
        return verdict
    except Exception as e:
        return {"error": str(e), "raw_response": resp[:2000], "label_map": label_map}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--idea_dir", required=True)
    p.add_argument("--model_writeup", default="gpt-4o")
    p.add_argument("--model_critique", default="gpt-4o")
    p.add_argument("--model_judge", default="gpt-5.5")
    p.add_argument("--strategies", default="precision,narrative,complete",
                   help="comma-separated strategy keys to run")
    p.add_argument("--skip_evidence", action="store_true",
                   help="reuse existing evidence_distilled.json")
    p.add_argument("--out_log", default=None)
    args = p.parse_args()

    assert osp.isdir("ai_scientist/blank_icbinb_latex"), \
        "cwd must be an AI-Scientist-v2 repo root"
    idea_dir = args.idea_dir.rstrip("/")
    log_path = args.out_log or osp.join(idea_dir, "multi_strategy_log.json")
    strategy_keys = [s.strip() for s in args.strategies.split(",")]

    log = {"experiment": idea_dir, "pipeline": "multi_strategy_v1",
           "args": vars(args), "strategies_requested": strategy_keys,
           "steps": []}

    def log_step(name, **kw):
        entry = {"step": name, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}
        log["steps"].append(entry)
        brief = {k: v for k, v in kw.items()
                 if k not in ("details", "final_tex", "strategy_results")}
        print(f"[msw] === {name} === {json.dumps(brief)}")
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2, default=str)

    t_total = time.time()

    # Step 1: Evidence distillation (shared across strategies)
    ev_path = osp.join(idea_dir, "evidence_distilled.json")
    if args.skip_evidence and osp.exists(ev_path):
        print(f"[msw] reusing existing evidence: {ev_path}")
        with open(ev_path) as f:
            evidence = json.load(f)
    else:
        sys.path.insert(0, TOOLS_DIR)
        from evidence_distiller import distill
        t0 = time.time()
        evidence = distill(idea_dir, model=args.model_critique)
        with open(ev_path, "w") as f:
            json.dump(evidence, f, indent=2)
        log_step("evidence_distill", duration_s=round(time.time() - t0, 1),
                 n_validated=evidence["meta"]["n_validated_findings"])

    # Step 2: Gather citations (shared)
    t_cite = time.time()
    import ai_scientist.perform_icbinb_writeup as icb
    citations_text = icb.gather_citations(
        idea_dir, num_cite_rounds=5, small_model=args.model_writeup)
    log_step("citations", duration_s=round(time.time() - t_cite, 1))

    # Step 3: Run each strategy on its own copy
    strategy_results = []
    for sk in strategy_keys:
        if sk not in STRATEGIES:
            print(f"[msw] unknown strategy '{sk}', skipping")
            continue

        work_dir = make_strategy_workdir(idea_dir, sk)
        if osp.exists(work_dir):
            shutil.rmtree(work_dir)
        os.makedirs(osp.dirname(work_dir), exist_ok=True)
        shutil.copytree(idea_dir, work_dir, symlinks=True)
        print(f"\n[msw] === STRATEGY {STRATEGIES[sk]['label']}: {sk} ===")

        sr = run_single_strategy(
            idea_dir, sk, evidence, args.model_writeup,
            args.model_critique, citations_text, work_dir)
        strategy_results.append(sr)
        log_step(f"strategy_{sk}",
                 total_time_s=sr.get("total_time_s"),
                 writeup_success=sr.get("writeup_success"),
                 pre_critique_grounded=sr.get("pre_critique_grounded"),
                 post_critique_grounded=sr.get("post_critique_grounded"),
                 pdf_exists=sr.get("pdf_exists"))

    # Step 4: Blind judge
    successful = [sr for sr in strategy_results if sr.get("pdf_exists")]
    if len(successful) >= 2:
        t_judge = time.time()
        verdict = blind_judge(successful, args.model_judge)
        log_step("blind_judge", duration_s=round(time.time() - t_judge, 1),
                 winner=verdict.get("winner"),
                 scores=verdict.get("scores"),
                 reasoning=verdict.get("reasoning"))

        # Copy winner to main dir
        winner_label = verdict.get("winner", "")
        label_to_key = {STRATEGIES[k]["label"]: k for k in STRATEGIES}
        winner_key = label_to_key.get(winner_label)
        if winner_key:
            winner_sr = next((sr for sr in successful
                              if sr["strategy"] == winner_key), None)
            if winner_sr and winner_sr.get("pdf_path"):
                dst = osp.join(idea_dir,
                               f"{osp.basename(idea_dir)}_best_strategy.pdf")
                shutil.copy(winner_sr["pdf_path"], dst)
                # Also copy the winning tex
                winner_work = make_strategy_workdir(idea_dir, winner_key)
                winner_tex = osp.join(winner_work, "latex", "template.tex")
                if osp.exists(winner_tex):
                    shutil.copy(winner_tex,
                                osp.join(idea_dir, "latex", "template.tex"))
                print(f"[msw] winner: strategy {winner_label} ({winner_key}) → {dst}")
    elif len(successful) == 1:
        verdict = {"winner": successful[0]["label"],
                   "reasoning": "only one strategy produced a PDF"}
        log_step("blind_judge", winner=verdict["winner"],
                 reasoning=verdict["reasoning"])
    else:
        verdict = {"error": "no strategies produced a PDF"}
        log_step("blind_judge", error="no PDFs produced")

    # Summary
    log["summary"] = {
        "total_duration_s": round(time.time() - t_total, 1),
        "strategies_run": len(strategy_results),
        "strategies_succeeded": len(successful),
        "winner": verdict.get("winner"),
        "winner_strategy": label_to_key.get(verdict.get("winner", "")) if verdict.get("winner") else None,
    }
    # Strip large tex from log before final write
    for sr in strategy_results:
        sr.pop("final_tex", None)
    log["strategy_results"] = strategy_results

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)

    print(f"\n[msw] DONE — log at {log_path}")
    print(f"[msw] summary: {json.dumps(log['summary'])}")


if __name__ == "__main__":
    main()
