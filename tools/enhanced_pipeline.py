#!/usr/bin/env python
"""Full enhanced pipeline: P0 + P1 integrated.

Chains all improvements into a single post-tree-search pipeline:
  1. Evidence distillation (P0a) — structured evidence summary
  2. Multi-strategy writeup (P1) — N independent drafts
  3. Per-strategy critique loop (P0b) — fix hallucinations
  4. Blind judge (P1) — pick best draft
  5. Final review — quality score

Compared to the standard pipeline (rescue_writeup → audit → reground),
this pipeline front-loads evidence grounding and generates diverse drafts,
reducing post-hoc repair needs.

Usage (from AI-Scientist-v2 repo root, PYTHONPATH includes it):
  python tools/enhanced_pipeline.py --idea_dir experiments/<run> \
      [--model_writeup gpt-4o] [--model_critique gpt-5.5] \
      [--strategies precision,narrative,complete] \
      [--max_critique_rounds 3]

For single-strategy mode (faster, P0 only):
  python tools/enhanced_pipeline.py --idea_dir experiments/<run> \
      --strategies precision --max_critique_rounds 5
"""
import argparse
import json
import os
import os.path as osp
import shutil
import sys
import time
import traceback

TOOLS_DIR = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, TOOLS_DIR)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--idea_dir", required=True)
    p.add_argument("--model_writeup", default="gpt-4o",
                   help="model for writeup generation and review")
    p.add_argument("--model_critique", default="gpt-4o",
                   help="model for evidence distillation and critique")
    p.add_argument("--model_judge", default="gpt-5.5",
                   help="model for blind judge (P1)")
    p.add_argument("--strategies", default="precision,narrative,complete",
                   help="comma-separated writeup strategies (P1)")
    p.add_argument("--max_critique_rounds", type=int, default=3,
                   help="max critique iterations per strategy")
    p.add_argument("--grounded_threshold", type=float, default=0.90)
    p.add_argument("--skip_writeup", action="store_true",
                   help="use existing tex, only run critique + judge")
    p.add_argument("--skip_agg", action="store_true")
    p.add_argument("--out_log", default=None)
    args = p.parse_args()

    assert osp.isdir("ai_scientist/blank_icbinb_latex"), \
        "cwd must be an AI-Scientist-v2 repo root"
    idea_dir = args.idea_dir.rstrip("/")
    base = osp.basename(idea_dir)
    log_path = args.out_log or osp.join(idea_dir, "enhanced_pipeline_log.json")
    strategies = [s.strip() for s in args.strategies.split(",")]

    log = {"experiment": idea_dir, "pipeline": "enhanced_v1",
           "args": vars(args), "steps": []}

    def log_step(name, **kw):
        entry = {"step": name, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}
        log["steps"].append(entry)
        brief = {k: v for k, v in kw.items()
                 if k not in ("details", "final_tex", "strategy_results")}
        print(f"[ep] === {name} === {json.dumps(brief, default=str)}")
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2, default=str)

    t_total = time.time()

    # ================================================================
    # Step 1: Evidence distillation (P0a)
    # ================================================================
    from evidence_distiller import distill
    ev_path = osp.join(idea_dir, "evidence_distilled.json")

    t0 = time.time()
    evidence = distill(idea_dir, model=args.model_critique)
    with open(ev_path, "w") as f:
        json.dump(evidence, f, indent=2)
    log_step("evidence_distill",
             duration_s=round(time.time() - t0, 1),
             n_validated=evidence["meta"]["n_validated_findings"],
             n_not_observed=evidence["meta"]["n_not_observed"],
             n_constraints=evidence["meta"]["n_writeup_constraints"])

    # ================================================================
    # Step 2: Multi-strategy writeup + per-strategy critique (P0b + P1)
    # ================================================================
    if len(strategies) == 1 and args.skip_writeup:
        # Single strategy, skip_writeup: just run critique on existing tex
        print("[ep] single-strategy skip_writeup mode — critique only")
        from critique_writeup import critique_tex, fix_tex, build_constraint_block
        tex_path = osp.join(idea_dir, "latex", "template.tex")
        assert osp.exists(tex_path), f"no tex at {tex_path}"
        with open(tex_path) as f:
            tex = f.read()
        shutil.copy(tex_path, tex_path + ".pre_enhanced")

        from critique_writeup import MIN_FIXABLE_RATIO
        for rnd in range(1, args.max_critique_rounds + 1):
            t2 = time.time()
            claims, counts, ratio = critique_tex(tex, evidence, args.model_critique)
            bad = [c for c in claims
                   if c.get("audit_verdict") in ("unsupported", "contradicted")]
            if ratio >= args.grounded_threshold:
                log_step(f"critique_round_{rnd}",
                         duration_s=round(time.time()-t2, 1),
                         grounded_ratio=ratio, action="passed")
                break
            if ratio < MIN_FIXABLE_RATIO:
                log_step(f"critique_round_{rnd}",
                         duration_s=round(time.time()-t2, 1),
                         grounded_ratio=ratio,
                         action="skipped_low_grounding")
                print(f"[ep] grounding {ratio:.2f} < {MIN_FIXABLE_RATIO} — "
                      "skipping fix to avoid over-hedging")
                break
            new_tex, applied = fix_tex(tex, bad, evidence, args.model_critique)
            log_step(f"critique_round_{rnd}",
                     duration_s=round(time.time()-t2, 1),
                     grounded_ratio=ratio, edits=applied, action="fixed")
            if applied == 0:
                break
            tex = new_tex
            with open(tex_path, "w") as f:
                f.write(tex)

        # Compile
        from ai_scientist.perform_icbinb_writeup import compile_latex
        pdf_out = osp.join(idea_dir, f"{base}_enhanced.pdf")
        compile_latex(osp.join(idea_dir, "latex"), pdf_out)
        log_step("compile", pdf_exists=osp.exists(pdf_out))

    elif len(strategies) >= 2:
        # Multi-strategy mode (P1)
        from multi_strategy_writeup import (
            run_single_strategy, blind_judge, STRATEGIES,
            make_strategy_workdir,
        )
        import ai_scientist.perform_icbinb_writeup as icb

        # Gather citations (shared)
        t_cite = time.time()
        citations_text = icb.gather_citations(
            idea_dir, num_cite_rounds=5, small_model=args.model_writeup)
        log_step("citations", duration_s=round(time.time() - t_cite, 1))

        strategy_results = []
        for sk in strategies:
            if sk not in STRATEGIES:
                continue
            work_dir = make_strategy_workdir(idea_dir, sk)
            os.makedirs(osp.dirname(work_dir), exist_ok=True)
            if osp.exists(work_dir):
                shutil.rmtree(work_dir)
            shutil.copytree(idea_dir, work_dir, symlinks=True)
            print(f"\n[ep] === STRATEGY {STRATEGIES[sk]['label']}: {sk} ===")

            sr = run_single_strategy(
                idea_dir, sk, evidence, args.model_writeup,
                args.model_critique, citations_text, work_dir)
            strategy_results.append(sr)
            log_step(f"strategy_{sk}",
                     total_time_s=sr.get("total_time_s"),
                     writeup_success=sr.get("writeup_success"),
                     post_critique_grounded=sr.get("post_critique_grounded"),
                     pdf_exists=sr.get("pdf_exists"))

        # Blind judge
        successful = [sr for sr in strategy_results if sr.get("pdf_exists")]
        if len(successful) >= 2:
            t_judge = time.time()
            verdict = blind_judge(successful, args.model_judge)
            log_step("blind_judge",
                     duration_s=round(time.time() - t_judge, 1),
                     winner=verdict.get("winner"),
                     scores=verdict.get("scores"),
                     reasoning=verdict.get("reasoning"))

            # Copy winner
            label_to_key = {STRATEGIES[k]["label"]: k for k in STRATEGIES}
            winner_key = label_to_key.get(verdict.get("winner", ""))
            if winner_key:
                winner_work = make_strategy_workdir(idea_dir, winner_key)
                winner_tex = osp.join(winner_work, "latex", "template.tex")
                if osp.exists(winner_tex):
                    shutil.copy(winner_tex,
                                osp.join(idea_dir, "latex", "template.tex"))
                winner_pdf = next(
                    (sr.get("pdf_path") for sr in successful
                     if sr["strategy"] == winner_key), None)
                if winner_pdf and osp.exists(winner_pdf):
                    dst = osp.join(idea_dir, f"{base}_enhanced.pdf")
                    shutil.copy(winner_pdf, dst)
                    print(f"[ep] winner: {winner_key} → {dst}")
        elif len(successful) == 1:
            log_step("blind_judge",
                     winner=successful[0]["label"],
                     reasoning="only one strategy succeeded")
            if successful[0].get("pdf_path"):
                dst = osp.join(idea_dir, f"{base}_enhanced.pdf")
                shutil.copy(successful[0]["pdf_path"], dst)

        # Strip large tex from results
        for sr in strategy_results:
            sr.pop("final_tex", None)
        log["strategy_results"] = strategy_results

    else:
        # Single strategy with writeup
        from critique_writeup import main as critique_main
        # Delegate to critique_writeup which handles writeup + critique
        print("[ep] single-strategy mode — delegating to critique_writeup")
        sys.argv = [
            "critique_writeup.py",
            "--idea_dir", idea_dir,
            "--model_distill", args.model_critique,
            "--model_writeup", args.model_writeup,
            "--model_critique", args.model_critique,
            "--model_review", args.model_writeup,
            "--max_critique_rounds", str(args.max_critique_rounds),
            "--grounded_threshold", str(args.grounded_threshold),
            "--out_log", log_path,
        ]
        critique_main()
        return

    # ================================================================
    # Step 3: Final review
    # ================================================================
    pdf_out = osp.join(idea_dir, f"{base}_enhanced.pdf")
    if osp.exists(pdf_out):
        t3 = time.time()
        try:
            from ai_scientist.llm import create_client
            from ai_scientist.perform_llm_review import load_paper, perform_review
            paper_content = load_paper(pdf_out)
            client, client_model = create_client(args.model_writeup)
            review = perform_review(paper_content, client_model, client)
            review_path = osp.join(idea_dir, f"{base}_enhanced_review.json")
            with open(review_path, "w") as f:
                json.dump(review, f, indent=4)
            overall = review.get("Overall") if isinstance(review, dict) else "?"
            log_step("review", duration_s=round(time.time()-t3, 1),
                     model=args.model_writeup, overall=overall)
        except Exception:
            traceback.print_exc()
            log_step("review", error=traceback.format_exc()[-500:])

    # ================================================================
    # Summary
    # ================================================================
    total_time = time.time() - t_total
    final_ratio = 0
    for s in reversed(log["steps"]):
        if "grounded_ratio" in s:
            final_ratio = s["grounded_ratio"]
            break
        if "post_critique_grounded" in s:
            final_ratio = s["post_critique_grounded"]
            break

    log["summary"] = {
        "total_duration_s": round(total_time, 1),
        "pipeline": "enhanced_v1",
        "strategies": strategies,
        "final_grounded_ratio": final_ratio,
        "pdf": pdf_out if osp.exists(pdf_out) else None,
    }
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)

    print(f"\n[ep] DONE — {log_path}")
    print(f"[ep] summary: {json.dumps(log['summary'])}")


if __name__ == "__main__":
    main()
