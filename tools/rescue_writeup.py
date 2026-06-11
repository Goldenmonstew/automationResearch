#!/usr/bin/env python
"""Offline rescue for killed tree-search runs.

A run killed mid-search has per-stage journal.json files on disk (written every
step) but lacks the 4 stage summary JSONs (only written after all stages
finish), which are hard dependencies of plot aggregation and writeup. This
driver rebuilds the summaries from the on-disk journals, then runs the official
post-processing chain: plot aggregation -> citations -> icbinb writeup ->
review.

MUST be executed with cwd = the repo root of the run's working copy (journal
exp_results_dir entries and the latex template are cwd-relative), with
PYTHONPATH including that root.

Usage:
  python rescue_writeup.py --idea_dir experiments/<ts>_<idea>_attempt_0
"""
import argparse
import json
import os
import os.path as osp
import re
import shutil
import sys
import traceback
from dataclasses import fields


def pick_stage_journals(run_dir):
    """Group stage_* dirs by main stage number; per stage pick the journal.json
    with the newest mtime (= latest substage). Returns [(stage_name, data)]
    ordered by main stage number, skipping unparseable journals."""
    groups = {}
    for d in sorted(os.listdir(run_dir)):
        m = re.match(r"stage_(\d+)_", d)
        jp = osp.join(run_dir, d, "journal.json")
        if m and osp.exists(jp):
            groups.setdefault(int(m.group(1)), []).append(jp)
    picked = []
    for stage_num in sorted(groups):
        candidates = sorted(groups[stage_num], key=lambda p: os.path.getmtime(p))
        chosen = None
        # prefer newest substage, fall back to older ones if json is truncated
        for jp in reversed(candidates):
            try:
                with open(jp) as f:
                    data = json.load(f)
                chosen = (jp, data)
                break
            except json.JSONDecodeError as e:
                print(f"[rescue] WARNING corrupt journal {jp}: {e}")
        if chosen:
            stage_name = osp.basename(osp.dirname(chosen[0]))
            picked.append((stage_name, chosen[0], chosen[1]))
            print(f"[rescue] stage {stage_num}: using {chosen[0]} "
                  f"({len(chosen[1].get('nodes', []))} nodes)")
        else:
            print(f"[rescue] stage {stage_num}: no usable journal, skipping")
    return picked


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--idea_dir", required=True,
                   help="run dir relative to repo root, e.g. experiments/..._attempt_0")
    p.add_argument("--model_agg_plots", default="gpt-4o")
    p.add_argument("--model_citation", default="gpt-4o")
    p.add_argument("--model_writeup", default="gpt-4o")
    p.add_argument("--model_review", default="gpt-4o")
    p.add_argument("--num_cite_rounds", type=int, default=5)
    p.add_argument("--writeup_retries", type=int, default=3)
    p.add_argument("--skip_summaries", action="store_true",
                   help="reuse already-rebuilt summary JSONs")
    p.add_argument("--skip_agg", action="store_true",
                   help="keep existing figures/ (e.g. after fix_aggregation)")
    args = p.parse_args()

    assert osp.isdir("ai_scientist/blank_icbinb_latex"), \
        "cwd must be an AI-Scientist-v2 repo root"
    idea_dir = args.idea_dir.rstrip("/")
    run_dir = osp.join(idea_dir, "logs", "0-run")
    assert osp.isdir(run_dir), f"missing {run_dir}"

    from omegaconf import OmegaConf
    from ai_scientist.treesearch.journal import Journal, Node
    from ai_scientist.treesearch import log_summarization as ls

    node_fields = {f.name for f in fields(Node)}

    def reconstruct_journal(jd):
        id_to_node = {}
        for nd in jd.get("nodes", []):
            clean = {k: v for k, v in nd.items() if k in node_fields}
            node = Node.from_dict(clean)
            id_to_node[node.id] = node
        for nid, pid in jd.get("node2parent", {}).items():
            if nid in id_to_node and pid in id_to_node:
                child, parent = id_to_node[nid], id_to_node[pid]
                child.parent = parent
                parent.children.add(child)
        j = Journal()
        j.nodes.extend(id_to_node.values())
        return j

    # ---- 1. rebuild the 4 stage summary JSONs from on-disk journals ----
    if not args.skip_summaries:
        picked = pick_stage_journals(run_dir)
        assert picked, "no usable stage journals found"
        journals = [(name, reconstruct_journal(data)) for name, _, data in picked]
        cfg = OmegaConf.load(osp.join(osp.dirname(picked[-1][1]), "config.yaml"))
        summaries = ls.overall_summarize(journals, cfg=cfg)
        for name, data in zip(["draft", "baseline", "research", "ablation"], summaries):
            out = osp.join(run_dir, f"{name}_summary.json")
            with open(out, "w") as f:
                json.dump(data, f, indent=2)
            print(f"[rescue] wrote {out}")

    # ---- 2. official post-processing chain (mirrors launch_scientist_bfts) ----
    if not args.skip_agg:
        src = osp.join(run_dir, "experiment_results")
        dst = osp.join(idea_dir, "experiment_results")
        if osp.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)

        from ai_scientist.perform_plotting import aggregate_plots
        aggregate_plots(base_folder=idea_dir, model=args.model_agg_plots)
        if osp.isdir(dst):
            shutil.rmtree(dst)

    figdir = osp.join(idea_dir, "figures")
    n_figs = len(os.listdir(figdir)) if osp.isdir(figdir) else 0
    print(f"[rescue] aggregated figures: {n_figs}")
    if n_figs == 0:
        print("[rescue] WARNING: no figures produced; writeup will lack figures")

    import ai_scientist.perform_icbinb_writeup as icb
    citations_text = icb.gather_citations(
        idea_dir,
        num_cite_rounds=args.num_cite_rounds,
        small_model=args.model_citation,
    )
    writeup_fn = getattr(icb, "perform_writeup", None) or getattr(icb, "perform_icbinb_writeup")
    success = False
    for attempt in range(args.writeup_retries):
        print(f"[rescue] writeup attempt {attempt + 1}/{args.writeup_retries}")
        try:
            success = writeup_fn(
                base_folder=idea_dir,
                small_model=args.model_writeup,
                big_model=args.model_writeup,
                page_limit=4,
                citations_text=citations_text,
            )
        except Exception:
            traceback.print_exc()
        if success:
            break
    print(f"[rescue] writeup success flag: {success}")

    # ---- 3. review (PDF may exist even when the success flag is False) ----
    pdf_files = [f for f in os.listdir(idea_dir) if f.endswith(".pdf")]
    pdf_path = None
    reflections = [f for f in pdf_files if "reflection" in f]
    if reflections:
        finals = [f for f in reflections if "final" in f.lower()]
        if finals:
            pdf_path = osp.join(idea_dir, finals[0])
        else:
            nums = []
            for f in reflections:
                m = re.search(r"reflection[_.]?(\d+)", f)
                if m:
                    nums.append((int(m.group(1)), f))
            pdf_path = osp.join(idea_dir, max(nums)[1] if nums else reflections[0])
    elif pdf_files:
        pdf_path = osp.join(idea_dir, pdf_files[0])

    if pdf_path and osp.exists(pdf_path):
        print(f"[rescue] reviewing {pdf_path}")
        from ai_scientist.llm import create_client
        from ai_scientist.perform_llm_review import load_paper, perform_review
        from ai_scientist.perform_vlm_review import perform_imgs_cap_ref_review
        paper_content = load_paper(pdf_path)
        client, client_model = create_client(args.model_review)
        review_text = perform_review(paper_content, client_model, client)
        with open(osp.join(idea_dir, "review_text.txt"), "w") as f:
            f.write(json.dumps(review_text, indent=4))
        try:
            img_rev = perform_imgs_cap_ref_review(client, client_model, pdf_path)
            with open(osp.join(idea_dir, "review_img_cap_ref.json"), "w") as f:
                json.dump(img_rev, f, indent=4)
        except Exception:
            traceback.print_exc()
        print(f"[rescue] DONE pdf={pdf_path} overall={review_text.get('Overall') if isinstance(review_text, dict) else '?'}")
    else:
        print("[rescue] DONE but no PDF produced")


if __name__ == "__main__":
    main()
