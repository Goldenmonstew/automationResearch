#!/usr/bin/env python
"""Heterogeneous 5-vote ensemble review: gpt-5.5 x3 + gpt-4o x2 + meta-review.

Each vote is an independent single review (perform_review with
num_reviews_ensemble=1, num_fs_examples=1, num_reflections=1 — same setting as
the historical single-vote calibration so scores stay comparable). The official
get_meta_review aggregates the 5 into an Area-Chair style meta review; we also
report mean/std per numeric dimension and the decision votes.

Run from an AI-Scientist-v2 repo root (fewshot example paths are cwd-relative).

Usage:
  python ensemble_review.py --out <dir> --pdf a.pdf [--pdf b.pdf ...] [--tag label]
"""
import argparse
import json
import os
import os.path as osp
import time
import traceback

import numpy as np

VOTERS = ["gpt-5.5", "gpt-5.5", "gpt-5.5", "gpt-4o", "gpt-4o"]
META_MODEL = "gpt-4o"
NUMERIC = [
    ("Originality", (1, 4)),
    ("Quality", (1, 4)),
    ("Clarity", (1, 4)),
    ("Significance", (1, 4)),
    ("Soundness", (1, 4)),
    ("Presentation", (1, 4)),
    ("Contribution", (1, 4)),
    ("Overall", (1, 10)),
    ("Confidence", (1, 5)),
]


def ensemble_review_pdf(pdf_path):
    from ai_scientist.llm import create_client
    from ai_scientist.perform_llm_review import load_paper, perform_review, get_meta_review

    text = load_paper(pdf_path)
    votes, vote_models = [], []
    for i, m in enumerate(VOTERS):
        t0 = time.time()
        try:
            client, mname = create_client(m)
            r = perform_review(
                text, mname, client,
                num_reflections=1, num_fs_examples=1, num_reviews_ensemble=1,
            )
            if isinstance(r, dict):
                votes.append(r)
                vote_models.append(m)
                print(f"  vote {i + 1}/5 [{m}] Overall={r.get('Overall')} "
                      f"Decision={r.get('Decision')} ({time.time() - t0:.0f}s)")
            else:
                print(f"  vote {i + 1}/5 [{m}] returned non-dict, dropped")
        except Exception as e:
            print(f"  vote {i + 1}/5 [{m}] FAILED: {e}")

    meta = None
    if votes:
        try:
            client, mname = create_client(META_MODEL)
            meta = get_meta_review(mname, client, 0.1, votes)
        except Exception as e:
            print(f"  meta-review FAILED: {e}")

    agg = {}
    for key, (lo, hi) in NUMERIC:
        xs = [v[key] for v in votes
              if isinstance(v.get(key), (int, float)) and lo <= v[key] <= hi]
        if xs:
            agg[key] = {
                "mean": round(float(np.mean(xs)), 3),
                "std": round(float(np.std(xs)), 3),
                "n": len(xs),
                "votes": xs,
            }
    return {
        "pdf": osp.abspath(pdf_path),
        "voters": vote_models,
        "votes": votes,
        "meta_review": meta,
        "meta_model": META_MODEL,
        "aggregate": agg,
        "decisions": [v.get("Decision") for v in votes],
        "settings": {"num_reflections": 1, "num_fs_examples": 1,
                     "form": "neurips_conference"},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for pdf in args.pdf:
        stem = osp.splitext(osp.basename(pdf))[0]
        if args.tag:
            stem = f"{args.tag}__{stem}"
        out_path = osp.join(args.out, f"{stem}_ensemble.json")
        if osp.exists(out_path):
            print(f"== skip (exists): {out_path}")
            continue
        print(f"== ensemble review: {pdf}")
        try:
            result = ensemble_review_pdf(pdf)
        except Exception:
            traceback.print_exc()
            continue
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        ov = result["aggregate"].get("Overall", {})
        print(f"== DONE {stem}: Overall mean={ov.get('mean')} std={ov.get('std')} "
              f"votes={ov.get('votes')} decisions={result['decisions']}")


if __name__ == "__main__":
    main()
