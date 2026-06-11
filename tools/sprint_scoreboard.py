#!/usr/bin/env python
"""Aggregate the sprint's review/audit artifacts into one G1/G2 scoreboard.

Scans <sprint>/reviews/*_ensemble.json (tags: calib/official/rescue/sprint/
regrounded) and <sprint>/audit_*.json, joins them per paper, and prints a
markdown table sorted by honest-version score. Also dumps scoreboard.json.

Usage: python sprint_scoreboard.py [--sprint_dir ~/automationResearch/sprint]
"""
import argparse
import glob
import json
import os
import os.path as osp
import re


def expname_of(path_or_name):
    m = re.search(r"(2026-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_[a-z0-9_]+_attempt_0)",
                  path_or_name)
    return m.group(1) if m else None


def overall_of(review_json):
    a = review_json.get("aggregate", {}).get("Overall", {})
    return a.get("mean"), a.get("votes")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sprint_dir",
                   default=osp.expanduser("~/automationResearch/sprint"))
    args = p.parse_args()
    sp = args.sprint_dir

    papers = {}   # expname -> record
    extras = []   # calib / official rows

    for f in sorted(glob.glob(osp.join(sp, "reviews", "*_ensemble.json"))):
        d = json.load(open(f))
        base = osp.basename(f)
        tag = base.split("__", 1)[0]
        mean, votes = overall_of(d)
        exp = expname_of(base)
        if tag in ("calib", "official", "officialv2", "legacy") or not exp:
            extras.append({"tag": tag,
                           "name": base.split("__", 1)[1].replace(
                               "_ensemble.json", ""),
                           "overall": mean, "votes": votes})
            continue
        rec = papers.setdefault(exp, {"exp": exp, "source": None,
                                      "raw": None, "honest": None,
                                      "rewritten": None,
                                      "raw_grounded": None,
                                      "final_grounded": None})
        if tag == "regrounded":
            rec["honest"] = mean
            rec["honest_votes"] = votes
        elif tag == "rewritten":
            rec["rewritten"] = mean
            rec["rewritten_votes"] = votes
        else:
            rec["source"] = tag
            rec["raw"] = mean
            rec["raw_votes"] = votes

    for f in glob.glob(osp.join(sp, "audit_*.json")):
        d = json.load(open(f))
        s = d.get("summary", {})
        exp = expname_of(s.get("pdf", "") or "")
        if exp and exp in papers:
            papers[exp]["raw_grounded"] = s.get("grounded_ratio")
            papers[exp]["n_claims"] = s.get("n_claims")

    # final grounding state from per-experiment grounding_r*.json (search both
    # phase2 slots and sprint lanes)
    for exp, rec in papers.items():
        cands = glob.glob(osp.expanduser(
            f"~/automationResearch/*/*/AI-Scientist-v2/experiments/{exp}/grounding_r*.json"))
        if cands:
            last = max(cands, key=lambda x: x)
            s = json.load(open(last)).get("summary", {})
            rec["final_grounded"] = s.get("grounded_ratio")

    rows = sorted(papers.values(),
                  key=lambda r: (r.get("honest") or 0, r.get("raw") or 0),
                  reverse=True)

    print("## 候选论文(同一把 5 票尺;honest = 过 G2 gate;rewritten = writeup-v2)\n")
    print("| paper | src | raw | honest | rewritten | grounded raw→final |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        name = r["exp"][20:].replace("_attempt_0", "")[:48]
        fg = r.get("final_grounded")
        print(f"| {name} | {r.get('source') or '?'} | {r.get('raw')} "
              f"| {r.get('honest')} | {r.get('rewritten')} "
              f"| {r.get('raw_grounded')} → {fg if fg is not None else '…'} |")

    print("\n## 基线(官方 + 校准)\n")
    print("| paper | overall | votes |")
    print("|---|---|---|")
    for e in sorted(extras, key=lambda x: (x["tag"], -(x["overall"] or 0))):
        print(f"| {e['tag']}:{e['name'][:40]} | {e['overall']} | {e['votes']} |")

    out = {"papers": rows, "baselines": extras}
    with open(osp.join(sp, "scoreboard.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n(json: {osp.join(sp, 'scoreboard.json')})")


if __name__ == "__main__":
    main()
