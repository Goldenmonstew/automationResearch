#!/usr/bin/env python
"""Blind pairwise preference tournament (PREREGISTRATION §1).

Every (ours, theirs) pair is judged by multiple model families, in BOTH
presentation orders (position de-biasing). Reports per-judge and overall
win-rates with Wilson binomial confidence intervals.

Run from an AI-Scientist-v2 repo root with PYTHONPATH set.

Usage:
  python pairwise_tournament.py --ours a.pdf --ours b.pdf \
      --theirs x.pdf --theirs y.pdf --out <prefix>
"""
import argparse
import itertools
import json
import math
import os.path as osp
import re
import sys
import time

JUDGE_SYS = """You are an experienced workshop reviewer. You compare two
anonymous short papers and decide which is the stronger contribution overall,
considering soundness, clarity, honesty of claims, and practical value of the
findings. Negative or null results presented rigorously are valuable."""

JUDGE_PROMPT = """Below are two anonymous workshop papers, A and B. Decide which
one is the STRONGER contribution overall. You must pick exactly one.

PAPER A:
```
{a}
```

PAPER B:
```
{b}
```

Return JSON in a ```json fence: {{"winner": "A" or "B", "reason": "<2-3 sentences>"}}"""


def llm(prompt, system, model):
    if model.startswith("gpt-5"):
        try:
            import openai
            client = openai.OpenAI()
            r = client.responses.create(model=model, instructions=system,
                                        input=prompt, service_tier="priority",
                                        max_output_tokens=4096)
            txt = getattr(r, "output_text", None)
            if txt and txt.strip():
                return txt
        except Exception as e:
            print(f"[pt] responses path failed ({e}); chat fallback")
    from ai_scientist.llm import create_client, get_response_from_llm
    client, mname = create_client(model)
    resp, _ = get_response_from_llm(prompt, client=client, model=mname,
                                    system_message=system, print_debug=False)
    return resp


def parse_winner(text):
    m = re.search(r"```json\s*(.*?)```", text or "", re.DOTALL)
    blob = m.group(1) if m else (text or "")
    m2 = re.search(r'"winner"\s*:\s*"([AB])"', blob)
    return m2.group(1) if m2 else None


def wilson_ci(wins, n, z=1.96):
    if n == 0:
        return (None, None)
    p = wins / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(centre - half, 3), round(centre + half, 3))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ours", action="append", required=True)
    p.add_argument("--theirs", action="append", required=True)
    p.add_argument("--judges", default="gpt-5.5,gpt-4o,deepseek-v3.2")
    p.add_argument("--max_chars", type=int, default=24000)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    sys.path.insert(0, ".")
    from ai_scientist.perform_llm_review import load_paper

    texts = {}
    for f in args.ours + args.theirs:
        texts[f] = load_paper(f)[:args.max_chars]
        print(f"[pt] loaded {osp.basename(f)}: {len(texts[f])} chars")

    judges = [j.strip() for j in args.judges.split(",")]
    records = []
    for ours, theirs in itertools.product(args.ours, args.theirs):
        for judge in judges:
            for ours_is in ("A", "B"):
                a, b = (ours, theirs) if ours_is == "A" else (theirs, ours)
                t0 = time.time()
                try:
                    resp = llm(JUDGE_PROMPT.format(a=texts[a], b=texts[b]),
                               JUDGE_SYS, judge)
                    w = parse_winner(resp)
                except Exception as e:
                    print(f"[pt] call failed: {e}")
                    w = None
                ours_won = (w == ours_is) if w else None
                records.append({"ours": osp.basename(ours),
                                "theirs": osp.basename(theirs),
                                "judge": judge, "ours_position": ours_is,
                                "winner_letter": w, "ours_won": ours_won})
                print(f"[pt] {osp.basename(ours)[:28]} vs "
                      f"{osp.basename(theirs)[:18]} [{judge}/{ours_is}] -> "
                      f"ours_won={ours_won} ({time.time() - t0:.0f}s)")

    def rate(rs):
        valid = [r for r in rs if r["ours_won"] is not None]
        wins = sum(r["ours_won"] for r in valid)
        lo, hi = wilson_ci(wins, len(valid))
        return {"wins": wins, "n": len(valid),
                "win_rate": round(wins / len(valid), 3) if valid else None,
                "wilson95": [lo, hi]}

    summary = {"overall": rate(records)}
    for judge in judges:
        summary[f"judge:{judge}"] = rate([r for r in records if r["judge"] == judge])
    for theirs in args.theirs:
        b = osp.basename(theirs)
        summary[f"vs:{b}"] = rate([r for r in records if r["theirs"] == b])

    with open(args.out + ".json", "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    print("[pt] SUMMARY:", json.dumps(summary, indent=1))
    print(f"[pt] DONE -> {args.out}.json")


if __name__ == "__main__":
    main()
