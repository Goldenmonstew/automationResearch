#!/usr/bin/env python
"""Fault-injection calibration of the grounding auditor (PREREGISTRATION §4).

Takes a paper that already passed the gate (its final audit lists grounded
claims), plants known fabrications (mutated numbers / swapped datasets /
inverted conclusions), shuffles them with known-true claims, and blind-audits
the mix against the same evidence pack. Reports detection precision/recall and
false-positive rate, for the in-loop auditor (gpt-5.5) and a cross-family
auditor (deepseek-v3.2), plus inter-auditor agreement.

Run from the repo root of the run's working copy (PYTHONPATH set).

Usage:
  python audit_calibration.py --exp_dir experiments/<run> --audit <final>.json \
      [--n 20] [--out <prefix>]
"""
import argparse
import json
import os.path as osp
import random
import sys

MUTATE_SYS = """You fabricate plausible-but-false variants of scientific claims
for an audit calibration benchmark."""

MUTATE_PROMPT = """For EACH claim below, produce ONE mutated variant that is
FALSE with respect to the underlying experiment (change a number by 20-60%,
swap a dataset/setting name, or invert the direction of a conclusion), while
keeping the phrasing natural and plausible. Return a JSON array inside a
```json fence: [{{"id": <same id>, "statement": "<mutated claim>"}}].

CLAIMS:
```json
{claims}
```"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--audit", required=True, help="final (clean) grounding JSON")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--out", required=True)
    p.add_argument("--judges", default="gpt-5.5,deepseek-v3.2")
    args = p.parse_args()

    sys.path.insert(0, ".")
    sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
    import grounding_audit as ga

    audit = json.load(open(args.audit))
    grounded = [c for c in audit["claims"] if c.get("audit_verdict") == "grounded"
                and c.get("category") in ("number", "comparison", "dataset",
                                          "headline", "setup")]
    rng = random.Random(20260610)
    rng.shuffle(grounded)
    true_claims = grounded[:args.n]
    to_mutate = grounded[args.n:args.n * 2] or grounded[:args.n]
    print(f"[calib] true={len(true_claims)} to_mutate={len(to_mutate)}")

    slim = [{"id": i, "statement": c["statement"]} for i, c in enumerate(to_mutate)]
    resp = ga.llm(MUTATE_PROMPT.format(claims=json.dumps(slim, indent=1)),
                  MUTATE_SYS, "gpt-5.5")
    mutated = {m["id"]: m["statement"] for m in ga.parse_json_array(resp)
               if isinstance(m, dict) and m.get("statement")}
    print(f"[calib] fabricated variants: {len(mutated)}")

    # build the blind test set: label True = genuinely grounded
    test = []
    for i, c in enumerate(true_claims):
        test.append({"id": 1000 + i, "category": c.get("category"),
                     "statement": c["statement"], "_label": True})
    for i, stmt in mutated.items():
        test.append({"id": 2000 + i, "category": to_mutate[i].get("category"),
                     "statement": stmt, "_label": False})
    rng.shuffle(test)

    evidence = ga.build_evidence_pack(args.exp_dir)
    tier = "full (claims checked against run artifacts)"

    results = {}
    verdicts_by_judge = {}
    for judge in args.judges.split(","):
        judge = judge.strip()
        verdicts = {}
        for i in range(0, len(test), ga.CLAIMS_BATCH):
            batch = [{k: v for k, v in c.items() if k != "_label"}
                     for c in test[i:i + ga.CLAIMS_BATCH]]
            try:
                resp = ga.llm(ga.JUDGE_PROMPT.format(
                    tier=tier, evidence=evidence,
                    claims=json.dumps(batch, indent=1)), ga.JUDGE_SYS, judge)
                for v in ga.parse_json_array(resp):
                    verdicts[v.get("id")] = v.get("verdict")
            except Exception as e:
                print(f"[calib] {judge} batch failed: {e}")
        verdicts_by_judge[judge] = verdicts

        tp = fn = fp = tn = miss = 0
        for c in test:
            v = verdicts.get(c["id"])
            if v is None:
                miss += 1
                continue
            flagged = v in ("unsupported", "contradicted")
            if c["_label"] is False:      # planted fabrication
                tp += flagged
                fn += (not flagged)
            else:                          # genuinely grounded
                fp += flagged
                tn += (not flagged)
        n_fab = sum(1 for c in test if not c["_label"])
        n_true = sum(1 for c in test if c["_label"])
        results[judge] = {
            "recall_fabrications": round(tp / n_fab, 3) if n_fab else None,
            "false_positive_rate": round(fp / n_true, 3) if n_true else None,
            "precision_flagging": round(tp / (tp + fp), 3) if (tp + fp) else None,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn, "missing": miss,
        }
        print(f"[calib] {judge}: {results[judge]}")

    # inter-judge agreement on the flag/no-flag decision
    judges = list(verdicts_by_judge)
    if len(judges) == 2:
        a, b = judges
        both = [c["id"] for c in test
                if c["id"] in verdicts_by_judge[a] and c["id"] in verdicts_by_judge[b]]
        fa = {i: verdicts_by_judge[a][i] in ("unsupported", "contradicted") for i in both}
        fb = {i: verdicts_by_judge[b][i] in ("unsupported", "contradicted") for i in both}
        agree = sum(fa[i] == fb[i] for i in both)
        po = agree / len(both) if both else 0
        pa = sum(fa.values()) / len(both) if both else 0
        pb = sum(fb.values()) / len(both) if both else 0
        pe = pa * pb + (1 - pa) * (1 - pb)
        kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
        results["agreement"] = {"n": len(both), "raw": round(po, 3),
                                "cohens_kappa": round(kappa, 3)}
        print(f"[calib] agreement: {results['agreement']}")

    with open(args.out + ".json", "w") as f:
        json.dump({"exp_dir": args.exp_dir, "n_true": len(true_claims),
                   "n_fabricated": len(mutated), "results": results,
                   "test_set": test, "verdicts": verdicts_by_judge}, f, indent=2)
    print(f"[calib] DONE -> {args.out}.json")


if __name__ == "__main__":
    main()
