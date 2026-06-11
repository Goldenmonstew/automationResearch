#!/usr/bin/env python
"""Grounding audit: verify every empirical claim in a generated paper against
the run's actual experimental artifacts.

Pipeline:
  1. extract claims from the paper PDF (LLM -> structured JSON)
  2. build an evidence pack from the experiment dir (stage summaries, journal
     node metrics, figure inventory, experiment-data file inventory; no LLM)
  3. judge each claim against the evidence pack (LLM, batched)
Output: <out>.json (machine) + <out>.md (human certificate table).

Audit tiers:
  - full      : experiment dir available (our runs) — claims checked vs logs
  - text-only : --no_exp_dir (e.g. external papers without public logs) —
                internal-consistency check only; explicitly marked shallower.

Run from an AI-Scientist-v2 repo root with PYTHONPATH set to it.
"""
import argparse
import json
import os
import os.path as osp
import re
import sys
import traceback

MAX_EVIDENCE_CHARS = 60000
CLAIMS_BATCH = 8

CLAIM_EXTRACT_SYS = """You are a meticulous research-integrity auditor."""

CLAIM_EXTRACT_PROMPT = """Below is the full text of a machine-generated research paper.

Extract EVERY empirical claim the paper makes about its own experiments. Include:
- reported numbers (accuracies, losses, deltas, percentages, counts)
- dataset usage claims (which datasets were used / trained on)
- experimental-setup claims (architectures, training procedures, seeds, splits)
- comparative conclusions ("X outperforms Y", "no significant difference")
- figure/table content claims (what a figure or table allegedly shows)
- headline findings in abstract/intro/conclusion

Do NOT include citations of other papers or general background statements.

Return a JSON array inside a ```json fence. Each element:
{{"id": <int>, "category": "number|dataset|setup|comparison|figure|table|headline",
 "statement": "<claim, close to verbatim>", "section": "<where it appears>"}}

PAPER TEXT:
```
{paper_text}
```"""

JUDGE_SYS = """You are a strict research-integrity auditor. You decide whether
claims in a machine-generated paper are supported by the actual experiment
artifacts. Be conservative: a claim is "grounded" only if the evidence pack
clearly supports it; numerical claims must match the logged values (small
rounding is fine). If the artifacts show different numbers/datasets/setups,
the claim is "contradicted". If the artifacts simply contain nothing about it,
it is "unsupported"."""

JUDGE_PROMPT = """EVIDENCE PACK (ground truth from the run's logs; audit tier: {tier}):
```
{evidence}
```

CLAIMS TO AUDIT:
```json
{claims}
```

For EACH claim return a verdict object. Return a JSON array inside a ```json fence:
{{"id": <claim id>, "verdict": "grounded|unsupported|contradicted",
 "evidence_pointer": "<which summary/node/figure/file supports or refutes it, or 'none'>",
 "note": "<1-2 sentence justification>"}}"""


def parse_json_array(text):
    m = re.findall(r"```json\s*(.*?)```", text, re.DOTALL)
    candidates = m if m else [text]
    for c in candidates:
        c = c.strip()
        start = c.find("[")
        if start == -1:
            continue
        dec = json.JSONDecoder()
        try:
            obj, _ = dec.raw_decode(c[start:])
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON array found in LLM response")


def llm(prompt, system, model):
    # Prefer /v1/responses with priority service tier (the only path where the
    # router honors it) + a 32k output budget; fall back to the chat path.
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
            print(f"[audit] responses-API failed ({e}); falling back to chat path")
    from ai_scientist.llm import create_client, get_response_from_llm
    client, mname = create_client(model)
    resp, _ = get_response_from_llm(
        prompt, client=client, model=mname, system_message=system, print_debug=False
    )
    return resp


def shorten(x, n):
    s = x if isinstance(x, str) else json.dumps(x, default=str)
    return s if len(s) <= n else s[:n] + f"...<truncated {len(s) - n} chars>"


def build_evidence_pack(idea_dir):
    """Compact, lossy-but-faithful dump of what the run actually did."""
    parts = []
    run_dir = osp.join(idea_dir, "logs", "0-run")

    idea_md = osp.join(idea_dir, "idea.md")
    if osp.exists(idea_md):
        parts.append("## RESEARCH IDEA (input)\n" + shorten(open(idea_md).read(), 3000))

    for name in ["draft", "baseline", "research", "ablation"]:
        p = osp.join(run_dir, f"{name}_summary.json")
        if osp.exists(p):
            try:
                data = json.load(open(p))
                parts.append(f"## {name.upper()} STAGE SUMMARY\n" + shorten(data, 14000))
            except json.JSONDecodeError:
                parts.append(f"## {name.upper()} STAGE SUMMARY\n<unparseable>")

    # journal node inventory: per stage, per node: buggy flag + metric + plan head
    node_lines = []
    if osp.isdir(run_dir):
        for d in sorted(os.listdir(run_dir)):
            jp = osp.join(run_dir, d, "journal.json")
            if not (d.startswith("stage_") and osp.exists(jp)):
                continue
            try:
                jd = json.load(open(jp))
            except json.JSONDecodeError:
                continue
            for nd in jd.get("nodes", []):
                metric = shorten(nd.get("metric", ""), 220)
                node_lines.append(
                    f"{d} | node {str(nd.get('id'))[:8]} | buggy={nd.get('is_buggy')} "
                    f"| metric={metric}"
                )
    if node_lines:
        parts.append("## TREE-SEARCH NODE INVENTORY (every executed node)\n"
                     + shorten("\n".join(node_lines), 18000))

    figdir = osp.join(idea_dir, "figures")
    if osp.isdir(figdir):
        figs = [f"{f} ({os.path.getsize(osp.join(figdir, f))} bytes)"
                for f in sorted(os.listdir(figdir))]
        parts.append("## AGGREGATED FIGURES ON DISK\n" + "\n".join(figs))

    er = osp.join(run_dir, "experiment_results")
    if osp.isdir(er):
        files = []
        for root, _, fnames in os.walk(er):
            for f in fnames:
                files.append(osp.relpath(osp.join(root, f), er))
        parts.append("## EXPERIMENT DATA FILES (names only)\n"
                     + shorten("\n".join(sorted(files)), 6000))

    pack = "\n\n".join(parts)
    return pack[:MAX_EVIDENCE_CHARS]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", required=True)
    p.add_argument("--exp_dir", default=None,
                   help="idea_dir of the run (relative to cwd)")
    p.add_argument("--no_exp_dir", action="store_true",
                   help="external paper: internal-consistency tier only")
    p.add_argument("--out", required=True, help="output path prefix (no extension)")
    p.add_argument("--model", default="gpt-5.5")
    args = p.parse_args()
    assert args.exp_dir or args.no_exp_dir, "need --exp_dir or --no_exp_dir"

    from ai_scientist.perform_llm_review import load_paper
    paper_text = load_paper(args.pdf)

    # ---- 1. claim extraction ----
    print("[audit] extracting claims ...")
    resp = llm(CLAIM_EXTRACT_PROMPT.format(paper_text=paper_text[:80000]),
               CLAIM_EXTRACT_SYS, args.model)
    claims = parse_json_array(resp)
    print(f"[audit] {len(claims)} claims extracted")

    # ---- 2. evidence pack ----
    if args.no_exp_dir:
        tier = "text-only (no experiment logs available; internal consistency only)"
        evidence = ("NO EXPERIMENT LOGS AVAILABLE. Judge only internal consistency: "
                    "do the paper's own numbers/tables/figures agree with each other "
                    "and with its stated setup? Mark externally unverifiable claims "
                    "as 'unsupported' with note 'no logs'.\n\nPAPER TEXT AGAIN:\n"
                    + paper_text[:50000])
    else:
        tier = "full (claims checked against run artifacts)"
        evidence = build_evidence_pack(args.exp_dir)
        print(f"[audit] evidence pack: {len(evidence)} chars")

    # ---- 3. batched judging ----
    verdicts = []
    for i in range(0, len(claims), CLAIMS_BATCH):
        batch = claims[i:i + CLAIMS_BATCH]
        print(f"[audit] judging claims {i + 1}-{i + len(batch)} ...")
        try:
            resp = llm(
                JUDGE_PROMPT.format(tier=tier, evidence=evidence,
                                    claims=json.dumps(batch, indent=1)),
                JUDGE_SYS, args.model)
            verdicts.extend(parse_json_array(resp))
        except Exception as e:
            print(f"[audit] batch failed: {e}")
            for c in batch:
                verdicts.append({"id": c.get("id"), "verdict": "audit_error",
                                 "evidence_pointer": "none", "note": str(e)[:200]})

    vmap = {v.get("id"): v for v in verdicts}
    rows = []
    counts = {}
    for c in claims:
        v = vmap.get(c.get("id"), {"verdict": "missing"})
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
        rows.append({**c, **{f"audit_{k}": v.get(k) for k in
                             ("verdict", "evidence_pointer", "note")}})

    n = len(claims) or 1
    summary = {
        "pdf": osp.abspath(args.pdf),
        "exp_dir": args.exp_dir,
        "tier": tier,
        "model": args.model,
        "n_claims": len(claims),
        "verdict_counts": counts,
        "grounded_ratio": round(counts.get("grounded", 0) / n, 3),
    }
    with open(args.out + ".json", "w") as f:
        json.dump({"summary": summary, "claims": rows}, f, indent=2)

    md = [f"# Grounding certificate — {osp.basename(args.pdf)}",
          f"- audit tier: {tier}",
          f"- model: {args.model}",
          f"- claims: {len(claims)}; verdicts: {json.dumps(counts)}",
          f"- grounded ratio: {summary['grounded_ratio']}",
          "",
          "| id | cat | claim | verdict | evidence |",
          "|---|---|---|---|---|"]
    for r in rows:
        md.append("| {} | {} | {} | **{}** | {} |".format(
            r.get("id"), r.get("category", ""),
            str(r.get("statement", "")).replace("|", "/")[:140],
            r.get("audit_verdict"),
            str(r.get("audit_evidence_pointer", "")).replace("|", "/")[:100]))
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(md))

    print(f"[audit] DONE {summary}")


if __name__ == "__main__":
    main()
