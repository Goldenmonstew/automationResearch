#!/usr/bin/env python
"""Evidence Distiller — structured synthesis of experiment artifacts (P0a).

ICR-inspired pre-writeup step: compress tree-search outputs into a structured
evidence summary with explicit boundaries.  Designed to constrain the writeup
agent and reduce hallucination by making "what was actually observed" and
"what was NOT observed" explicit.

Input:  experiment directory (tree search output with stage journals)
Output: JSON + markdown with structured categories

Usage (from AI-Scientist-v2 repo root, PYTHONPATH includes it):
  python tools/evidence_distiller.py --exp_dir experiments/<run> --out <prefix>
"""
import argparse
import json
import os
import os.path as osp
import re
import sys
import time

# ---------------------------------------------------------------------------
# Evidence gathering (adapted from grounding_audit.build_evidence_pack, but
# returns structured parts instead of a single string — the LLM synthesis
# step later decides how to compress).
# ---------------------------------------------------------------------------

def _shorten(x, n):
    s = x if isinstance(x, str) else json.dumps(x, default=str)
    return s if len(s) <= n else s[:n] + f"...<truncated {len(s)-n} chars>"


def gather_raw_evidence(idea_dir):
    """Return a dict of evidence sections, each a string ready for LLM input."""
    parts = {}
    run_dir = osp.join(idea_dir, "logs", "0-run")

    idea_md = osp.join(idea_dir, "idea.md")
    if osp.exists(idea_md):
        parts["idea"] = _shorten(open(idea_md).read(), 4000)

    for name in ["draft", "baseline", "research", "ablation"]:
        p = osp.join(run_dir, f"{name}_summary.json")
        if osp.exists(p):
            try:
                data = json.load(open(p))
                parts[f"stage_{name}"] = _shorten(data, 16000)
            except json.JSONDecodeError:
                parts[f"stage_{name}"] = "<unparseable JSON>"

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
                metric = _shorten(nd.get("metric", ""), 300)
                plan = _shorten(nd.get("plan", ""), 200)
                node_lines.append(
                    f"{d} | node {str(nd.get('id',''))[:8]} "
                    f"| buggy={nd.get('is_buggy')} "
                    f"| metric={metric} | plan={plan}")
    if node_lines:
        parts["node_inventory"] = _shorten("\n".join(node_lines), 20000)

    figdir = osp.join(idea_dir, "figures")
    if osp.isdir(figdir):
        figs = []
        for f in sorted(os.listdir(figdir)):
            sz = os.path.getsize(osp.join(figdir, f))
            figs.append(f"{f} ({sz} bytes)" + (" [SUSPECT: <500 bytes]" if sz < 500 else ""))
        parts["figures"] = "\n".join(figs)

    er = osp.join(run_dir, "experiment_results")
    if osp.isdir(er):
        files = []
        for root, _, fnames in os.walk(er):
            for f in fnames:
                fp = osp.join(root, f)
                files.append(f"{osp.relpath(fp, er)} ({os.path.getsize(fp)} bytes)")
        parts["data_files"] = _shorten("\n".join(sorted(files)), 8000)

    return parts

# ---------------------------------------------------------------------------
# LLM synthesis
# ---------------------------------------------------------------------------

DISTILL_SYS = """You are a meticulous scientific evidence curator.  You read
raw experiment logs and produce a STRUCTURED, BOUNDED evidence summary.
Your output will be used as the SOLE factual source for writing a research
paper — anything you omit will not appear in the paper, anything you include
will be cited.  Therefore:
- Include ONLY facts that appear in the evidence.
- Report EXACT numbers (do not round unless the source already rounded).
- Explicitly list what was NOT tested/measured.
- Flag ambiguities and contradictions between sources."""

DISTILL_PROMPT = """Below are the raw artifacts from an automated ML experiment
run (tree search with 4 stages: draft → baseline → creative → ablation).

Synthesize them into a structured evidence summary with EXACTLY these sections.
Return JSON inside a ```json fence with these top-level keys:

1. "experimental_setup": What was actually built and run. Datasets, model
   architectures, training procedures, hyperparameters, seeds, hardware.
   Only facts from the logs — no inference.

2. "validated_findings": Results with exact numbers that appear in the stage
   summaries or node metrics.  Each entry: {{"finding": "...", "evidence_source":
   "stage_X / node_Y", "exact_numbers": "..."}}

3. "failed_approaches": Approaches tried but abandoned (buggy nodes, negative
   results, dead-end branches).  Each: {{"approach": "...", "why_failed": "...",
   "evidence_source": "..."}}

4. "key_metrics": The most important quantitative results, with exact values
   and which node/stage produced them.  Each: {{"metric_name": "...",
   "value": "...", "source": "...", "context": "baseline/improvement/ablation"}}

5. "unexplained_observations": Anomalies or surprising patterns in the data
   that the experiment did not explain.  (Empty list is fine if none.)

6. "not_observed": Things the experiment did NOT test or measure, but a paper
   MIGHT be tempted to claim.  Be thorough — this section prevents hallucination.
   E.g. "No evaluation on dataset X was performed", "No comparison with method Y
   was run", "Generalization to Z was not tested".

7. "figure_inventory": Which figures exist, what each one appears to show,
   and which are suspect (empty/tiny).

8. "writeup_constraints": Direct rules for a writeup agent, derived from the
   above.  E.g. "DO NOT claim results on dataset X — only Y was used",
   "The best accuracy is A% not B%", "Only N seeds were run, do not claim
   statistical significance without caveat".

RAW EVIDENCE:

## RESEARCH IDEA
{idea}

## STAGE SUMMARIES
{stages}

## TREE-SEARCH NODE INVENTORY
{nodes}

## AGGREGATED FIGURES ON DISK
{figures}

## EXPERIMENT DATA FILES
{data_files}
"""


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
            print(f"[distill] responses-API failed ({e}); falling back to chat")
    from ai_scientist.llm import create_client, get_response_from_llm
    client, mname = create_client(model)
    resp, _ = get_response_from_llm(
        prompt, client=client, model=mname, system_message=system, print_debug=False)
    return resp


def _try_parse_json(text):
    start = text.find("{")
    if start == -1:
        return None
    dec = json.JSONDecoder()
    try:
        obj, _ = dec.raw_decode(text[start:])
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # try fixing truncated JSON by closing open braces/brackets
    fragment = text[start:]
    opens = fragment.count("{") - fragment.count("}")
    opens_b = fragment.count("[") - fragment.count("]")
    if opens > 0 or opens_b > 0:
        patched = fragment.rstrip().rstrip(",")
        patched += "]" * max(0, opens_b) + "}" * max(0, opens)
        try:
            obj = json.loads(patched)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def parse_json_object(text):
    patterns = [
        re.findall(r"```json\s*(.*?)```", text, re.DOTALL),
        re.findall(r"```\s*(.*?)```", text, re.DOTALL),
        [text],
    ]
    for candidates in patterns:
        for c in candidates:
            obj = _try_parse_json(c.strip())
            if obj:
                return obj
    raise ValueError(f"no JSON object found in LLM response (len={len(text)})")


EXPECTED_KEYS = [
    "experimental_setup", "validated_findings", "failed_approaches",
    "key_metrics", "unexplained_observations", "not_observed",
    "figure_inventory", "writeup_constraints",
]


def distill(idea_dir, model="gpt-5.5"):
    t0 = time.time()
    raw = gather_raw_evidence(idea_dir)
    gather_time = time.time() - t0
    print(f"[distill] gathered {len(raw)} evidence sections "
          f"({sum(len(v) for v in raw.values())} chars) in {gather_time:.1f}s")

    stages_text = ""
    for k in ["stage_draft", "stage_baseline", "stage_research", "stage_ablation"]:
        if k in raw:
            stages_text += f"\n### {k.upper()}\n{raw[k]}\n"

    prompt = DISTILL_PROMPT.format(
        idea=raw.get("idea", "(no idea text found)"),
        stages=stages_text or "(no stage summaries available)",
        nodes=raw.get("node_inventory", "(no node data)"),
        figures=raw.get("figures", "(no figures)"),
        data_files=raw.get("data_files", "(no data files)"),
    )

    t1 = time.time()
    evidence = None
    max_attempts = 3
    for attempt in range(max_attempts):
        resp = llm_call(prompt, DISTILL_SYS, model)
        llm_time = time.time() - t1
        print(f"[distill] LLM synthesis attempt {attempt+1} took {llm_time:.1f}s "
              f"(response {len(resp) if resp else 0} chars)")
        try:
            evidence = parse_json_object(resp)
            break
        except ValueError as e:
            is_last = (attempt == max_attempts - 1)
            print(f"[distill] parse failed ({e}); "
                  f"{'falling back to minimal evidence' if is_last else 'retrying'}")
            if is_last:
                evidence = {
                    "experimental_setup": "(LLM evidence synthesis failed after "
                                          f"{max_attempts} attempts — using raw data)",
                    "validated_findings": [],
                    "failed_approaches": [],
                    "key_metrics": [],
                    "unexplained_observations": [],
                    "not_observed": ["all claims — evidence distillation unavailable"],
                    "figure_inventory": [],
                    "writeup_constraints": [
                        "Evidence distillation failed; all claims must be independently verified"
                    ],
                }
    for k in EXPECTED_KEYS:
        if k not in evidence:
            evidence[k] = [] if k != "experimental_setup" else "(missing)"
            print(f"[distill] WARNING: missing key '{k}' in LLM output")

    meta = {
        "exp_dir": idea_dir,
        "model": model,
        "gather_time_s": round(gather_time, 1),
        "llm_time_s": round(llm_time, 1),
        "raw_evidence_chars": sum(len(v) for v in raw.values()),
        "n_validated_findings": len(evidence.get("validated_findings", [])),
        "n_failed_approaches": len(evidence.get("failed_approaches", [])),
        "n_key_metrics": len(evidence.get("key_metrics", [])),
        "n_not_observed": len(evidence.get("not_observed", [])),
        "n_writeup_constraints": len(evidence.get("writeup_constraints", [])),
    }
    return {"meta": meta, "evidence": evidence, "raw_sections": list(raw.keys())}


def render_markdown(result):
    ev = result["evidence"]
    lines = ["# Evidence Distillation Report",
             f"- experiment: {result['meta']['exp_dir']}",
             f"- model: {result['meta']['model']}",
             f"- validated findings: {result['meta']['n_validated_findings']}",
             f"- failed approaches: {result['meta']['n_failed_approaches']}",
             f"- key metrics: {result['meta']['n_key_metrics']}",
             f"- NOT observed: {result['meta']['n_not_observed']}",
             f"- writeup constraints: {result['meta']['n_writeup_constraints']}",
             ""]

    lines.append("## Experimental Setup")
    setup = ev.get("experimental_setup", "")
    lines.append(setup if isinstance(setup, str)
                 else json.dumps(setup, indent=2))
    lines.append("")

    for section, title in [
        ("validated_findings", "Validated Findings"),
        ("failed_approaches", "Failed Approaches"),
        ("key_metrics", "Key Metrics"),
        ("unexplained_observations", "Unexplained Observations"),
        ("not_observed", "NOT Observed (hallucination boundaries)"),
        ("figure_inventory", "Figure Inventory"),
        ("writeup_constraints", "Writeup Constraints"),
    ]:
        items = ev.get(section, [])
        lines.append(f"## {title}")
        if isinstance(items, list):
            for i, item in enumerate(items, 1):
                if isinstance(item, dict):
                    lines.append(f"{i}. " + " | ".join(
                        f"**{k}**: {v}" for k, v in item.items()))
                else:
                    lines.append(f"{i}. {item}")
        elif isinstance(items, str):
            lines.append(items)
        else:
            lines.append(json.dumps(items, indent=2))
        lines.append("")

    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", required=True,
                   help="idea dir relative to repo root")
    p.add_argument("--out", required=True, help="output path prefix (no ext)")
    p.add_argument("--model", default="gpt-5.5")
    args = p.parse_args()
    assert osp.isdir(args.exp_dir), f"missing {args.exp_dir}"

    result = distill(args.exp_dir, model=args.model)

    with open(args.out + ".json", "w") as f:
        json.dump(result, f, indent=2)
    with open(args.out + ".md", "w") as f:
        f.write(render_markdown(result))

    print(f"[distill] DONE → {args.out}.json / .md")
    print(f"[distill] {result['meta']}")


if __name__ == "__main__":
    main()
