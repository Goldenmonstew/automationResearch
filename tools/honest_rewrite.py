#!/usr/bin/env python
"""Honest rewrite (writeup-v2), per the frozen protocol in PREREGISTRATION.md §5.

Rewrites a gated paper completely — clean narrative instead of surgical
deletions — using ONLY machine artifacts as input: the audited grounded-claim
list, stage summaries, idea text, figure inventory and the existing
bibliography. The output must re-pass the same grounding gate (run separately
via grounding_audit + reground_rewrite --suffix rewritten).

Run from the repo root of the run's working copy (PYTHONPATH set to it).

Usage:
  python honest_rewrite.py --exp_dir experiments/<run> --audit <final_grounding>.json
"""
import argparse
import json
import os.path as osp
import re
import shutil
import sys

# ---- GLOBAL PROMPT — frozen before batch application (PREREGISTRATION §5) ----
SYS = """You are an expert scientific writer producing honest, clear papers
about negative, null or unexpected results. You never state an empirical
result that is not present in the provided grounded materials."""

PROMPT = """Rewrite the following machine-generated workshop paper COMPLETELY,
as a clean, well-told 4-page ICBINB-style paper (negative / unexpected
results track). The current version was produced by surgically deleting
unsupported claims, which left the narrative choppy; your job is a coherent
honest retelling, NOT new science.

HARD CONSTRAINTS:
1. Every empirical statement must be supported by the GROUNDED CLAIMS list or
   the EXPERIMENT SUMMARIES below. Do not introduce any new numbers, datasets,
   baselines or experiments. When materials are thin, say less, clearly.
2. Embrace the negative/unexpected-results framing: state crisply what was
   hypothesized, what was actually run, what was (not) found, and why that is
   informative for practitioners.
3. Keep the LaTeX preamble of the current document EXACTLY as is; only rewrite
   content from \\title{{...}} (you may improve the title) through the end of
   the conclusion. Keep \\bibliography as is.
4. Cite only keys that appear in AVAILABLE BIBTEX KEYS.
5. Reference figures only by the exact filenames in AVAILABLE FIGURES, with
   informative captions; you may drop figures that do not support the story.
6. Aim for a full 4 pages of substantive, honest content.

RESEARCH IDEA:
```
{idea}
```

GROUNDED CLAIMS (verified against run logs — your factual universe):
```json
{claims}
```

EXPERIMENT SUMMARIES (machine-generated from the run's journals):
```json
{summaries}
```

AVAILABLE FIGURES: {figures}

AVAILABLE BIBTEX KEYS: {bibkeys}

CURRENT LATEX:
```latex
{tex}
```

Return the complete rewritten LaTeX document in a ```latex fence."""


def llm(prompt, system, model):
    try:
        import openai
        client = openai.OpenAI()
        r = client.responses.create(model=model, instructions=system,
                                    input=prompt, service_tier="priority",
                                    max_output_tokens=32768)
        txt = getattr(r, "output_text", None)
        print(f"[rewrite] responses-API tier={getattr(r, 'service_tier', '?')} "
              f"len={len(txt) if txt else 0}")
        if txt and txt.strip():
            return txt
    except Exception as e:
        print(f"[rewrite] responses path failed ({e}); chat fallback")
    from ai_scientist.llm import create_client, get_response_from_llm
    client, mname = create_client(model)
    resp, _ = get_response_from_llm(prompt, client=client, model=mname,
                                    system_message=system, print_debug=False)
    return resp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--audit", required=True,
                   help="final grounding JSON of the gated version")
    p.add_argument("--model", default="gpt-5.5")
    args = p.parse_args()

    assert osp.isdir("ai_scientist"), "cwd must be repo root"
    sys.path.insert(0, ".")
    from ai_scientist.perform_icbinb_writeup import (
        compile_latex, _extract_latex_block, load_idea_text, load_exp_summaries,
        filter_experiment_summaries)

    exp_dir = args.exp_dir.rstrip("/")
    base = osp.basename(exp_dir)
    latex_dir = osp.join(exp_dir, "latex")
    tex_path = osp.join(latex_dir, "template.tex")
    with open(tex_path) as f:
        tex = f.read()
    shutil.copy(tex_path, tex_path + ".pre_rewrite")

    audit = json.load(open(args.audit))
    grounded = [{"statement": c.get("statement"), "category": c.get("category")}
                for c in audit["claims"]
                if c.get("audit_verdict") == "grounded"]
    print(f"[rewrite] grounded claims: {len(grounded)}")

    idea = load_idea_text(exp_dir)[:3000]
    summaries = filter_experiment_summaries(load_exp_summaries(exp_dir),
                                            step_name="writeup")
    summaries_str = json.dumps(summaries, indent=1)[:30000]

    figdir = osp.join(exp_dir, "figures")
    import os
    figures = sorted(os.listdir(figdir)) if osp.isdir(figdir) else []
    bib = ""
    for cand in ("references.bib", "iclr2025.bib"):
        bp = osp.join(latex_dir, cand)
        if osp.exists(bp):
            bib += open(bp, errors="ignore").read()
    bibkeys = sorted(set(re.findall(r"@\w+\{([^,\s]+)\s*,", bib)))[:80]

    prompt = PROMPT.format(idea=idea, claims=json.dumps(grounded, indent=1),
                           summaries=summaries_str, figures=figures,
                           bibkeys=bibkeys, tex=tex)

    new_tex = None
    for attempt in range(2):
        resp = llm(prompt, SYS, args.model)
        new_tex = _extract_latex_block(resp)
        if new_tex and "\\begin{document}" in new_tex and "\\end{document}" in new_tex:
            break
        m = re.search(r"(\\documentclass.*\\end\{document\})", resp or "", re.DOTALL)
        if m:
            new_tex = m.group(1)
            break
        print(f"[rewrite] extraction failed attempt {attempt + 1} "
              f"(len={len(resp) if resp else 0})")
        new_tex = None
    if not new_tex:
        print("[rewrite] FINAL: failed to obtain rewritten LaTeX")
        sys.exit(1)

    with open(tex_path, "w") as f:
        f.write(new_tex)
    out_pdf = osp.join(exp_dir, f"{base}_rewritten_r0.pdf")
    compile_latex(latex_dir, out_pdf)
    if not osp.exists(out_pdf):
        print("[rewrite] compile failed; restoring pre_rewrite tex")
        shutil.copy(tex_path + ".pre_rewrite", tex_path)
        print("[rewrite] FINAL: compile-failed")
        sys.exit(2)
    print(f"[rewrite] FINAL: ok pdf={out_pdf}")


if __name__ == "__main__":
    main()
