#!/usr/bin/env python
"""Enhanced writeup driver with evidence distillation + semantic critique (P0b).

ICR-inspired pipeline that catches hallucination DURING generation instead of
only in post-hoc audit.  Chains:

  1. evidence_distiller  → structured evidence summary
  2. constrained writeup → standard writeup with evidence constraints injected
  3. tex-level critique  → extract claims from LaTeX, check vs evidence (no PDF)
  4. targeted fix        → surgical edits to ungrounded claims
  5. iterate 3-4         → until grounded rate >= threshold or max rounds
  6. compile + review    → final PDF

Produces a detailed log JSON for pipeline-comparison experiments.

Usage (from AI-Scientist-v2 repo root, PYTHONPATH includes it):
  python tools/critique_writeup.py --idea_dir experiments/<run> \
      [--model_writeup gpt-4o] [--model_critique gpt-5.5] \
      [--max_critique_rounds 3] [--grounded_threshold 0.90]

Compare results with standard pipeline (rescue_writeup.py) to measure:
  - initial grounding rate (before any post-hoc repair)
  - number of repair rounds needed
  - final grounding rate
  - quality score (ensemble review)
  - total token/time cost
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

# ---------------------------------------------------------------------------
# LLM helpers (same pattern as other tools/)
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
            print(f"[cw] responses-API failed ({e}); chat fallback")
    from ai_scientist.llm import create_client, get_response_from_llm
    client, mname = create_client(model)
    resp, _ = get_response_from_llm(
        prompt, client=client, model=mname, system_message=system, print_debug=False)
    return resp


def _try_parse_array(text):
    start = text.find("[")
    if start == -1:
        return None
    dec = json.JSONDecoder()
    try:
        obj, _ = dec.raw_decode(text[start:])
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass
    # try closing truncated brackets
    fragment = text[start:].rstrip().rstrip(",")
    opens_b = fragment.count("[") - fragment.count("]")
    opens = fragment.count("{") - fragment.count("}")
    if opens > 0 or opens_b > 0:
        patched = fragment + "}" * max(0, opens) + "]" * max(0, opens_b)
        try:
            obj = json.loads(patched)
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def parse_json_array(text):
    patterns = [
        re.findall(r"```json\s*(.*?)```", text or "", re.DOTALL),
        re.findall(r"```\s*(.*?)```", text or "", re.DOTALL),
        [text or ""],
    ]
    for candidates in patterns:
        for c in candidates:
            obj = _try_parse_array(c.strip())
            if obj is not None:
                return obj
    raise ValueError("no JSON array in response")

# ---------------------------------------------------------------------------
# Step 1: Evidence distillation (delegates to evidence_distiller module)
# ---------------------------------------------------------------------------

def run_evidence_distill(idea_dir, model):
    tools_dir = osp.dirname(osp.abspath(__file__))
    sys.path.insert(0, tools_dir)
    from evidence_distiller import distill
    return distill(idea_dir, model=model)

# ---------------------------------------------------------------------------
# Step 2: Inject evidence constraints into idea text
# ---------------------------------------------------------------------------

CONSTRAINT_HEADER = """
====================================================================
EVIDENCE CONSTRAINTS (auto-generated — every claim MUST be traceable
to one of the validated findings or key metrics below. Claims about
items in NOT_OBSERVED are FORBIDDEN.)
====================================================================
"""

def build_constraint_block(evidence):
    """Build a compact text block to append to idea.md before writeup."""
    ev = evidence["evidence"]
    parts = [CONSTRAINT_HEADER]

    parts.append("## VALIDATED FINDINGS (you may cite these):")
    for f in ev.get("validated_findings", []):
        if isinstance(f, dict):
            parts.append(f"- {f.get('finding', '')} [{f.get('evidence_source', '')}]")
        else:
            parts.append(f"- {f}")

    parts.append("\n## KEY METRICS (exact numbers — do not alter):")
    for m in ev.get("key_metrics", []):
        if isinstance(m, dict):
            parts.append(f"- {m.get('metric_name', '')}: {m.get('value', '')} "
                         f"[{m.get('source', '')}] ({m.get('context', '')})")
        else:
            parts.append(f"- {m}")

    parts.append("\n## NOT OBSERVED (DO NOT claim any of these):")
    for n in ev.get("not_observed", []):
        parts.append(f"- {n}" if isinstance(n, str) else f"- {json.dumps(n)}")

    parts.append("\n## WRITEUP RULES:")
    for c in ev.get("writeup_constraints", []):
        parts.append(f"- {c}" if isinstance(c, str) else f"- {json.dumps(c)}")

    parts.append("\n## FAILED APPROACHES (mention honestly if relevant):")
    for f in ev.get("failed_approaches", []):
        if isinstance(f, dict):
            parts.append(f"- {f.get('approach', '')}: {f.get('why_failed', '')}")
        else:
            parts.append(f"- {f}")

    return "\n".join(parts)

# ---------------------------------------------------------------------------
# Step 3: Tex-level claim extraction + evidence check (fast, no PDF needed)
# ---------------------------------------------------------------------------

CLAIM_EXTRACT_SYS = "You are a meticulous research-integrity auditor."

CLAIM_EXTRACT_TEX_PROMPT = """Below is the LaTeX source of a machine-generated
research paper (NOT the PDF — raw .tex).

Extract EVERY empirical claim the paper makes about its own experiments:
- reported numbers (accuracies, losses, deltas, percentages, counts)
- dataset usage claims
- experimental-setup claims
- comparative conclusions
- figure/table content claims
- headline findings in abstract/intro/conclusion

Do NOT include citations of other papers or general background.

Return a JSON array inside a ```json fence. Each element:
{{"id": <int>, "category": "number|dataset|setup|comparison|figure|headline",
 "statement": "<claim, close to verbatim from the tex>",
 "section": "<approx where>"}}

LATEX SOURCE:
```latex
{tex}
```"""

EVIDENCE_CHECK_SYS = """You are a strict research-integrity auditor. You check
claims against a structured evidence summary. A claim is "grounded" only if
the evidence clearly supports it; numerical claims must match exactly (small
rounding OK). "contradicted" if the evidence shows different values.
"unsupported" if the evidence says nothing about it."""

EVIDENCE_CHECK_PROMPT = """STRUCTURED EVIDENCE SUMMARY (ground truth):
```json
{evidence}
```

CLAIMS TO CHECK:
```json
{claims}
```

For EACH claim return a verdict. Return a JSON array inside a ```json fence:
{{"id": <claim id>, "verdict": "grounded|unsupported|contradicted",
 "evidence_pointer": "<which finding/metric/constraint>",
 "note": "<1-2 sentence justification>"}}"""

def critique_tex(tex, evidence, model):
    """Extract claims from LaTeX source and check against evidence summary.
    Returns (claims_list, verdicts_list, grounded_ratio)."""
    for attempt in range(3):
        resp = llm_call(
            CLAIM_EXTRACT_TEX_PROMPT.format(tex=tex[:80000]),
            CLAIM_EXTRACT_SYS, model)
        try:
            claims = parse_json_array(resp)
            break
        except ValueError:
            if attempt < 2:
                print(f"[cw] claim extraction parse failed, retry {attempt+1}")
                continue
            print("[cw] claim extraction failed after 3 attempts, returning empty")
            return [], [], 0.0
    print(f"[cw] extracted {len(claims)} claims from tex")

    ev_compact = {k: v for k, v in evidence["evidence"].items()
                  if k in ("validated_findings", "key_metrics", "not_observed",
                           "experimental_setup", "failed_approaches",
                           "writeup_constraints")}

    verdicts = []
    BATCH = 10
    for i in range(0, len(claims), BATCH):
        batch = claims[i:i+BATCH]
        try:
            resp = llm_call(
                EVIDENCE_CHECK_PROMPT.format(
                    evidence=json.dumps(ev_compact, indent=1)[:40000],
                    claims=json.dumps(batch, indent=1)),
                EVIDENCE_CHECK_SYS, model)
            verdicts.extend(parse_json_array(resp))
        except Exception as e:
            print(f"[cw] critique batch failed: {e}")
            for c in batch:
                verdicts.append({"id": c.get("id"), "verdict": "audit_error",
                                 "note": str(e)[:200]})

    n = len(claims) or 1
    counts = {}
    for v in verdicts:
        counts[v.get("verdict", "unknown")] = counts.get(v.get("verdict", "unknown"), 0) + 1
    grounded = counts.get("grounded", 0)
    ratio = round(grounded / n, 3)

    vmap = {v.get("id"): v for v in verdicts}
    merged = []
    for c in claims:
        v = vmap.get(c.get("id"), {"verdict": "missing"})
        merged.append({**c, "audit_verdict": v.get("verdict"),
                       "audit_note": v.get("note", ""),
                       "audit_evidence_pointer": v.get("evidence_pointer", "")})

    return merged, counts, ratio

# ---------------------------------------------------------------------------
# Step 4: Targeted fix for ungrounded claims
# ---------------------------------------------------------------------------

FIX_SYS = """You are a research-integrity editor. You fix CONTRADICTED claims in
a paper's LaTeX source using ONLY the provided evidence. Never invent results.

CRITICAL RULES:
- For contradicted claims: correct the numbers/statements to match the evidence.
- Do NOT delete unsupported claims — they may be reasonable inferences.
  Only fix claims that DIRECTLY CONFLICT with the evidence.
- Keep LaTeX compilable. Do not leave orphan fragments or empty sections."""

FIX_PROMPT = """CONTRADICTED CLAIMS (these conflict with the evidence and must be corrected):
```json
{bad_claims}
```

EVIDENCE SUMMARY (ground truth):
```json
{evidence}
```

CURRENT LATEX:
```latex
{tex}
```

Produce surgical text edits to fix the CONTRADICTED claims only. RULES:
1. The "find" string must be an EXACT, VERBATIM copy from the LaTeX above —
   copy-paste, do not retype. Include at least 40 chars of surrounding context
   so the match is unique.
2. The "replace" string must NOT introduce any new empirical claims that are not
   in the evidence summary.
3. Correct the numbers/statements to match the evidence exactly.
4. Do NOT delete sentences unless they are entirely wrong. Prefer correcting
   over removing. Preserve the narrative flow.

Return a JSON array inside a ```json fence:
{{"find": "<EXACT verbatim substring, 40+ chars for uniqueness>",
 "replace": "<corrected text>"}}"""


def _locate_claim_in_tex(tex, statement, context_chars=200):
    """Find the approximate location of a claim in the tex and return context."""
    words = re.findall(r'\w+', statement.lower())
    key_words = [w for w in words if len(w) > 4][:5]
    if not key_words:
        return None
    best_pos, best_count = -1, 0
    for i in range(0, len(tex) - 100, 50):
        chunk = tex[i:i+300].lower()
        hits = sum(1 for w in key_words if w in chunk)
        if hits > best_count:
            best_count = hits
            best_pos = i
    if best_count >= 2 and best_pos >= 0:
        start = max(0, best_pos - 50)
        end = min(len(tex), best_pos + context_chars + 50)
        return tex[start:end]
    return None


def _fuzzy_find(tex, find_str, threshold=0.7):
    """Try to locate find_str in tex even if not an exact match.
    Returns (start, end) of best match or None."""
    if find_str in tex:
        idx = tex.index(find_str)
        return idx, idx + len(find_str)
    stripped = find_str.strip()
    if stripped in tex:
        idx = tex.index(stripped)
        return idx, idx + len(stripped)
    norm_find = re.sub(r'\s+', ' ', find_str.strip())
    norm_tex = re.sub(r'\s+', ' ', tex)
    if norm_find in norm_tex:
        pos = norm_tex.index(norm_find)
        orig_pos = 0
        norm_i = 0
        while norm_i < pos and orig_pos < len(tex):
            orig_pos += 1
            norm_i = len(re.sub(r'\s+', ' ', tex[:orig_pos]))
        end_pos = orig_pos
        while norm_i < pos + len(norm_find) and end_pos < len(tex):
            end_pos += 1
            norm_i = len(re.sub(r'\s+', ' ', tex[:end_pos]))
        return orig_pos, end_pos
    return None


MIN_FIXABLE_RATIO = 0.15

def fix_tex(tex, bad_claims, evidence, model):
    """Apply surgical edits to fix contradicted claims. Returns (new_tex, n_applied).

    Only corrects claims that CONFLICT with evidence. Unsupported claims are
    left alone — they may be reasonable inferences and deleting them damages
    narrative coherence (validated by A/B tournament: critique-only 30% WR vs
    rewrite 35% WR, with content deletion being the primary cause of losses).
    """
    contradicted = [c for c in bad_claims if c.get("audit_verdict") == "contradicted"]
    unsupported = [c for c in bad_claims if c.get("audit_verdict") == "unsupported"]
    if unsupported:
        print(f"[cw] skipping {len(unsupported)} unsupported claims (preserve content)")
    bad_claims = contradicted
    if not bad_claims:
        print("[cw] no contradicted claims to fix")
        return tex, 0

    slim = []
    for c in bad_claims:
        entry = {"statement": c.get("statement"), "verdict": c.get("audit_verdict"),
                 "note": c.get("audit_note")}
        ctx = _locate_claim_in_tex(tex, c.get("statement", ""))
        if ctx:
            entry["tex_context"] = ctx
        slim.append(entry)

    ev_compact = {k: v for k, v in evidence["evidence"].items()
                  if k in ("validated_findings", "key_metrics",
                           "experimental_setup", "not_observed")}

    resp = llm_call(
        FIX_PROMPT.format(bad_claims=json.dumps(slim, indent=1),
                          evidence=json.dumps(ev_compact, indent=1)[:30000],
                          tex=tex),
        FIX_SYS, model)

    try:
        edits = [e for e in parse_json_array(resp)
                 if isinstance(e, dict) and e.get("find")]
    except Exception as e:
        print(f"[cw] fix parse failed: {e}")
        return tex, 0

    new_tex, applied, skipped = tex, 0, 0
    for e in edits:
        find_str = e["find"]
        replace_str = e.get("replace", "")
        if find_str in new_tex:
            new_tex = new_tex.replace(find_str, replace_str, 1)
            applied += 1
        else:
            match = _fuzzy_find(new_tex, find_str)
            if match:
                start, end = match
                new_tex = new_tex[:start] + replace_str + new_tex[end:]
                applied += 1
                print(f"[cw]   fuzzy-matched edit (len {len(find_str)})")
            else:
                skipped += 1
                print(f"[cw]   SKIP edit — no match: {find_str[:60]}...")
    if skipped:
        print(f"[cw] {applied} edits applied, {skipped} skipped (no match)")
    return new_tex, applied

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def find_latest_pdf(idea_dir):
    """Find the latest reflection PDF in the idea directory."""
    pdf_files = [f for f in os.listdir(idea_dir)
                 if f.endswith(".pdf") and "reflection" in f]
    if not pdf_files:
        pdf_files = [f for f in os.listdir(idea_dir) if f.endswith(".pdf")]
    if not pdf_files:
        return None
    nums = []
    for f in pdf_files:
        m = re.search(r"reflection[_.]?(\d+)", f)
        if m:
            nums.append((int(m.group(1)), f))
    if nums:
        return osp.join(idea_dir, max(nums)[1])
    return osp.join(idea_dir, pdf_files[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--idea_dir", required=True,
                   help="experiment dir relative to repo root")
    p.add_argument("--model_distill", default="gpt-5.5",
                   help="model for evidence distillation")
    p.add_argument("--model_writeup", default="gpt-4o",
                   help="model for writeup generation")
    p.add_argument("--model_critique", default="gpt-5.5",
                   help="model for critique + fix")
    p.add_argument("--model_review", default="gpt-4o",
                   help="model for final review")
    p.add_argument("--max_critique_rounds", type=int, default=5)
    p.add_argument("--grounded_threshold", type=float, default=0.90)
    p.add_argument("--num_cite_rounds", type=int, default=5)
    p.add_argument("--writeup_retries", type=int, default=2)
    p.add_argument("--skip_writeup", action="store_true",
                   help="use existing tex (skip writeup, only critique loop)")
    p.add_argument("--skip_agg", action="store_true")
    p.add_argument("--out_log", default=None,
                   help="output log JSON path (default: <idea_dir>/critique_log.json)")
    args = p.parse_args()

    assert osp.isdir("ai_scientist/blank_icbinb_latex"), \
        "cwd must be an AI-Scientist-v2 repo root"
    idea_dir = args.idea_dir.rstrip("/")
    run_dir = osp.join(idea_dir, "logs", "0-run")
    latex_dir = osp.join(idea_dir, "latex")
    tex_path = osp.join(latex_dir, "template.tex")
    base = osp.basename(idea_dir)
    log_path = args.out_log or osp.join(idea_dir, "critique_log.json")
    log = {"experiment": idea_dir, "pipeline": "critique_writeup_v1",
           "args": vars(args), "steps": []}

    def log_step(name, **kw):
        entry = {"step": name, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}
        log["steps"].append(entry)
        print(f"[cw] === {name} === {json.dumps({k: v for k, v in kw.items() if k != 'details'})}")
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

    # ---- Step 1: Evidence distillation ----
    t0 = time.time()
    evidence = run_evidence_distill(idea_dir, model=args.model_distill)
    log_step("evidence_distill",
             duration_s=round(time.time() - t0, 1),
             n_validated=evidence["meta"]["n_validated_findings"],
             n_not_observed=evidence["meta"]["n_not_observed"],
             n_constraints=evidence["meta"]["n_writeup_constraints"])

    ev_path = osp.join(idea_dir, "evidence_distilled.json")
    with open(ev_path, "w") as f:
        json.dump(evidence, f, indent=2)

    # ---- Step 2: Inject constraints into idea.md ----
    idea_md = osp.join(idea_dir, "idea.md")
    idea_backup = idea_md + ".pre_critique"
    if osp.exists(idea_md):
        shutil.copy(idea_md, idea_backup)
        constraint_block = build_constraint_block(evidence)
        with open(idea_md, "a") as f:
            f.write("\n" + constraint_block)
        print(f"[cw] injected {len(constraint_block)} chars of constraints into idea.md")

    # ---- Step 3: Run writeup (standard pipeline) ----
    if not args.skip_writeup:
        t1 = time.time()
        try:
            from omegaconf import OmegaConf

            if not args.skip_agg:
                src = osp.join(run_dir, "experiment_results")
                dst = osp.join(idea_dir, "experiment_results")
                if osp.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                from ai_scientist.perform_plotting import aggregate_plots
                aggregate_plots(base_folder=idea_dir, model=args.model_writeup)
                if osp.isdir(dst):
                    shutil.rmtree(dst)

            import ai_scientist.perform_icbinb_writeup as icb
            citations_text = icb.gather_citations(
                idea_dir, num_cite_rounds=args.num_cite_rounds,
                small_model=args.model_writeup)

            writeup_fn = getattr(icb, "perform_writeup", None) or \
                         getattr(icb, "perform_icbinb_writeup")
            success = False
            for attempt in range(args.writeup_retries):
                print(f"[cw] writeup attempt {attempt+1}/{args.writeup_retries}")
                try:
                    success = writeup_fn(
                        base_folder=idea_dir, small_model=args.model_writeup,
                        big_model=args.model_writeup, page_limit=4,
                        citations_text=citations_text)
                except Exception:
                    traceback.print_exc()
                if success:
                    break

            log_step("writeup", duration_s=round(time.time() - t1, 1),
                     success=success, model=args.model_writeup)
        finally:
            if osp.exists(idea_backup):
                shutil.copy(idea_backup, idea_md)
                print("[cw] restored original idea.md")

    # ---- Step 4: Critique loop on tex ----
    assert osp.exists(tex_path), f"no tex at {tex_path}"
    with open(tex_path) as f:
        tex = f.read()
    shutil.copy(tex_path, tex_path + ".pre_critique")

    from ai_scientist.perform_icbinb_writeup import compile_latex

    for rnd in range(1, args.max_critique_rounds + 1):
        t2 = time.time()

        claims, counts, ratio = critique_tex(tex, evidence, args.model_critique)
        bad = [c for c in claims if c.get("audit_verdict") in ("unsupported", "contradicted")]

        crit_entry = {
            "round": rnd,
            "claims_total": len(claims),
            "verdict_counts": counts,
            "grounded_ratio": ratio,
            "n_bad": len(bad),
        }

        if ratio >= args.grounded_threshold:
            log_step(f"critique_round_{rnd}", duration_s=round(time.time()-t2, 1),
                     action="passed", **crit_entry)
            print(f"[cw] round {rnd}: grounded {ratio:.1%} >= threshold — done")
            break

        new_tex, applied = fix_tex(tex, bad, evidence, args.model_critique)
        crit_entry["edits_applied"] = applied
        log_step(f"critique_round_{rnd}", duration_s=round(time.time()-t2, 1),
                 action="fixed", **crit_entry)

        if applied == 0:
            print(f"[cw] round {rnd}: no edits applied, stopping")
            break
        tex = new_tex
        with open(tex_path, "w") as f:
            f.write(tex)
    else:
        print(f"[cw] max critique rounds ({args.max_critique_rounds}) reached")

    # ---- Step 5: Compile final PDF ----
    out_pdf = osp.join(idea_dir, f"{base}_critique_final.pdf")
    compile_latex(latex_dir, out_pdf)
    if not osp.exists(out_pdf):
        print("[cw] compile failed; trying compile fix")
        tools_dir = osp.dirname(osp.abspath(__file__))
        sys.path.insert(0, tools_dir)
        from reground_rewrite import ensure_compilable
        ensure_compilable(latex_dir, tex_path, idea_dir, base,
                          args.model_critique, compile_latex)
        compile_latex(latex_dir, out_pdf)

    pdf_ok = osp.exists(out_pdf)
    log_step("compile", pdf_produced=pdf_ok, pdf=out_pdf if pdf_ok else None)

    # ---- Step 6: Review (if PDF exists) ----
    if pdf_ok:
        t3 = time.time()
        try:
            from ai_scientist.llm import create_client
            from ai_scientist.perform_llm_review import load_paper, perform_review
            paper_content = load_paper(out_pdf)
            client, client_model = create_client(args.model_review)
            review = perform_review(paper_content, client_model, client)
            review_path = osp.join(idea_dir, f"{base}_critique_review.json")
            with open(review_path, "w") as f:
                json.dump(review, f, indent=4)
            overall = review.get("Overall") if isinstance(review, dict) else "?"
            log_step("review", duration_s=round(time.time()-t3, 1),
                     model=args.model_review, overall=overall)
        except Exception:
            traceback.print_exc()
            log_step("review", error=traceback.format_exc()[-500:])

    # ---- Summary ----
    total_time = sum(s.get("duration_s", 0) for s in log["steps"])
    final_ratio = 0
    for s in reversed(log["steps"]):
        if "grounded_ratio" in s:
            final_ratio = s["grounded_ratio"]
            break

    log["summary"] = {
        "total_duration_s": round(total_time, 1),
        "critique_rounds_used": sum(1 for s in log["steps"]
                                    if s["step"].startswith("critique_round")),
        "final_grounded_ratio": final_ratio,
        "pdf": out_pdf if pdf_ok else None,
    }
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n[cw] DONE — log at {log_path}")
    print(f"[cw] summary: {json.dumps(log['summary'])}")


if __name__ == "__main__":
    main()
