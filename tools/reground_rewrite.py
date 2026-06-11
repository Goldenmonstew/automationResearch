#!/usr/bin/env python
"""Re-grounding rewrite loop (G2 gate).

Takes a grounding-audit report, rewrites the paper's LaTeX to remove or correct
every unsupported/contradicted claim (never adding new results), recompiles,
re-audits, and repeats until clean or max rounds. The final PDF plus its final
audit JSON form the paper's grounding certificate.

Run from an AI-Scientist-v2 repo root with PYTHONPATH set to it; the run dir
must contain latex/template.tex (left in place by the writeup step).

Usage:
  python reground_rewrite.py --exp_dir experiments/<run> --audit <audit>.json \
      --audit_script /path/to/grounding_audit.py [--model gpt-5.5] [--max_rounds 3]
"""
import argparse
import json
import os
import os.path as osp
import shutil
import subprocess
import sys

REWRITE_SYS = """You are a research-integrity editor. You revise machine-written
papers so that every empirical claim is supported by the actual experiment
logs. You NEVER invent results, numbers, datasets or experiments. When a claim
is unsupported you remove it or explicitly soften it to an honest statement of
what was actually run; when a claim contradicts the logs you correct it to the
logged value. Keep the paper compilable LaTeX and keep its overall structure."""

EDITS_PROMPT = """Below is the current LaTeX source of the paper and the list of
claims that a grounding audit found to be UNSUPPORTED or CONTRADICTED by the
run's actual experiment artifacts (with auditor notes).

Produce a list of surgical text edits that fix ALL the problem claims:
- a contradicted claim is corrected to match the logs (see auditor notes);
- an unsupported claim is removed, or rewritten as an honest limitation
  (e.g. "we did not evaluate X");
- never introduce new results, numbers, datasets or experiments;
- also fix abstract/intro/conclusion sentences that repeat those claims;
- edits must keep the document compilable (balanced braces/environments).

Return a JSON array inside a ```json fence. Each element:
{{"find": "<EXACT contiguous substring copied verbatim from the LaTeX below,
 long enough to be unique>", "replace": "<corrected LaTeX (may be empty to
 delete)>"}}

PROBLEM CLAIMS:
```json
{claims}
```

CURRENT LATEX:
```latex
{tex}
```"""


def parse_json_array_loose(text):
    import re as _re
    m = _re.findall(r"```json\s*(.*?)```", text or "", _re.DOTALL)
    candidates = m if m else [text or ""]
    dec = json.JSONDecoder()
    for c in candidates:
        start = c.find("[")
        if start == -1:
            continue
        try:
            obj, _ = dec.raw_decode(c[start:].strip())
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON array in response")


def rewrite_llm(prompt, system, model):
    """Full-document rewrites need a big output budget: gpt-5.x reasoning eats
    into chat-completions max_tokens (MAX_NUM_TOKENS=16384) and truncates the
    LaTeX. Prefer the /v1/responses endpoint with priority service tier and a
    32k output budget; fall back to the standard llm.py path."""
    try:
        import openai
        client = openai.OpenAI()  # OPENAI_API_KEY / OPENAI_BASE_URL from env
        r = client.responses.create(
            model=model,
            instructions=system,
            input=prompt,
            service_tier="priority",
            max_output_tokens=32768,
        )
        txt = getattr(r, "output_text", None)
        print(f"[reground] responses-API ok: tier={getattr(r, 'service_tier', '?')} "
              f"len={len(txt) if txt else 0}")
        if txt and txt.strip():
            return txt
    except Exception as e:
        print(f"[reground] responses-API path failed ({e}); falling back to chat path")
    from ai_scientist.llm import create_client, get_response_from_llm
    client, mname = create_client(model)
    resp, _ = get_response_from_llm(prompt, client=client, model=mname,
                                    system_message=system, print_debug=False)
    print(f"[reground] chat path len={len(resp) if resp else 0}")
    return resp


COMPILE_FIX_PROMPT = """The LaTeX document below fails to compile. Fix ONLY the
compilation errors — do not change any content, numbers, claims or wording
beyond what the fix strictly requires.

COMPILER ERRORS (from the .log):
```
{errors}
```

CURRENT LATEX:
```latex
{tex}
```

Return a JSON array of surgical edits inside a ```json fence, each element:
{{"find": "<EXACT contiguous substring copied from the LaTeX>",
 "replace": "<fixed substring>"}}"""


def ensure_compilable(latex_dir, tex_path, exp_dir, base, model, compile_fn):
    """The writeup's last reflection sometimes leaves a template.tex that does
    not compile (the on-disk reflection PDF came from an earlier revision).
    Probe-compile and LLM-repair before attempting any regrounding edits."""
    probe = osp.join(exp_dir, f"{base}_compilecheck.pdf")
    for attempt in range(1, 4):
        if osp.exists(probe):
            os.remove(probe)
        compile_fn(latex_dir, probe)
        if osp.exists(probe):
            os.remove(probe)
            if attempt > 1:
                print(f"[reground] base tex repaired after {attempt - 1} fix round(s)")
            return True
        errors = ""
        logf = osp.join(latex_dir, "template.log")
        if osp.exists(logf):
            lines = open(logf, errors="ignore").read().splitlines()
            ctx = []
            for i, l in enumerate(lines):
                if l.startswith("!"):
                    ctx.extend(lines[max(0, i - 2):i + 4])
                    if len(ctx) > 40:
                        break
            errors = "\n".join(ctx) or "\n".join(lines[-40:])
        print(f"[reground] base tex does not compile (probe {attempt}); LLM repair")
        with open(tex_path) as f:
            tex = f.read()
        resp = rewrite_llm(COMPILE_FIX_PROMPT.format(errors=errors[:4000], tex=tex),
                           REWRITE_SYS, model)
        try:
            edits = [e for e in parse_json_array_loose(resp)
                     if isinstance(e, dict) and e.get("find")]
        except Exception as e:
            print(f"[reground] compile-fix parse failed: {e}")
            continue
        applied = 0
        for e in edits:
            if e["find"] in tex:
                tex = tex.replace(e["find"], e.get("replace", ""), 1)
                applied += 1
        print(f"[reground] compile-fix edits applied: {applied}/{len(edits)}")
        if applied:
            with open(tex_path, "w") as f:
                f.write(tex)
    return False


def run_audit(audit_script, pdf, exp_dir, out_prefix, model):
    cmd = [sys.executable, "-u", audit_script, "--pdf", pdf, "--exp_dir", exp_dir,
           "--out", out_prefix, "--model", model]
    r = subprocess.run(cmd, text=True, capture_output=True)
    print(r.stdout[-2000:])
    if r.returncode != 0:
        print(r.stderr[-2000:])
        raise RuntimeError(f"audit failed rc={r.returncode}")
    with open(out_prefix + ".json") as f:
        return json.load(f)


def bad_claims_of(audit):
    return [c for c in audit["claims"]
            if c.get("audit_verdict") in ("unsupported", "contradicted")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--audit", required=True, help="initial grounding audit JSON")
    p.add_argument("--audit_script", required=True)
    p.add_argument("--model", default="gpt-5.5")
    p.add_argument("--max_rounds", type=int, default=3)
    p.add_argument("--suffix", default="regrounded",
                   help="output PDF/audit name suffix (e.g. 'rewritten' for the writeup-v2 lineage)")
    args = p.parse_args()

    assert osp.isdir("ai_scientist"), "cwd must be an AI-Scientist-v2 repo root"
    from ai_scientist.llm import create_client, get_response_from_llm
    from ai_scientist.perform_icbinb_writeup import compile_latex, _extract_latex_block

    exp_dir = args.exp_dir.rstrip("/")
    latex_dir = osp.join(exp_dir, "latex")
    tex_path = osp.join(latex_dir, "template.tex")
    assert osp.exists(tex_path), f"missing {tex_path}"
    shutil.copy(tex_path, tex_path + ".pre_reground")

    with open(args.audit) as f:
        audit = json.load(f)

    base = osp.basename(exp_dir)
    if not ensure_compilable(latex_dir, tex_path, exp_dir, base, args.model,
                             compile_latex):
        print("[reground] WARNING base tex cannot be made compilable; aborting")
        return
    final_pdf = None
    for rnd in range(1, args.max_rounds + 1):
        bad = bad_claims_of(audit)
        print(f"[reground] round {rnd}: {len(bad)} problem claims "
              f"(of {audit['summary']['n_claims']})")
        if not bad:
            print("[reground] all claims grounded — done")
            break

        slim = [{"statement": c.get("statement"),
                 "verdict": c.get("audit_verdict"),
                 "auditor_note": c.get("audit_note"),
                 "evidence": c.get("audit_evidence_pointer")} for c in bad]
        with open(tex_path) as f:
            tex = f.read()
        new_tex = None
        for attempt in range(2):
            resp = rewrite_llm(
                EDITS_PROMPT.format(claims=json.dumps(slim, indent=1), tex=tex),
                REWRITE_SYS, args.model)
            try:
                edits = [e for e in parse_json_array_loose(resp)
                         if isinstance(e, dict) and e.get("find")]
            except Exception as e:
                print(f"[reground] edit parse failed (attempt {attempt + 1}): {e}")
                continue
            cand, applied, missed = tex, 0, 0
            for e in edits:
                if e["find"] in cand:
                    cand = cand.replace(e["find"], e.get("replace", ""), 1)
                    applied += 1
                else:
                    missed += 1
            print(f"[reground] edits: {applied} applied, {missed} unmatched "
                  f"(attempt {attempt + 1})")
            if applied and applied >= missed:
                new_tex = cand
                break
        if not new_tex:
            print("[reground] WARNING no usable edits, stopping")
            break

        pre_round = tex_path + f".r{rnd}_backup"
        with open(pre_round, "w") as f:
            f.write(tex)  # pre-edit source, for restore on compile failure
        out_pdf = osp.join(exp_dir, f"{base}_{args.suffix}_r{rnd}.pdf")
        with open(tex_path, "w") as f:
            f.write(new_tex)
        compile_latex(latex_dir, out_pdf)
        if not osp.exists(out_pdf):
            print(f"[reground] round {rnd} compile failed, restoring tex and retrying")
            shutil.copy(pre_round, tex_path)
            continue
        final_pdf = out_pdf

        audit = run_audit(args.audit_script, out_pdf, exp_dir,
                          osp.join(exp_dir, f"grounding_{args.suffix}_r{rnd}"), args.model)

    n_bad = len(bad_claims_of(audit))
    status = "CLEAN" if n_bad == 0 else f"{n_bad} residual problem claims"
    print(f"[reground] FINAL pdf={final_pdf or 'original unchanged'} | {status} | "
          f"verdicts={audit['summary']['verdict_counts']}")


if __name__ == "__main__":
    main()
