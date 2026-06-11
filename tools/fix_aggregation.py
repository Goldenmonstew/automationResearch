#!/usr/bin/env python
"""Force-fix plot aggregation for a run whose auto aggregator produced too few
figures. Instead of letting the LLM guess the data layout (the systematic
failure mode), dump the REAL nested structure of every experiment_data.npy and
have gpt-5.5 write a defensive matplotlib script against it.

Run from the repo root of the run's working copy (cwd-relative paths).

Usage:
  python fix_aggregation.py --exp_dir experiments/<run> [--min_figs 5]
"""
import argparse
import json
import os
import os.path as osp
import shutil
import subprocess
import sys

import numpy as np

SYS = """You are an expert at writing robust matplotlib aggregation scripts for
machine-learning experiment data."""

PROMPT = """Write a complete standalone python script that aggregates experiment
results into publication-quality figures.

RESEARCH IDEA (context for what matters scientifically):
```
{idea}
```

DATA FILES AND THEIR EXACT NESTED STRUCTURE (truth — code against THIS, do not
guess other keys):
{structure}

REQUIREMENTS:
1. Load each .npy with np.load(path, allow_pickle=True).item().
2. Iterate ALL dataset keys dynamically; never hardcode a single dataset.
3. Produce AT LEAST {min_figs} distinct, informative figures (loss curves,
   metric comparisons across datasets/settings, final-value bar charts, etc.).
4. Save every figure into "figures/" with descriptive snake-free names via
   plt.savefig; create the dir with os.makedirs(..., exist_ok=True);
   NEVER call plt.show(); close figures after saving.
5. Wrap EACH figure in its own try/except so one failure cannot kill the rest;
   print the filename after each successful save.
6. Use only numpy/matplotlib/os.

Return the full script in a ```python fence."""


def describe(obj, depth=0, max_depth=4):
    pad = "  " * depth
    if isinstance(obj, dict):
        if depth >= max_depth:
            return pad + f"dict({len(obj)} keys: {list(obj)[:6]}...)\n"
        out = ""
        for k in list(obj)[:12]:
            out += pad + f"{k!r}:\n" + describe(obj[k], depth + 1, max_depth)
        if len(obj) > 12:
            out += pad + f"... ({len(obj) - 12} more keys)\n"
        return out
    if isinstance(obj, np.ndarray):
        return pad + f"ndarray shape={obj.shape} dtype={obj.dtype}\n"
    if isinstance(obj, (list, tuple)):
        inner = type(obj[0]).__name__ if obj else "empty"
        return pad + f"{type(obj).__name__} len={len(obj)} of {inner}\n"
    return pad + f"{type(obj).__name__}: {str(obj)[:60]}\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--min_figs", type=int, default=5)
    p.add_argument("--model", default="gpt-5.5")
    p.add_argument("--tries", type=int, default=3)
    args = p.parse_args()

    exp_dir = args.exp_dir.rstrip("/")
    src = osp.join(exp_dir, "logs", "0-run", "experiment_results")
    dst = osp.join(exp_dir, "experiment_results")
    if osp.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    assert osp.isdir(dst), "no experiment_results available"

    npys = []
    for root, _, files in os.walk(dst):
        for f in files:
            if f == "experiment_data.npy":
                npys.append(osp.relpath(osp.join(root, f), exp_dir))
    npys = sorted(npys)[:8]
    assert npys, "no experiment_data.npy found"

    chunks = []
    for rel in npys:
        try:
            # allow_pickle is safe here: these .npy files are produced by our
            # own tree-search runs on this machine (same trust domain — the
            # official aggregator loads them identically).
            d = np.load(osp.join(exp_dir, rel), allow_pickle=True).item()
            chunks.append(f"## {rel}\n" + describe(d)[:5000])
        except Exception as e:
            chunks.append(f"## {rel}\n<load error: {e}>")
    structure = "\n".join(chunks)[:40000]

    idea = ""
    for cand in ("research_idea.md", "idea.md"):
        ip = osp.join(exp_dir, cand)
        if osp.exists(ip):
            idea = open(ip).read()[:3000]
            break

    figdir = osp.join(exp_dir, "figures")
    os.makedirs(figdir, exist_ok=True)
    script_path = osp.join(exp_dir, "auto_plot_aggregator.py")

    sys.path.insert(0, os.getcwd())
    prompt = PROMPT.format(idea=idea, structure=structure, min_figs=args.min_figs)
    feedback = ""
    for attempt in range(1, args.tries + 1):
        try:
            import openai
            client = openai.OpenAI()
            r = client.responses.create(
                model=args.model, instructions=SYS,
                input=prompt + feedback,
                service_tier="priority", max_output_tokens=32768)
            resp = r.output_text
        except Exception as e:
            print(f"[fixagg] responses path failed ({e}); chat fallback")
            from ai_scientist.llm import create_client, get_response_from_llm
            client, mname = create_client(args.model)
            resp, _ = get_response_from_llm(prompt + feedback, client=client,
                                            model=mname, system_message=SYS,
                                            print_debug=False)
        import re
        m = re.search(r"```python(.*?)```", resp or "", re.DOTALL)
        if not m:
            m = re.search(r"```(.*?)```", resp or "", re.DOTALL)
        if not m:
            print(f"[fixagg] attempt {attempt}: no code block in response")
            continue
        with open(script_path, "w") as f:
            f.write(m.group(1).strip())
        for old in os.listdir(figdir):
            os.remove(osp.join(figdir, old))
        run = subprocess.run([sys.executable, "auto_plot_aggregator.py"],
                             cwd=exp_dir, capture_output=True, text=True,
                             timeout=600)
        n = len([f for f in os.listdir(figdir) if f.lower().endswith(".png")])
        print(f"[fixagg] attempt {attempt}: {n} figures "
              f"(rc={run.returncode})")
        if n >= args.min_figs:
            print("[fixagg] SUCCESS:", sorted(os.listdir(figdir)))
            break
        feedback = (f"\n\nPREVIOUS ATTEMPT produced only {n} figures. "
                    f"rc={run.returncode}\nstdout tail:\n{run.stdout[-1500:]}\n"
                    f"stderr tail:\n{run.stderr[-1500:]}\n"
                    "Fix the script and return the FULL corrected version.")
    else:
        print("[fixagg] FAILED to reach min_figs")

    if osp.isdir(dst):
        shutil.rmtree(dst)
    n = len([f for f in os.listdir(figdir) if f.lower().endswith(".png")])
    print(f"[fixagg] FINAL figures: {n}")
    sys.exit(0 if n >= args.min_figs else 1)


if __name__ == "__main__":
    main()
