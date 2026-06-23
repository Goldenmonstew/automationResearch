#!/usr/bin/env python
"""Confound / validity detector — the second layer above grounding.

Grounding answers "does the paper's claim match the experimental data?" (catches
fabrication). This module answers the harder, higher-order question: "even if every
claim is grounded, is the EXPERIMENT DESIGN valid — or is the headline effect an
artifact of a confounded / circular design?" (catches a real finding that is false).

Motivating case (verified from real code): the `gradient_alignment_grokking` study
claims "noise aligned with the generalization direction accelerates grokking". But
its `aligned` intervention adds a gradient pulling the model TOWARD a `ref_state`
that is itself an already-trained (already-grokked) solution. So the "treatment"
secretly contains the OUTCOME — pulling toward the answer trivially reaches the
answer faster. The big effect is a distillation artifact, not evidence for the
hypothesis. That is the class of confound this detector targets: the operational
definition of the treatment leaks the outcome (construct invalidity / circularity).

Usage:
  OPENAI_API_KEY=... OPENAI_BASE_URL=... python confound_detector.py \
      --hypothesis "<the paper's causal hypothesis>" \
      --code_file <experiment_code.py> \
      [--claims "<key empirical claims, optional>"] \
      [--model gpt-5.4-pro] [--out verdict.json]
"""
import argparse
import json
import os
import re
import sys

try:
    from openai import OpenAI
except ImportError:
    print("pip install openai", file=sys.stderr)
    raise

# The confound taxonomy the reviewer is told to scan for. Ordered by how often it
# is the *hidden* killer in auto-generated ML papers (top ones are the subtle, high
# severity kind that grounding can never catch because the data genuinely shows it).
CONFOUND_TAXONOMY = """\
1. TREATMENT_LEAKS_OUTCOME (construct invalidity): the operationalization of the
   independent variable secretly contains information about the dependent variable.
   E.g. defining a "direction toward generalization" using an already-generalized
   solution, then testing whether moving along it generalizes faster. The effect is
   tautological. THIS IS THE MOST IMPORTANT AND MOST OFTEN MISSED.
2. CIRCULAR_DESIGN / TAUTOLOGY: the experimental setup guarantees the conclusion
   regardless of whether the hypothesis is true.
3. ALTERNATIVE_EXPLANATION: a more mundane mechanism (distillation, memorization,
   leakage) produces the same observed result, and was not ruled out.
4. INFORMATION_LEAKAGE: train/intervention pipeline uses validation/test data,
   labels, or a model trained on them.
5. INSUFFICIENT_CONTROL: missing the control/negative-control condition needed to
   attribute the effect to the claimed cause.
6. METRIC_SATURATION / CEILING: the metric is saturated (e.g. 100% acc) so it
   cannot discriminate between conditions.
7. SELECTION / P_HACKING: cherry-picked configs, seeds, or subsets.
8. UNDERPOWERED: too few seeds/samples to rule out the effect being noise, while
   claiming an effect "large enough to overcome seed variance".
"""

SYS = (
    "You are an exceptionally rigorous experimental-methodology reviewer (think a "
    "skeptical senior scientist running a confound check). Your ONLY job is to judge "
    "whether the experiment's DESIGN actually tests the stated causal hypothesis, or "
    "whether the headline effect could be an artifact of a confounded/circular design. "
    "You are NOT checking whether claims match the data (assume they do). You are "
    "checking whether a finding that IS in the data is nonetheless INVALID because of "
    "how the experiment was set up. Be concrete, cite the actual code, and prefer "
    "calling out the single most damaging confound over listing many minor ones. "
    "If the design is genuinely clean, say so — do not invent confounds."
)

PROMPT_TMPL = """\
## Stated causal hypothesis
{hypothesis}

## Key empirical claims (assume these ARE supported by the data)
{claims}

## Experiment code (the actual intervention; read it carefully)
```python
{code}
```

## Confound taxonomy to scan for
{taxonomy}

## Your task
Decide whether the experiment design VALIDLY tests the hypothesis. Focus hardest on
whether the TREATMENT's operationalization secretly encodes the OUTCOME, and on
whether a more mundane mechanism explains the result. Read the actual intervention
code — do not trust the variable names (e.g. something called "noise" may not be
noise; something called a "reference direction" may secretly be the answer).

Return ONLY a JSON object:
{{
  "overall_validity": "VALID" | "QUESTIONABLE" | "INVALID",
  "headline_effect_is_artifact": true | false,
  "confounds": [
    {{
      "type": "<one of the taxonomy keys>",
      "description": "<what is wrong, in one or two sentences>",
      "code_evidence": "<short quote or line reference from the code that shows it>",
      "alternative_explanation": "<the mundane mechanism that produces the same result>",
      "severity": "high" | "medium" | "low"
    }}
  ],
  "verdict_summary": "<2-3 sentence bottom line a reviewer would write>"
}}
"""


def _extract_json(text):
    """Pull the JSON object out of the model reply (handles code fences / prose)."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    # first {...} balanced-ish fallback
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in reply:\n" + text[:500])
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unbalanced JSON in reply:\n" + text[:500])


def detect(hypothesis, experiment_code, claims="(not provided)", model="gpt-5.4-pro",
           max_code_chars=24000):
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE"),
    )
    code = experiment_code
    if len(code) > max_code_chars:  # keep head+tail; the intervention is usually early
        code = code[:max_code_chars] + "\n# ...[truncated]...\n"
    prompt = PROMPT_TMPL.format(
        hypothesis=hypothesis, claims=claims, code=code, taxonomy=CONFOUND_TAXONOMY)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYS},
                  {"role": "user", "content": prompt}],
        temperature=0.0,
    )
    reply = resp.choices[0].message.content or ""
    verdict = _extract_json(reply)
    verdict["_model"] = model
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypothesis", required=True)
    ap.add_argument("--code_file", required=True)
    ap.add_argument("--claims", default="(not provided)")
    ap.add_argument("--model", default="gpt-5.4-pro")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.code_file) as f:
        code = f.read()
    verdict = detect(args.hypothesis, code, args.claims, args.model)

    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(verdict, f, indent=2, ensure_ascii=False)
        print(f"\n[saved] {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
