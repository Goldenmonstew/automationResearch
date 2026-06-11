#!/usr/bin/env python
"""Validation for the /v1/responses treesearch backend adapter.

Exercises text and forced-function-call queries through both backends with
identical inputs, checks output contracts, and verifies dispatcher wiring.
Run from an AI-Scientist-v2 repo root with router env set.
"""
import os
import sys
import time

os.environ["USE_RESPONSES_API"] = "1"
sys.path.insert(0, os.getcwd())

from ai_scientist.treesearch.backend import (  # noqa: E402
    backend_openai,
    backend_openai_responses,
    query as dispatch_query,
)
from ai_scientist.treesearch.backend.utils import FunctionSpec  # noqa: E402

MODEL = "gpt-5.5"
SYS = "You are a terse coding assistant."
USER_TEXT = ("In exactly two sentences, explain what early stopping is "
             "in neural network training.")
SPEC = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {"type": "boolean",
                       "description": "whether the run failed"},
            "summary": {"type": "string",
                        "description": "one-sentence summary"},
            "metric": {"type": ["number", "null"],
                       "description": "validation accuracy if reported"},
        },
        "required": ["is_bug", "summary"],
    },
    description="Submit a structured review of an experiment log.",
)
USER_FC = ("Experiment log: trained 3-layer MLP on MNIST subset, "
           "final val_acc=0.914, no errors. Submit your review.")

results = []
for label, fn in [("responses", backend_openai_responses.query),
                  ("chat", backend_openai.query)]:
    # text query
    t0 = time.time()
    out, rt, itok, otok, info = fn(system_message=SYS, user_message=USER_TEXT,
                                   model=MODEL, max_tokens=16000)
    ok_text = isinstance(out, str) and len(out) > 20
    print(f"[{label}/text] ok={ok_text} wall={time.time()-t0:.1f}s "
          f"tier={info.get('service_tier')} endpoint={info.get('endpoint', 'chat')} "
          f"tokens={itok}/{otok}\n  out: {out[:140]!r}")
    # forced function call
    t0 = time.time()
    out2, rt2, itok2, otok2, info2 = fn(system_message=SYS, user_message=USER_FC,
                                        func_spec=SPEC, model=MODEL,
                                        max_tokens=16000)
    ok_fc = (isinstance(out2, dict) and "is_bug" in out2 and "summary" in out2
             and out2["is_bug"] is False)
    print(f"[{label}/fc]   ok={ok_fc} wall={time.time()-t0:.1f}s "
          f"tier={info2.get('service_tier')}\n  out: {out2}")
    results.append((label, ok_text, ok_fc))

# dispatcher wiring (env flag set above): must route gpt-5.5 to responses
out3 = dispatch_query(system_message=SYS, user_message=USER_FC,
                      model=MODEL, func_spec=SPEC)
print(f"[dispatch/fc] type={type(out3).__name__} keys="
      f"{sorted(out3.keys()) if isinstance(out3, dict) else 'n/a'}")

all_ok = all(t and f for _, t, f in results) and isinstance(out3, dict)
print("RESULT:", "ALL-PASS" if all_ok else "FAILURES-PRESENT")
sys.exit(0 if all_ok else 1)
