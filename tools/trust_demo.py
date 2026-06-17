#!/usr/bin/env python
"""Demo: Trust Verifier on 4 failure modes (fabricated / honest / degenerate / overstated).
Run: export OPENAI_API_KEY/OPENAI_BASE_URL/TRUST_MODEL=gpt-5.4-pro; python tools/trust_demo.py
"""
import os.path as osp, sys, time
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from trust_verifier import verify
SAMPLES = [
 {"name": "① 造假", "expect": "NOT TRUSTWORTHY", "metrics": {},
  "claim": "Achieves 95.2% on CIFAR-10 and 88.7% on DomainNet, beating EWC (82.1%) and Replay (85.3%).",
  "evidence": "Ran ONLY synthetic MNIST. Best val 76.5%. CIFAR-10/DomainNet never loaded. EWC/Replay not implemented."},
 {"name": "② 诚实", "expect": "TRUSTWORTHY", "metrics": {"acc_finals": [0.765, 0.52]},
  "claim": "On synthetic MNIST we observe 76.5% best validation accuracy; larger datasets untested.",
  "evidence": "Synthetic MNIST. Method 76.5% vs baseline 52.0%. No other datasets claimed."},
 {"name": "③ 退化", "expect": "NOT TRUSTWORTHY", "metrics": {"acc_finals": [1.0, 1.0, 0.999, 1.0]},
  "claim": "Reaches 100% accuracy across all distributions, fully solving the task.",
  "evidence": "All conditions saturate at ~100%; no distributional difference measurable."},
 {"name": "④ 夸大", "expect": "QUESTIONABLE", "metrics": {"acc_finals": [0.4658, 0.3288]},
  "claim": "Compositional regularization significantly improves generalization.",
  "evidence": "Single seed, no significance test: 0.4658 vs 0.3288. High variance, not repeated."},
]
def verify_retry(s, tries=3):
    r = None
    for _ in range(tries):
        r = verify(s)
        if r["grounding"]["verdict"] not in ("ERROR", "UNCLEAR"): return r
        time.sleep(5)
    return r
def main():
    print("=" * 60 + "\n AI 科研可信验证器 — demo\n" + "=" * 60)
    hit = 0
    for s in SAMPLES:
        r = verify_retry(s); ok = "✓" if r["verdict"] == s["expect"] else "✗"; hit += ok == "✓"
        sig = r["signal"]
        sg = ("退化!" + ";".join(sig.get("degeneracy_reasons", [])) if sig.get("assessed") and sig["non_degenerate"] is False
              else ("非退化" if sig.get("assessed") else "未评估"))
        print(f"\n{s['name']}\n  信号层:{sg}\n  溯源层:{r['grounding']['verdict']} — {r['grounding'].get('why','')}\n  >>> {r['verdict']} (期望{s['expect']}) {ok}")
    print(f"\n命中 {hit}/{len(SAMPLES)}")
if __name__ == "__main__":
    main()
