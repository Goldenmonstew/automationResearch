#!/usr/bin/env python
"""Signal-quality ruler v1 — objective dims 1 & 2.

Measures whether an experiment produced a real, stable EFFECT (signal), not
whether its numbers look high. An experiment with all metrics saturated at
~100% has no measurable effect (every condition is identical) and is zero
signal, however pretty it looks.

    S = 1[non-degenerate] * 1[reproducible] * g(reliability) * h(relevance)

This v1 implements the two PURE-OBJECTIVE dimensions (no LLM):
  Dim 1  non-degeneracy : metric not saturated (>=.98) / collapsed (<=.02) /
                          constant; a measurable contrast exists.   (hard veto)
  Dim 2  reliability     : an effect rises above seed noise (cross-run spread).
Dim 3 (reproducibility re-run) and Dim 4 (hypothesis relevance) are stubbed for
the full version.

Acceptance test (the reversal): deepseek run (100%-saturated, scored Overall 4)
must come out S~=0; gpt-5.5 run (real 0.5-0.9 effects, scored Overall 3) must
come out S>0. If the ruler reproduces that, the metric reviewer was rewarding a
degenerate result and this ruler corrects it.

Usage: python signal_ruler.py <experiment_dir> [--tag NAME]
"""
import argparse
import glob
import json
import os

import numpy as np

SAT_HI, SAT_LO = 0.98, 0.02    # accuracy saturation / collapse band -> degenerate
MIN_CONTRAST = 0.02            # smallest effect that is not quantization noise
RELIAB_FULL = 0.10            # cross-run spread that earns full reliability credit


def _norm_acc(x):
    """Normalize an accuracy-like scalar to [0,1] (handle percent-scale)."""
    return x / 100.0 if x > 1.5 else x


def extract_numeric(obj, prefix=""):
    """Recursively pull every numeric leaf (scalar or array) keyed by path."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(extract_numeric(v, f"{prefix}/{k}"))
    elif isinstance(obj, (list, tuple)):
        if obj and all(isinstance(x, (int, float, np.number)) and not isinstance(x, bool) for x in obj):
            out[prefix] = np.asarray(obj, float)
        else:
            for i, v in enumerate(obj):
                out.update(extract_numeric(v, f"{prefix}[{i}]"))
    elif isinstance(obj, np.ndarray):
        if obj.dtype.kind in "fiu" and obj.size:
            out[prefix] = obj.astype(float)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = float(obj)
    return out


def is_acc(k):
    kl = k.lower()
    return (("acc" in kl) or ("f1" in kl) or ("auc" in kl)) and "loss" not in kl


def final_of(v):
    a = np.asarray(v, float).ravel()
    a = a[np.isfinite(a)]
    return float(a[-1]) if a.size else None


def load_all(exp_dir):
    paths = sorted(glob.glob(os.path.join(exp_dir, "**", "experiment_data.npy"),
                             recursive=True))
    mats = []
    for p in paths:
        try:
            d = np.load(p, allow_pickle=True)
            if hasattr(d, "dtype") and d.dtype == object:
                d = d.item()
        except Exception:
            continue
        mats.append((p, extract_numeric(d)))
    return mats


def signal_from_metrics(acc_finals, series=None):
    """Pure signal-quality judgment from accuracy numbers — no npy/files needed.

    acc_finals: list of accuracy-class final values (raw or percent; auto-normalized).
    series: optional {name: [values across runs]} for the reliability dim.
    Returns the same degeneracy/signal verdict as ruler(), so trust_verifier can
    judge constructed/streamed metrics locally without reading experiment_data.npy.
    """
    finals = [_norm_acc(float(x)) for x in (acc_finals or []) if x is not None]
    n = len(finals)
    arr = np.array(finals) if n else np.array([])
    saturated = arr[(arr >= SAT_HI) | (arr <= SAT_LO)] if n else arr
    sat_ratio = len(saturated) / n if n else 1.0
    contrast = float(arr.max() - arr.min()) if arr.size >= 2 else 0.0
    degenerate = (n == 0) or (sat_ratio > 0.5) or (contrast < MIN_CONTRAST)
    reasons = []
    if n == 0:
        reasons.append("no accuracy metric provided")
    if sat_ratio > 0.5:
        reasons.append(f"{sat_ratio:.0%} of metrics saturated/collapsed")
    if contrast < MIN_CONTRAST and n >= 2:
        reasons.append(f"no measurable contrast (range={contrast:.3f})")
    spreads = [float(np.ptp(v)) for v in (series or {}).values() if len(v) >= 2]
    reliability = float(np.median(spreads)) if spreads else 0.0
    g = min(1.0, reliability / RELIAB_FULL)
    S = (0 if degenerate else 1) * g
    return {
        "non_degenerate": not degenerate,
        "S": round(S, 3),
        "saturation_ratio": round(sat_ratio, 3),
        "measurable_contrast": round(contrast, 3),
        "g_reliability": round(g, 3),
        "degeneracy_reasons": reasons,
    }


def ruler(exp_dir, tag=None):
    mats = load_all(exp_dir)
    acc_finals = []                 # normalized accuracy finals across all npy
    series_by_key = {}              # key -> [finals across npy] (reliability)
    nonfinite = 0
    for i, (_p, m) in enumerate(mats):
        for k, v in m.items():
            a = np.asarray(v, float).ravel()
            if a.size and not np.all(np.isfinite(a)):
                nonfinite += 1
            if is_acc(k):
                fin = final_of(v)
                if fin is None:
                    continue
                fin = _norm_acc(fin)
                acc_finals.append(fin)
                series_by_key.setdefault(k, []).append(fin)

    n_acc = len(acc_finals)
    arr = np.array(acc_finals) if n_acc else np.array([])

    # ---- Dim 1: non-degeneracy (hard veto) ----
    saturated = arr[(arr >= SAT_HI) | (arr <= SAT_LO)] if n_acc else arr
    sat_ratio = len(saturated) / n_acc if n_acc else 1.0
    contrast = float(arr.max() - arr.min()) if arr.size >= 2 else 0.0
    degenerate = (n_acc == 0) or (sat_ratio > 0.5) or (contrast < MIN_CONTRAST)
    nondegen = 0 if degenerate else 1
    reasons = []
    if n_acc == 0:
        reasons.append("no accuracy-class metric found")
    if sat_ratio > 0.5:
        reasons.append(f"{sat_ratio:.0%} of accuracy metrics saturated/collapsed")
    if contrast < MIN_CONTRAST:
        reasons.append(f"no measurable contrast (range={contrast:.3f})")

    # ---- Dim 2: reliability (does an effect rise above run-to-run noise) ----
    spreads = [float(np.ptp(vals)) for vals in series_by_key.values() if len(vals) >= 2]
    reliability = float(np.median(spreads)) if spreads else 0.0
    g = min(1.0, reliability / RELIAB_FULL)

    S = nondegen * g

    return {
        "tag": tag or os.path.basename(exp_dir),
        "n_npy": len(mats),
        "n_acc_metrics": n_acc,
        "dim1_nondegenerate": bool(nondegen),
        "saturation_ratio": round(sat_ratio, 3),
        "measurable_contrast": round(contrast, 3),
        "nonfinite_series": nonfinite,
        "dim2_reliability_spread": round(reliability, 4),
        "g_reliability": round(g, 3),
        "S": round(S, 3),
        "degeneracy_reasons": reasons,
        "dim3_reproducibility": "NOT_RUN (needs seed re-run)",
        "dim4_relevance": "NOT_RUN (needs idea + LLM)",
        "acc_finals_sample": [round(float(x), 3) for x in acc_finals[:24]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir")
    ap.add_argument("--tag")
    a = ap.parse_args()
    print(json.dumps(ruler(a.exp_dir, a.tag), indent=2))


if __name__ == "__main__":
    main()
