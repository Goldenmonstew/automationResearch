#!/usr/bin/env python3
"""Measure real token usage from an AI-Scientist-v2 experiment dir.

token_tracker.json only captures models whose API returns a usage field
(OpenAI-native, i.e. the gpt-4o helper calls); router models (deepseek/gemini)
report no usage so the main code/feedback tokens are missing there. The full
prompt+response text IS persisted in token_tracker_interactions.json, so we
tiktoken-count every string leaf in that file to recover the true volume of
tokens sent+received across the whole run.

Usage: python measure_tokens.py <experiment_dir> <tag>
"""
import json
import sys

import tiktoken

enc = tiktoken.get_encoding("cl100k_base")


def walk(o, acc):
    if isinstance(o, str):
        acc[0] += len(enc.encode(o))
        acc[1] += 1
    elif isinstance(o, dict):
        for v in o.values():
            walk(v, acc)
    elif isinstance(o, list):
        for v in o:
            walk(v, acc)


def main(base, tag):
    # token_tracker.json summary (helper models that return usage)
    try:
        tt = json.load(open(base + "/token_tracker.json"))
        print("[%s] token_tracker.json (仅有 usage 字段的模型, 通常是 gpt-4o 辅助调用):" % tag)
        for model, info in tt.items():
            tk = info.get("tokens", {})
            print("    %s: prompt=%s completion=%s cached=%s reasoning=%s" % (
                model, tk.get("prompt"), tk.get("completion"),
                tk.get("cached"), tk.get("reasoning")))
    except Exception as e:
        print("[%s] token_tracker.json err: %s" % (tag, e))

    # full interaction log -> recursive tiktoken sum
    d = json.load(open(base + "/token_tracker_interactions.json"))
    top_type = type(d).__name__
    top_len = len(d) if hasattr(d, "__len__") else -1
    inner = d
    if isinstance(d, list) and len(d) == 1 and isinstance(d[0], list):
        inner = d[0]
    rounds = len(inner) if hasattr(inner, "__len__") else -1
    acc = [0, 0]
    walk(d, acc)
    print("[%s] interactions.json: top=%s(len %d), 交互轮数~%d" % (tag, top_type, top_len, rounds))
    print("    全部字符串 token 合计(真实收发量) ~%s  (%s 个字符串叶子)" % (
        format(acc[0], ","), format(acc[1], ",")))
    if rounds > 0:
        print("    平均 ~%s token/轮" % format(acc[0] // rounds, ","))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
