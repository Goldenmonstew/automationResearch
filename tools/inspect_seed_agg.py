#!/usr/bin/env python
"""Inspect how the tree search records multi-seed results.

Key question for Gate B: does is_seed_agg_node already store per-seed values +
variance, or must Gate B re-run to get a distribution? Also dumps the exact
metric / parsed_metrics schema so the variance/significance checks read real
fields.

Usage: python inspect_seed_agg.py <journal.json>
"""
import json
import sys


def main():
    j = json.load(open(sys.argv[1]))
    nodes = j["nodes"] if isinstance(j, dict) else j

    def g(n, k, d=None):
        return n.get(k, d)

    seed_nodes = [n for n in nodes if g(n, "is_seed_node")]
    agg_nodes = [n for n in nodes if g(n, "is_seed_agg_node")]
    good = [n for n in nodes if not g(n, "is_buggy") and g(n, "metric") not in (None, "", {})]

    print(f"total={len(nodes)} good={len(good)} seed_nodes={len(seed_nodes)} agg_nodes={len(agg_nodes)}")

    # metric + parsed_metrics schema from a good node
    if good:
        n = good[0]
        print("\n=== GOOD node metric fields ===")
        print("metric       :", json.dumps(g(n, "metric"), default=str)[:500])
        print("parsed_metrics:", json.dumps(g(n, "parsed_metrics"), default=str)[:800])

    # how an aggregation node stores multi-seed
    if agg_nodes:
        n = agg_nodes[0]
        print("\n=== SEED-AGG node ===")
        print("metric       :", json.dumps(g(n, "metric"), default=str)[:500])
        print("parsed_metrics:", json.dumps(g(n, "parsed_metrics"), default=str)[:1200])
        # any field mentioning std/mean/seed/ci
        for k, v in n.items():
            ks = k.lower()
            if any(t in ks for t in ("seed", "std", "mean", "var", "ci")):
                print(f"  field[{k}]: {json.dumps(v, default=str)[:300]}")
    else:
        print("\n(no seed-agg node in this stage journal)")

    # a seed node's metric (single seed value)
    if seed_nodes:
        n = seed_nodes[0]
        print("\n=== single SEED node metric ===")
        print("parsed_metrics:", json.dumps(g(n, "parsed_metrics"), default=str)[:600])


if __name__ == "__main__":
    main()
