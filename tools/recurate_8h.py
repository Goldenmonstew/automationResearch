#!/usr/bin/env python3
"""8h-aware 重排:单节点 timeout 提到 8h、算力充足后,feasibility 不再是硬约束。

变化 vs merge_selection.py(1h 版):
  - 排序分 = manifestability + novelty + icbinb_fit(满15),**剔除 feasibility 惩罚**
    (8h 后几乎所有 idea 都跑得完;feasibility 只用来标记"这是重负载,要 8h")。
  - 候选 = 全部去重 unique(不再只取 router top-40),把当初因"太重"被压下去的 idea 捞回。
  - 仍裁:degenerate(效应<seed方差,调 timeout 救不了)+ 确认的硬/簇重复。

用法: python tools/recurate_8h.py <panel_a_out> <panel_b_out>
"""
import json
import re
import sys
from collections import defaultdict

CRIT = ["manifestability", "feasibility", "novelty", "icbinb_fit"]

# 仍裁(与 timeout 无关):退化 = 效应本身 < seed 方差;dup = 真重复
DROP = {
    71: "退化:forward/backward 插暂停 hand-wavy,效应淹于噪声",
    29: "退化:WD 步序 before/after 效应 <0.5% < seed 方差 → null",
    64: "退化:换激活 CIFAR 差异 <1% 且已知,饱和测不出",
    26: "退化:单个固定 TTA clean CIFAR <0.3%",
    68: "硬重复:与 67 逐字相同,保留 67",
    63: "重复:与 22 同题(The Augmentation Pathway),保留 22",
    4:  "簇收敛:与 20/8 同机制(aug 放大标签噪声),保留 20+8",
    13: "簇收敛:与 15 同 compressibility,保留 15",
    48: "簇收敛:与 57 同题(aug 强度调度),保留 57",
}


def load_panel_a(path):
    t = open(path).read(); a = t.find("{")
    while a != -1:
        try:
            d = json.loads(t[a:t.rfind("}") + 1])
            if "shortlist" in d:
                return d
        except Exception:
            pass
        a = t.find("{", a + 1)
    return {"shortlist": [], "dropped_dups": []}


def load_panel_b(path):
    t = open(path).read().strip()
    try:
        d = json.loads(t)
    except Exception:
        a, b = t.find("{"), t.rfind("}"); d = json.loads(t[a:b + 1])
    return d.get("result", d)


def main():
    router = json.load(open("selection/ledger.json"))
    panel_a = load_panel_a(sys.argv[1])
    panel_b = load_panel_b(sys.argv[2])
    pool = json.load(open("scaled_pool.json"))
    titles = {i: pool[i].get("Title", "") for i in range(len(pool))}

    # 收集每 idx 每判据的各源分 + dup 票
    crit_vals = defaultdict(lambda: defaultdict(list))  # idx -> crit -> [vals]
    nsrc = defaultdict(set)
    dup_votes = defaultdict(list)

    def ingest(i, o, src):
        if not (0 <= i < len(pool)):
            return
        got = False
        for c in CRIT:
            if c in o:
                try:
                    crit_vals[i][c].append(float(o[c])); got = True
                except Exception:
                    pass
        if got:
            nsrc[i].add(src)

    for model, byidx in router.items():
        for k, o in byidx.items():
            i = int(k); ingest(i, o, f"router:{model}")
            dv = int(o.get("dup_of", -1))
            if dv >= 0:
                dup_votes[i].append(dv)
    for o in panel_a.get("shortlist", []):
        try:
            ingest(int(o["index"]), o, "panel_a")
        except Exception:
            pass
    for s in panel_a.get("dropped_dups", []):
        m = re.match(r"\s*(\d+)\D+(\d+)", str(s))
        if m:
            dup_votes[int(m.group(1))].append(int(m.group(2)))
    csl = panel_b.get("shortlist", {})
    csl = csl.get("shortlist", csl) if isinstance(csl, dict) else csl
    for o in (csl or []):
        try:
            ingest(int(o["index"]), o, "panel_b")
        except Exception:
            pass

    # 每判据均值
    mean = {}
    for i, cv in crit_vals.items():
        mean[i] = {c: round(sum(v) / len(v), 2) if v else 0 for c, v in
                   {c: cv.get(c, []) for c in CRIT}.items()}

    # 8h-aware 排序分 = manif + novelty + icbinb(剔 feasibility)
    rank_score = {i: round(mean[i]["manifestability"] + mean[i]["novelty"] + mean[i]["icbinb_fit"], 2)
                  for i in mean}

    # 去重:dup 票 union → 组内保留 rank_score 最高者
    parent = {}
    for i, votes in dup_votes.items():
        if i not in rank_score or not votes:
            continue
        tgt = max(set(votes), key=votes.count)
        if tgt in rank_score and tgt != i:
            parent[i] = tgt

    def root(x):
        seen = set()
        while x in parent and x not in seen:
            seen.add(x); x = parent[x]
        return x
    groups = defaultdict(list)
    for i in rank_score:
        groups[root(i)].append(i)
    reps = {}
    for r, members in groups.items():
        best = max(members, key=lambda j: rank_score[j])
        reps[best] = sorted(members)

    final = [i for i in reps if i not in DROP and i in titles]
    final.sort(key=lambda i: -rank_score[i])

    # 输出
    with open("selection/SHORTLIST_8H.md", "w") as f:
        f.write("# 8h-aware 共识 shortlist(timeout 提到 8h,剔 feasibility 惩罚,捞回重 idea)\n\n")
        f.write("排序分 = manifestability + novelty + icbinb_fit(满15,不含 feasibility)。重负载项标 [需8h/T2]。\n\n")
        f.write(f"全池 76 → 去重 {len(reps)} unique → 裁 {len([i for i in reps if i in DROP])} → **shortlist {len(final)}**\n\n")
        f.write("| 排名 | idx | title | 排序分 | manif | nov | icbinb | feas | 评委 | 标记 |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for r, i in enumerate(final, 1):
            m = mean[i]
            heavy = "[需8h/T2]" if m["feasibility"] < 3.0 else ""
            f.write(f"| {r} | {i} | {titles[i][:42]} | **{rank_score[i]}** | "
                    f"{m['manifestability']} | {m['novelty']} | {m['icbinb_fit']} | {m['feasibility']} | "
                    f"{len(nsrc[i])} | {heavy} |\n")
        f.write("\n## 仍裁(与 timeout 无关)\n\n")
        for i, why in DROP.items():
            if i in titles:
                f.write(f"- [{i}] {titles[i][:46]} — {why}\n")

    out = [{"index": i, "title": titles[i], "rank_score": rank_score[i],
            **mean[i], "n_judges": len(nsrc[i]),
            "heavy_8h": mean[i]["feasibility"] < 3.0, "dup_merged": reps[i]}
           for i in final]
    json.dump(out, open("selection/shortlist_8h.json", "w"), ensure_ascii=False, indent=1)
    nheavy = sum(1 for i in final if mean[i]["feasibility"] < 3.0)
    print(f"76 → {len(reps)} unique → 裁 {len([i for i in reps if i in DROP])} → shortlist {len(final)}(其中重负载 {nheavy} 个)")
    print("  selection/SHORTLIST_8H.md + shortlist_8h.json")
    print("  新捞回(feasibility<3 的重 idea):")
    for i in final:
        if mean[i]["feasibility"] < 3.0:
            print(f"    [{i:2d}] rank {rank_score[i]} feas {mean[i]['feasibility']}  {titles[i][:48]}")


if __name__ == "__main__":
    main()
