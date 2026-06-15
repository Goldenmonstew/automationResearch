#!/usr/bin/env python3
"""合并多路评委(router 多模型 + 2 路外部评审面板)的 idea 打分,产出主 ledger + 共识 shortlist。

输入:
  selection/ledger.json     router 三评委原始分 {model:{idx:{scores}}}
  <panel_a_out>             外部评审面板 A 输出(prose+JSON {shortlist:[...],dropped_dups:[...]})
  <panel_b_out>             外部评审面板 B 结果({shortlist:{shortlist:[...]},check:{...}})
  scaled_pool.json          原始 76 idea(取 title)

聚合每个 idea 跨所有评委的总分(均值,各源都是 4 判据 1-5 求和,满分 20,可比),
并应用对抗复核的【裁剪决定】(退化/重负载/硬重复),全部带理由记进 MASTER_LEDGER.md。
"""
import json
import re
import sys
from collections import defaultdict

CRIT = ["manifestability", "feasibility", "novelty", "icbinb_fit"]

# ---- 对抗复核裁剪决定(逐条带理由,可审计)----
DROP = {
    71: "退化:forward/backward 插暂停机制 hand-wavy,效应在 full CIFAR 淹于噪声(最弱)",
    29: "退化:WD 步序 before/after 效应 <0.5% < seed 方差(~±0.3-0.5%)→ 大概率 null",
    64: "退化:换激活(ReLU vs Swish/Mish)CIFAR 上差异 <1% 且文献已知,饱和任务测不出",
    26: "退化:单个固定 TTA 在 clean CIFAR 提升 <0.3%,full data 测不出",
    68: "硬重复:与 67 abstract 逐字相同(grad-clip);保留 67",
    4:  "簇收敛:与 20(calibration)/8(duplication)同机制(aug 放大标签噪声),保留 20+8",
    13: "簇收敛:与 15 同 compressibility 机制;15 更具体可执行,保留 15",
    48: "簇收敛:与 57(aug 强度调度)同题,保留 57",
    63: "残留重复:与 22 标题完全相同(The Augmentation Pathway,固定增强顺序),保留 22",
}
FLAG_HEAVY = {
    5:  "重负载:含 DistilBERT/SST-2 文本 leg(T2 creep)→ 若入选须删文本 leg 只留 vision",
    23: "重负载:含 Tiny-ImageNet + 上千次训练 → 若入选须砍 Tiny-ImageNet、缩网格",
}


def load_panel_a(path):
    t = open(path).read()
    # 提取最外层 JSON 对象
    a = t.find("{")
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
        a, b = t.find("{"), t.rfind("}")
        d = json.loads(t[a:b + 1])
    # 兼容 {"result":{...}} 或直接 {...}
    return d.get("result", d)


def total_of(obj):
    try:
        return sum(float(obj.get(c, 0)) for c in CRIT)
    except Exception:
        return float(obj.get("total", 0) or 0)


def main():
    router_ledger = json.load(open("selection/ledger.json"))
    panel_a = load_panel_a(sys.argv[1])
    panel_b = load_panel_b(sys.argv[2])
    pool = json.load(open("scaled_pool.json"))
    titles = {i: pool[i].get("Title", "") for i in range(len(pool))}

    # 每个 idx 收集各源总分 + dup 票
    src_scores = defaultdict(dict)   # idx -> {source: total}
    dup_votes = defaultdict(list)    # idx -> [target,...]

    for model, byidx in router_ledger.items():
        for k, o in byidx.items():
            i = int(k)
            src_scores[i][f"router:{model}"] = round(total_of(o), 1)
            dv = int(o.get("dup_of", -1))
            if dv >= 0:
                dup_votes[i].append(dv)

    for o in panel_a.get("shortlist", []):
        try:
            i = int(o["index"]); src_scores[i]["panel_a"] = round(total_of(o), 1)
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
            i = int(o["index"]); src_scores[i]["panel_b"] = round(total_of(o), 1)
        except Exception:
            pass

    # 共识分 = 各源总分均值
    consensus = {}
    for i, srcs in src_scores.items():
        vals = list(srcs.values())
        consensus[i] = round(sum(vals) / len(vals), 2) if vals else 0

    # 候选基 = router pipeline 已去重的 40 shortlist(76→56→40),再叠其它源分 + 应用对抗裁剪
    router_shortlist = json.load(open("selection/shortlist.json"))
    candidates = [int(x["index"]) for x in router_shortlist]
    final = []
    for i in candidates:
        if i in DROP or i not in titles:
            continue
        final.append((i, consensus.get(i, 0)))
    final.sort(key=lambda x: -x[1])

    # 写主 ledger
    with open("selection/MASTER_LEDGER.md", "w") as f:
        f.write("# Idea 选择主 ledger(5 评委 ensemble + 对抗复核)\n\n")
        f.write("评委:router(deepseek-v3.2 / gpt-4o / kimi-k2.5) · 外部面板 A(reasoning 模型) · 外部面板 B(含对抗复核)\n\n")
        f.write(f"池子 76 → 评分去重 → 应用对抗裁剪 {len(DROP)} 项 → **共识 shortlist {len(final)}**\n\n")
        f.write("## 共识 shortlist(按综合分,满20)\n\n")
        f.write("| 排名 | idx | title | 综合分 | 各源分 | 重负载注 |\n|---|---|---|---|---|---|\n")
        for r, (i, c) in enumerate(final, 1):
            srcs = "; ".join(f"{k.split(':')[-1]}={v}" for k, v in src_scores[i].items())
            hv = FLAG_HEAVY.get(i, "")
            f.write(f"| {r} | {i} | {titles.get(i,'?')[:46]} | **{c}** | {srcs} | {hv} |\n")
        f.write("\n## 被对抗复核裁剪的(带理由)\n\n")
        for i, why in DROP.items():
            f.write(f"- **[{i}] {titles.get(i,'?')[:50]}** — {why}(各源分:{src_scores.get(i, {})})\n")
        f.write("\n## 需简化后再用的重负载项\n\n")
        for i, why in FLAG_HEAVY.items():
            inlist = "在shortlist" if i in dict(final) else "(已被裁)"
            f.write(f"- **[{i}] {titles.get(i,'?')[:50]}** {inlist} — {why}\n")

    json.dump([{"index": i, "title": titles.get(i,'?'), "consensus": c,
                "sources": src_scores[i], "heavy_note": FLAG_HEAVY.get(i, "")}
               for i, c in final],
              open("selection/final_shortlist.json", "w"), ensure_ascii=False, indent=1)
    print(f"76 → DROP {len(DROP)} → 共识 shortlist {len(final)}")
    print(f"  selection/MASTER_LEDGER.md + final_shortlist.json")
    print("  Top 10:")
    for i, c in final[:10]:
        print(f"    [{i:2d}] {c}  {titles.get(i,'?')[:50]}")


if __name__ == "__main__":
    main()
