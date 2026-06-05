#!/usr/bin/env python3
"""Generate a "Generation Transparency Appendix" for an AI-Scientist-v2 paper.

Reconstructs the real tree-search structure from each stage's tree_data.json
(`edges` = [parent_idx, child_idx] over the node list, which aligns 1:1 with
journal.json node order) and infers each node's primitive action:
  draft   = stage root (seeded from previous stage best) / no parent
  debug   = parent node was buggy -> this node fixes the traceback
  improve = parent node ran clean -> this node tunes/extends it

Usage: python gen_gen_appendix.py <experiment_dir> <model_label> <dest.md>
"""
import json
import os
import re
import sys
import glob

STAGE_NAMES = {
    1: "initial_implementation 初始实现",
    2: "baseline_tuning 基线调参",
    3: "creative_research 创造性探索",
    4: "ablation 消融实验",
}


def clean(s, n=140):
    s = (s or "").replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("|", "\\|")
    return s[:n] + ("…" if len(s) > n else "")


def metric_str(m):
    # AIDE MetricValue: {"value": <scalar|{"metric_names":[...]}|null>, "maximize": ...}
    if isinstance(m, dict) and "value" in m:
        m = m["value"]
    if m is None:
        return "—"
    if isinstance(m, (int, float)):
        return f"{m:.4g}"
    if isinstance(m, dict):
        mn = m.get("metric_names")
        if isinstance(mn, list) and mn:
            first = mn[0]
            name = (first.get("metric_name") or "metric")[:16]
            data = first.get("data") or []
            vals = [d.get("final_value") for d in data
                    if isinstance(d, dict) and isinstance(d.get("final_value"), (int, float))]
            if vals:
                return f"{name}={sum(vals) / len(vals):.3g}"
            return name
        return "(metric)"
    return clean(str(m), 16)


def load_stage(sdir):
    J = json.load(open(os.path.join(sdir, "journal.json")))
    ns = J if isinstance(J, list) else J.get("nodes", [])
    edges = []
    tp = os.path.join(sdir, "tree_data.json")
    if os.path.exists(tp):
        edges = json.load(open(tp)).get("edges", [])
    return ns, edges


def build(sdir):
    ns, edges = load_stage(sdir)
    n = len(ns)
    parent = {c: p for p, c in edges}
    children = {}
    for p, c in edges:
        children.setdefault(p, []).append(c)

    def action(i):
        if i not in parent:
            return "draft"
        return "debug" if ns[parent[i]].get("is_buggy") else "improve"

    def status(i):
        return "buggy" if ns[i].get("is_buggy") else "good"

    roots = [i for i in range(n) if i not in parent]
    lines = []

    def rec(i, pre, last):
        mk = "└─" if last else "├─"
        lines.append(f"{pre}{mk} [{i}] {action(i)}/{status(i)} m={metric_str(ns[i].get('metric'))}")
        ch = children.get(i, [])
        for k, c in enumerate(ch):
            rec(c, pre + ("   " if last else "│  "), k == len(ch) - 1)

    for r in roots:
        lines.append(f"[{r}] {action(r)}/{status(r)} m={metric_str(ns[r].get('metric'))}  (根)")
        ch = children.get(r, [])
        for k, c in enumerate(ch):
            rec(c, "", k == len(ch) - 1)
    return ns, parent, action, lines


def main(exp, model):
    logs = os.path.join(exp, "logs", "0-run")
    out = []
    out.append("# 生成透明度附录:树搜索过程\n")
    out.append(
        f"> 本附录披露本论文由 **The AI Scientist v2** 自动生成的完整过程:走了哪些搜索策略、"
        f"产生多少树节点、如何逐层选择/裁剪、agent 如何判断决策。\n>\n> **写作模型**: {model}\n"
    )
    out.append("## A. 搜索策略概述\n")
    out.append(
        "系统不使用人写模板,从一个研究 idea **零起点**出发,做 **4 阶段 best-first 并行树搜索**。"
        "每个节点是一份完整可执行的实验代码;agent 用三种原语扩展树:\n"
    )
    out.append(
        "- **draft(草稿)**:阶段根节点 — 从上一阶段最佳节点 seed,或全新起草一个实现方向\n"
        "- **debug(调试)**:父节点运行**报错/buggy** → 读 traceback,定位并修复\n"
        "- **improve(改进)**:父节点运行**成功** → 在其基础上调超参/换设计/加分析,争取更高指标\n"
    )
    out.append(
        "每个节点用**多 seed(本次 3 seed)**重复评估取稳定指标。每阶段结束按指标选最佳节点 seed 给下一阶段,"
        "其余分支被**裁剪**(不再扩展)。四阶段:initial→baseline_tuning→creative_research→ablation。\n"
    )

    totals = {"n": 0, "good": 0, "buggy": 0}
    stage_dirs = sorted(glob.glob(os.path.join(logs, "stage_*")))
    out.append("## B. 各阶段搜索树\n")
    for sdir in stage_dirs:
        base = os.path.basename(sdir)
        m = re.match(r"stage_(\d+)", base)
        if not m:
            continue
        snum = int(m.group(1))
        try:
            ns, parent, action, tree = build(sdir)
        except Exception as e:
            out.append(f"### Stage {snum}: (读取失败 {e})\n")
            continue
        ng = sum(1 for x in ns if not x.get("is_buggy"))
        nb = len(ns) - ng
        totals["n"] += len(ns)
        totals["good"] += ng
        totals["buggy"] += nb
        out.append(f"### Stage {snum}: {STAGE_NAMES.get(snum, base)}\n")
        out.append(f"**{len(ns)} 节点**(good {ng} / buggy {nb})。搜索树:\n")
        out.append("```")
        out.extend(tree)
        out.append("```\n")
        out.append("| idx | 动作 | 父 | 状态 | metric | agent 计划/分析摘要 |")
        out.append("|---|---|---|---|---|---|")
        for i, nd in enumerate(ns):
            st = "buggy" if nd.get("is_buggy") else "good"
            par = parent.get(i, "—")
            txt = clean(nd.get("analysis") or nd.get("plan"))
            out.append(f"| {i} | {action(i)} | {par} | {st} | {metric_str(nd.get('metric'))} | {txt} |")
        out.append("")

    out.append("## C. 汇总统计\n")
    succ = 100 * totals["good"] / max(1, totals["n"])
    out.append(
        f"- **总节点数 {totals['n']}**(good {totals['good']} / buggy {totals['buggy']},成功率 {succ:.0f}%)"
    )
    out.append(
        "- 最终论文 = 全部阶段最佳节点的实验结果聚合而成;上表每个 buggy 节点都经 agent 自我 debug"
        "(部分修复成功转 good,部分被裁剪)"
    )
    out.append("- 动作推断规则:无父=draft(阶段根/seed),父 buggy→debug,父 good→improve")
    return "\n".join(out)


if __name__ == "__main__":
    exp, model, dest = sys.argv[1], sys.argv[2], sys.argv[3]
    open(dest, "w").write(main(exp, model))
    print(f"{model}: wrote {dest} ({os.path.getsize(dest)} bytes)")
