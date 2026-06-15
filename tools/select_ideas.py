#!/usr/bin/env python3
"""多模型 ensemble idea 选择 pipeline（带完整决策留痕）。

把一个 idea 池(ideation 产出的 JSON 数组)交给 N 个 LLM 评委(router 上的多模型)
独立打分,聚合成共识排名 + 去重 shortlist,并把【每个评委对每个 idea 的分数和理由】
全部写进 ledger,形成可审计、可复用的自动化选择流程。

判据(各 1-5):
  manifestability  效应能否在小数据上真显现、不退化(避免 ~100% 测不出)
  feasibility      小模型能否在 1h/节点内跑完(避开预训练大模型/ImageNet 全量)
  novelty          研究问题新颖度
  icbinb_fit       负面结果/失败模式主题契合度
另要每评委标 dup_of(若与更早某 index 实质重复,填那个 index,否则 -1)+ 一句 reason。

用法:
  OPENAI_API_KEY=<k> OPENAI_BASE_URL=<url> python tools/select_ideas.py \
    --pool scaled_pool.json --judges deepseek-v3.2,gpt-4o,kimi-k2.5,gpt-5.5 \
    --top 40 --out-dir selection
输出:selection/ledger.json(全评委全 idea 原始打分)、selection/shortlist.json(聚合去重 top N)、
     selection/SELECTION_LOG.md(人类可读决策记录)。
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

from openai import OpenAI

CRITERIA = ["manifestability", "feasibility", "novelty", "icbinb_fit"]


def compact(ideas):
    """裁剪成评委够用的紧凑形式(省 token,保留判断所需)。"""
    out = []
    for i, x in enumerate(ideas):
        out.append({
            "index": i,
            "name": x.get("Name", ""),
            "title": x.get("Title", ""),
            "hypothesis": (x.get("Short Hypothesis", "") or "")[:400],
            "experiments": (x.get("Experiments", "") or "")[:500],
        })
    return out


JUDGE_SYS = (
    "You are a rigorous, terse reviewer selecting AI research ideas for an automated "
    "reproduction pipeline. Each experiment node has a 1-hour compute cap and runs on a "
    "single small GPU, so favour ideas that (a) make their effect MANIFEST on small real "
    "datasets (MNIST/CIFAR/etc.) without degenerating to ~100% accuracy, (b) are FEASIBLE "
    "in 1h with small models (penalise anything needing ImageNet-scale / pretrained large "
    "models / LLM finetuning), (c) are NOVEL, (d) FIT the 'negative results / failure modes' "
    "(ICBINB) theme. Output ONLY valid JSON, no prose."
)


def judge_prompt(batch):
    return (
        "Score EACH idea below 1-5 on manifestability, feasibility, novelty, icbinb_fit. "
        "Also set dup_of to the index of an EARLIER idea it substantially duplicates (else -1), "
        "and give a one-sentence reason.\n"
        "Return ONLY a JSON array, one object per idea, exactly:\n"
        '[{"index":N,"manifestability":1-5,"feasibility":1-5,"novelty":1-5,'
        '"icbinb_fit":1-5,"dup_of":N_or_-1,"reason":"..."}]\n\n'
        + json.dumps(batch, ensure_ascii=False)
    )


def parse_json_array(text):
    """从模型输出里鲁棒提取 JSON 数组(容忍 ```json fence / 前后文字)。"""
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    blob = m.group(1) if m else None
    if blob is None:
        a, b = text.find("["), text.rfind("]")
        blob = text[a:b + 1] if a != -1 and b > a else None
    if not blob:
        return []
    try:
        return json.loads(blob)
    except Exception:
        # 逐对象兜底
        objs = re.findall(r"\{[^{}]*\}", blob, re.DOTALL)
        out = []
        for o in objs:
            try:
                out.append(json.loads(o))
            except Exception:
                pass
        return out


def run_judge(client, model, items, batch_size=25):
    scores = {}
    for s in range(0, len(items), batch_size):
        batch = items[s:s + batch_size]
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYS},
                    {"role": "user", "content": judge_prompt(batch)},
                ],
                max_tokens=16000,
                temperature=0.2,
            )
            arr = parse_json_array(resp.choices[0].message.content or "")
        except Exception as e:
            print(f"  [{model}] batch {s} 出错: {e}", file=sys.stderr)
            arr = []
        for o in arr:
            try:
                idx = int(o["index"])
                scores[idx] = o
            except Exception:
                pass
        print(f"  [{model}] 评完 {min(s + batch_size, len(items))}/{len(items)}")
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--judges", required=True, help="逗号分隔的 router 模型名")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--out-dir", default="selection")
    args = ap.parse_args()

    ideas = json.load(open(args.pool))
    items = compact(ideas)
    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE"),
    )

    # 1) 跑所有评委 -> ledger
    ledger = {}  # model -> {index -> score obj}
    for m in judges:
        print(f"== 评委 {m} ==")
        ledger[m] = run_judge(client, m, items)

    # 2) 聚合:每 idea 跨评委的均分 + dup 票
    agg = {}
    for i in range(len(items)):
        per = [ledger[m][i] for m in judges if i in ledger[m]]
        if not per:
            continue
        means = {c: round(sum(float(p.get(c, 0)) for p in per) / len(per), 2) for c in CRITERIA}
        total = round(sum(means.values()), 2)
        dup_votes = [int(p.get("dup_of", -1)) for p in per if int(p.get("dup_of", -1)) >= 0]
        agg[i] = {
            "index": i, "name": items[i]["name"], "title": items[i]["title"],
            **means, "total": total, "n_judges": len(per),
            "dup_of_votes": dup_votes,
            "reasons": {m: ledger[m][i].get("reason", "") for m in judges if i in ledger[m]},
        }

    # 3) 去重:多数评委认为是某 index 的重复就并入(保留组内 total 最高者)
    parent = {}
    for i, a in agg.items():
        votes = a["dup_of_votes"]
        if votes:
            # 取被投最多的 target
            tgt = max(set(votes), key=votes.count)
            if votes.count(tgt) >= max(2, a["n_judges"] // 2) and tgt in agg:
                parent[i] = tgt
    groups = defaultdict(list)
    for i in agg:
        root = i
        seen = set()
        while root in parent and root not in seen:
            seen.add(root)
            root = parent[root]
        groups[root].append(i)
    reps = []
    for root, members in groups.items():
        best = max(members, key=lambda j: agg[j]["total"])
        rep = dict(agg[best])
        rep["merged_indices"] = sorted(members)
        reps.append(rep)

    # 4) 排序 + 取 top N
    reps.sort(key=lambda r: r["total"], reverse=True)
    shortlist = reps[:args.top]

    # 5) 落盘
    json.dump(ledger, open(f"{args.out_dir}/ledger.json", "w"), ensure_ascii=False, indent=1)
    json.dump(shortlist, open(f"{args.out_dir}/shortlist.json", "w"), ensure_ascii=False, indent=1)
    with open(f"{args.out_dir}/SELECTION_LOG.md", "w") as f:
        f.write(f"# Idea 选择记录(多模型 ensemble)\n\n")
        f.write(f"- 池子:{len(items)} idea  ·  评委:{', '.join(judges)}  ·  去重后 {len(reps)} unique  ·  选 top {len(shortlist)}\n\n")
        f.write("| 排名 | idx | title | 总分 | manif | feas | novel | icbinb | 评委数 | 合并 |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for r, x in enumerate(shortlist, 1):
            f.write(f"| {r} | {x['index']} | {x['title'][:50]} | {x['total']} | "
                    f"{x['manifestability']} | {x['feasibility']} | {x['novelty']} | {x['icbinb_fit']} | "
                    f"{x['n_judges']} | {x.get('merged_indices')} |\n")
        f.write("\n## 各评委理由(留痕)\n\n")
        for x in shortlist:
            f.write(f"### [{x['index']}] {x['title']}\n")
            for m, why in x["reasons"].items():
                f.write(f"- **{m}**: {why}\n")
            f.write("\n")
    print(f"\n{len(items)}→{len(reps)} unique→top {len(shortlist)}。"
          f"输出 {args.out_dir}/{{ledger,shortlist}}.json + SELECTION_LOG.md")


if __name__ == "__main__":
    main()
