#!/usr/bin/env python
"""P3: Anti-Anchoring Solution Pool for tree search.

ICR-inspired diversity forcing: when the tree search is about to expand
the current best node, sometimes inject a "counter-plan" that's explicitly
different from the dominant approach.  Prevents the search from anchoring
on a single strategy and missing better alternatives.

Mechanisms:
  1. Diversity prompt injection: modify the plan prompt to explicitly request
     an approach different from the current best
  2. Solution pool tracking: maintain a pool of distinct high-level strategies
     seen so far, force the next plan to avoid repeating them

Integration: called from parallel_agent.py's node expansion or
agent_manager.py's substage goal generation.

Usage (standalone, for testing):
  python tools/anti_anchoring.py --plans_file <journal_plans.json> \
      --generate_alternative --model gpt-4o
"""
import argparse
import json
import os.path as osp
import re
import sys
import time

# ---------------------------------------------------------------------------
# Solution Pool: tracks distinct strategies to force diversity
# ---------------------------------------------------------------------------

class SolutionPool:
    """Maintains a pool of high-level strategies seen during tree search.
    Used to force diversity by preventing the LLM from repeating approaches."""

    def __init__(self):
        self.strategies = []  # list of {"summary": str, "metric": float, "node_id": str}
        self.rejected = []    # approaches explicitly avoided

    def add(self, summary, metric=None, node_id=None):
        """Register a strategy that has been tried."""
        self.strategies.append({
            "summary": summary[:200],
            "metric": metric,
            "node_id": str(node_id)[:20] if node_id else None,
        })

    def get_diversity_constraint(self, top_k=5):
        """Generate a constraint string listing strategies to AVOID repeating."""
        if not self.strategies:
            return ""

        recent = self.strategies[-top_k:]
        lines = ["DIVERSITY CONSTRAINT — do NOT repeat these approaches:"]
        for i, s in enumerate(recent, 1):
            met = f" (metric={s['metric']})" if s['metric'] is not None else ""
            lines.append(f"  {i}. {s['summary']}{met}")
        lines.append("")
        lines.append("Your new approach MUST be fundamentally different from all of the above.")
        lines.append("Different means: different algorithm, different data processing, "
                     "different loss function, or different architecture — not just "
                     "hyperparameter changes.")
        return "\n".join(lines)

    def should_force_diversity(self, n_consecutive_same=3):
        """Return True if recent strategies look too similar (basic heuristic)."""
        if len(self.strategies) < n_consecutive_same:
            return False
        recent = self.strategies[-n_consecutive_same:]
        # Simple check: if all recent summaries share >50% words, force diversity
        if not recent:
            return False
        word_sets = []
        for s in recent:
            words = set(s["summary"].lower().split())
            word_sets.append(words)
        if not word_sets[0]:
            return False
        common = word_sets[0]
        for ws in word_sets[1:]:
            common = common & ws
        avg_size = sum(len(ws) for ws in word_sets) / len(word_sets)
        overlap = len(common) / max(avg_size, 1)
        return overlap > 0.5

    def to_dict(self):
        return {"strategies": self.strategies, "rejected": self.rejected}

    @classmethod
    def from_dict(cls, d):
        pool = cls()
        pool.strategies = d.get("strategies", [])
        pool.rejected = d.get("rejected", [])
        return pool

    @classmethod
    def from_journal(cls, journal_data):
        """Build pool from a journal's node plans."""
        pool = cls()
        for node in journal_data.get("nodes", []):
            plan = node.get("plan", "")
            metric = None
            m = node.get("metric")
            if isinstance(m, dict):
                metric = m.get("value")
                if isinstance(metric, dict):
                    metric = max(metric.values()) if metric else None
            elif isinstance(m, (int, float)):
                metric = m
            if plan:
                pool.add(plan[:200], metric=metric, node_id=node.get("id"))
        return pool


# ---------------------------------------------------------------------------
# Counter-plan generation
# ---------------------------------------------------------------------------

COUNTER_PLAN_SYS = """You are a creative ML researcher who specializes in finding
ALTERNATIVE approaches when the obvious path isn't working well enough.
You deliberately avoid the most popular approach and look for fundamentally
different strategies."""

COUNTER_PLAN_PROMPT = """The current best approach for this research task is:

TASK: {task_description}

CURRENT BEST PLAN:
{current_plan}

APPROACHES ALREADY TRIED:
{tried_approaches}

Generate a FUNDAMENTALLY DIFFERENT approach. Requirements:
1. Must use a different algorithm/architecture/loss than the current best
2. Must not be a minor variation (different hyperparameters don't count)
3. Should be technically sound and implementable
4. Should have a clear hypothesis for why it might work better

Return a JSON object inside a ```json fence:
{{
  "plan": "<detailed plan for the alternative approach>",
  "key_difference": "<what makes this fundamentally different>",
  "hypothesis": "<why this might outperform the current best>"
}}"""


def generate_counter_plan(task_desc, current_plan, pool, model="gpt-4o"):
    """Generate an alternative plan that's explicitly different from current best."""
    tried = pool.get_diversity_constraint() if pool.strategies else "(none yet)"

    prompt = COUNTER_PLAN_PROMPT.format(
        task_description=task_desc[:2000],
        current_plan=current_plan[:2000],
        tried_approaches=tried,
    )

    if model.startswith("gpt-5"):
        try:
            import openai
            client = openai.OpenAI()
            r = client.responses.create(
                model=model, instructions=COUNTER_PLAN_SYS,
                input=prompt, max_output_tokens=4096)
            resp = getattr(r, "output_text", "") or ""
        except Exception:
            from ai_scientist.llm import create_client, get_response_from_llm
            client, mname = create_client(model)
            resp, _ = get_response_from_llm(
                prompt, client=client, model=mname,
                system_message=COUNTER_PLAN_SYS, print_debug=False)
    else:
        from ai_scientist.llm import create_client, get_response_from_llm
        client, mname = create_client(model)
        resp, _ = get_response_from_llm(
            prompt, client=client, model=mname,
            system_message=COUNTER_PLAN_SYS, print_debug=False)

    # Parse response
    patterns = [
        re.findall(r"```json\s*(.*?)```", resp or "", re.DOTALL),
        re.findall(r"```\s*(.*?)```", resp or "", re.DOTALL),
        [resp or ""],
    ]
    for candidates in patterns:
        for c in candidates:
            c = c.strip()
            start = c.find("{")
            if start == -1:
                continue
            try:
                obj = json.JSONDecoder().raw_decode(c[start:])[0]
                if isinstance(obj, dict) and "plan" in obj:
                    return obj
            except json.JSONDecodeError:
                continue
    return {"plan": resp[:1000], "key_difference": "parse_failed", "hypothesis": ""}


# ---------------------------------------------------------------------------
# Integration hook for parallel_agent.py
# ---------------------------------------------------------------------------

def maybe_inject_diversity(plan_prompt, journal, stage_name,
                           diversity_prob=0.3, model=None):
    """Hook to call before plan generation in parallel_agent.

    With probability diversity_prob, prepend a diversity constraint to the
    plan prompt to force the LLM away from anchoring on the current best.

    Args:
        plan_prompt: the original plan prompt string
        journal: Journal object with nodes
        stage_name: current stage name
        diversity_prob: probability of injecting diversity

    Returns:
        modified plan_prompt (may have diversity prefix)
    """
    import random
    if random.random() > diversity_prob:
        return plan_prompt

    # Build pool from journal
    plans_seen = []
    for node in getattr(journal, "nodes", []):
        plan = getattr(node, "plan", "")
        if plan:
            plans_seen.append(plan[:150])

    if len(plans_seen) < 2:
        return plan_prompt

    diversity_prefix = (
        "IMPORTANT: The following approaches have already been tried. "
        "You MUST propose something fundamentally different — a different "
        "algorithm, architecture, loss function, or data processing approach. "
        "Minor variations (hyperparameter tuning, layer count changes) are NOT acceptable.\n\n"
        "APPROACHES ALREADY TRIED:\n"
    )
    for i, p in enumerate(plans_seen[-5:], 1):
        diversity_prefix += f"  {i}. {p}\n"
    diversity_prefix += "\nYour approach must differ from ALL of the above.\n\n"

    return diversity_prefix + plan_prompt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plans_file", help="journal JSON with node plans")
    p.add_argument("--generate_alternative", action="store_true")
    p.add_argument("--task_desc", default="Improve ML model performance")
    p.add_argument("--current_plan", default="")
    p.add_argument("--model", default="gpt-4o")
    args = p.parse_args()

    if args.plans_file:
        with open(args.plans_file) as f:
            journal_data = json.load(f)
        pool = SolutionPool.from_journal(journal_data)
        print(f"Built pool with {len(pool.strategies)} strategies")
        print(f"Should force diversity: {pool.should_force_diversity()}")
        print(f"\nDiversity constraint:\n{pool.get_diversity_constraint()}")
    else:
        pool = SolutionPool()

    if args.generate_alternative:
        result = generate_counter_plan(
            args.task_desc, args.current_plan or "baseline approach",
            pool, model=args.model)
        print(f"\nCounter-plan:\n{json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
