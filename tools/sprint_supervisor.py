#!/usr/bin/env python
"""Pipeline supervisor: per-paper state machine + automatic chain triggering.

Replaces the "poll logs by hand and launch processing chains manually" ops
mode. Scans one or more AI-Scientist-v2 repo roots, infers each experiment's
state from artifacts, and fires the right processing chain on completed trees:

  RUNNING        tree search still active (launcher alive in root, or journal
                 mtime fresh)
  NEEDS_CHAIN    tree finished; no honest-version ensemble yet -> fire
                 process_completed.sh (PDF exists) or process_killed.sh (none)
  CHAIN_RUNNING  a chain this supervisor launched is still alive
  CHAIN_FAILED   chain exited without "CHAIN COMPLETE" (retried up to
                 --max_retries, then terminal)
  DONE           reviews/ has a regrounded/rewritten ensemble for the paper

Ledger: <sprint_dir>/supervisor_ledger.json — full history per experiment.

Usage:
  python sprint_supervisor.py --roots <repo_root> [<repo_root> ...] \
      [--sprint_dir ~/automationResearch/sprint] [--interval 600] \
      [--once] [--dry_run] [--max_chains 2] [--max_retries 1]
"""
import argparse
import glob
import json
import os
import os.path as osp
import re
import subprocess
import time

FRESH_SECS = 30 * 60          # journal younger than this => tree considered live


def now():
    return time.strftime("%Y-%m-%d_%H:%M:%S")


def load_ledger(path):
    if osp.exists(path):
        return json.load(open(path))
    return {}


def save_ledger(path, ledger):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ledger, f, indent=1)
    os.replace(tmp, path)


def launcher_pids_in_root(root):
    """PIDs of launch_scientist_bfts.py whose cwd is this repo root."""
    try:
        out = subprocess.run(["pgrep", "-f", "launch_scientist_bfts.py"],
                             capture_output=True, text=True).stdout.split()
    except Exception:
        return []
    pids = []
    for pid in out:
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd")
            if osp.realpath(cwd) == osp.realpath(root):
                pids.append(pid)
        except OSError:
            continue
    return pids


def journal_fresh(exp_dir):
    latest = 0
    for j in glob.glob(osp.join(exp_dir, "logs", "*", "stage_*", "journal.json")):
        latest = max(latest, osp.getmtime(j))
    return latest and (time.time() - latest) < FRESH_SECS


def has_journals(exp_dir):
    return bool(glob.glob(osp.join(exp_dir, "logs", "*", "stage_*", "journal.json")))


def has_pdf(exp_dir):
    return bool(glob.glob(osp.join(exp_dir, "*reflection*.pdf")))


def honest_review_exists(reviews_dir, expname):
    pats = [f"regrounded__*{expname}*_ensemble.json",
            f"rewritten__*{expname}*_ensemble.json"]
    return any(glob.glob(osp.join(reviews_dir, p)) for p in pats)


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError, TypeError):
        return False


def chain_log_complete(log_path):
    if not osp.exists(log_path):
        return False
    try:
        with open(log_path, errors="ignore") as f:
            return "CHAIN COMPLETE" in f.read()
    except OSError:
        return False


def short_tag(expname):
    """sup_<idea-name-stem> — unique enough, filesystem-safe."""
    m = re.search(r"\d{2}-\d{2}-\d{2}_(.+?)_attempt", expname)
    stem = m.group(1) if m else expname
    return "sup_" + stem[:40]


def fire_chain(sprint_dir, root, expname, ledger_rec, dry_run):
    chain = "process_completed.sh" if has_pdf(osp.join(root, "experiments", expname)) \
        else "process_killed.sh"
    tag = short_tag(expname)
    log_path = osp.join(sprint_dir, f"{tag}.log")
    if dry_run:
        print(f"[dry-run] would fire {chain} {root} {expname} {tag}")
        return None
    with open(log_path, "a") as logf:
        proc = subprocess.Popen(
            ["bash", osp.join(sprint_dir, chain), root, expname, tag],
            stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    ledger_rec.setdefault("history", []).append(
        {"t": now(), "event": f"fired {chain} pid={proc.pid} tag={tag}"})
    ledger_rec.update({"chain_pid": proc.pid, "chain_log": log_path,
                       "chain": chain, "tag": tag,
                       "attempts": ledger_rec.get("attempts", 0) + 1})
    print(f"[fire] {chain} -> {expname} (pid={proc.pid}, log={log_path})")
    return proc.pid


def classify(root, expname, reviews_dir, rec, root_has_live_launcher):
    exp_dir = osp.join(root, "experiments", expname)
    if honest_review_exists(reviews_dir, expname):
        return "DONE"
    if rec.get("chain_pid"):
        if pid_alive(rec["chain_pid"]):
            return "CHAIN_RUNNING"
        if chain_log_complete(rec.get("chain_log", "")):
            # chain done but honest review missing (e.g. reground below bar)
            return "CHAIN_DONE_NO_REVIEW"
        return "CHAIN_FAILED"
    if journal_fresh(exp_dir):
        return "RUNNING"
    if root_has_live_launcher and newest_exp_in_root(root) == expname:
        return "RUNNING"
    if not has_journals(exp_dir):
        return "EMPTY"
    return "NEEDS_CHAIN"


def newest_exp_in_root(root):
    exps = [d for d in glob.glob(osp.join(root, "experiments", "*")) if osp.isdir(d)]
    if not exps:
        return None
    return osp.basename(max(exps, key=osp.getmtime))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--sprint_dir",
                   default=osp.expanduser("~/automationResearch/sprint"))
    p.add_argument("--interval", type=int, default=600)
    p.add_argument("--once", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--max_chains", type=int, default=2)
    p.add_argument("--max_retries", type=int, default=1)
    args = p.parse_args()

    ledger_path = osp.join(args.sprint_dir, "supervisor_ledger.json")
    reviews_dir = osp.join(args.sprint_dir, "reviews")

    while True:
        ledger = load_ledger(ledger_path)
        running_chains = sum(
            1 for r in ledger.values()
            if r.get("chain_pid") and pid_alive(r["chain_pid"]))
        rows = []
        for root in args.roots:
            live = bool(launcher_pids_in_root(root))
            for exp in sorted(glob.glob(osp.join(root, "experiments", "*"))):
                if not osp.isdir(exp):
                    continue
                expname = osp.basename(exp)
                rec = ledger.setdefault(expname, {"root": root})
                state = classify(root, expname, reviews_dir, rec, live)
                if state == "CHAIN_FAILED":
                    if rec.get("attempts", 0) <= args.max_retries:
                        rec["chain_pid"] = None
                        if running_chains < args.max_chains:
                            fire_chain(args.sprint_dir, root, expname, rec,
                                       args.dry_run)
                            running_chains += 1
                            state = "CHAIN_RUNNING(retry)"
                    else:
                        state = "FAILED"
                elif state == "NEEDS_CHAIN" and running_chains < args.max_chains:
                    fire_chain(args.sprint_dir, root, expname, rec, args.dry_run)
                    running_chains += 1
                    state = "CHAIN_RUNNING" if not args.dry_run else state
                if rec.get("state") != state:
                    rec.setdefault("history", []).append(
                        {"t": now(), "event": f"state -> {state}"})
                rec["state"] = state
                rec["updated"] = now()
                rows.append((expname[:58], state))
        if not args.dry_run:
            save_ledger(ledger_path, ledger)
        print(f"--- supervisor pass {now()} "
              f"(chains running: {running_chains}) ---")
        for name, state in rows:
            print(f"  {state:24s} {name}")
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
