#!/usr/bin/env python
"""D1 confound-ablation for the `gradient_alignment_grokking` headline finding.

Faithful minimal modification of headline node 38142299. The model (ModularMLP),
optimizer (AdamW), hyperparameters, gradient-noise injection mechanism, and the
grokking metric (fit_step at train_acc>=0.99; grok_step once fit_step set AND
val_acc>=0.95) are IDENTICAL to the original. The ONLY thing that changes between
arms is the reference teacher whose weights the "aligned" noise points toward.

Original claim: "noise aligned with the generalization direction accelerates
grokking". The aligned direction is d = current_weights - ref_state, and in the
original code ref_state is a teacher trained on the FULL table (train+val) -> the
treatment is literally "nudge weights toward a model that already solved the val
set". Reference arms isolate whether the effect is a real mechanism or leakage:

  REPLICATE  : teacher on the FULL table, TRUE labels  -> reproduces the headline.
  DECOY      : teacher on the full table with a fixed PERMUTED label map
               y' = pi[(a+b) % p]. A genuinely grokked net for a DIFFERENT
               function, NOT this task's answer.
  TRAIN_ONLY : teacher on the TRAIN split only (honest; no val leak).

Conditions:
  none    : no injected noise (baseline).
  aligned : correlated pull toward ref + isotropic noise (original).
  anti    : correlated pull away from ref + isotropic noise (original).
  iso     : isotropic noise ONLY (alpha=0). The undirected-noise control the
            original *idea* proposed but never implemented as a condition.

Predictions (confound hypothesis): aligned groks ONLY under REPLICATE (ref encodes
the true answer); under DECOY/TRAIN_ONLY/iso it does not grok -> the effect is
outcome leakage, not a gradient-alignment mechanism. Pure torch; no network/LLM.
"""
import os, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- verbatim from headline node 38142299 -----------------
class ModAddDataset(Dataset):
    def __init__(self, pairs, labels):
        self.x = torch.tensor(pairs, dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return {"x": self.x[idx], "y": self.y[idx]}

class ModularMLP(nn.Module):
    def __init__(self, p, d_model=128, hidden=512):
        super().__init__()
        self.p = p
        self.emb = nn.Embedding(p, d_model)
        self.scalar_proj = nn.Linear(2, d_model)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, p))
    def forward(self, x):
        x_norm = (x.float() / (self.p - 1)) * 2.0 - 1.0
        h = self.emb(x[:, 0]) + self.emb(x[:, 1]) + self.scalar_proj(x_norm)
        return self.net(h)

def all_mod_pairs(p):
    pairs = np.array([(a, b) for a in range(p) for b in range(p)], dtype=np.int64)
    labels = np.array([(a + b) % p for a, b in pairs], dtype=np.int64)
    return pairs, labels

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        logits = model(batch["x"]); loss = criterion(logits, batch["y"])
        preds = logits.argmax(dim=-1)
        total_loss += loss.item() * batch["y"].size(0)
        total_correct += (preds == batch["y"]).sum().item()
        total_n += batch["y"].size(0)
    return total_loss / max(total_n, 1), total_correct / max(total_n, 1)

def make_loader(ds, batch_size, shuffle, loader_seed=None):
    gen = None
    if loader_seed is not None:
        gen = torch.Generator(); gen.manual_seed(loader_seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=gen)

def cycle_loader(loader):
    while True:
        for batch in loader: yield batch

@torch.no_grad()
def inject_gradient_noise(model, ref_state, mode="none", alpha=0.08, iso_alpha=0.01):
    # VERBATIM. mode in {none, aligned, anti}. (iso is realized by calling with
    # mode="aligned", alpha=0.0 -> correlated term vanishes, isotropic remains.)
    if mode == "none":
        return
    named = [(name, p) for name, p in model.named_parameters() if p.grad is not None]
    if not named:
        return
    grads = [p.grad.detach() for _, p in named]
    grad_norm = torch.sqrt(torch.stack([(g**2).sum() for g in grads]).sum() + 1e-12)
    dirs = []
    dir_norm_sq = torch.tensor(0.0, device=device)
    for name, p in named:
        d = p.detach() - ref_state[name].to(device)
        dirs.append(d); dir_norm_sq += (d**2).sum()
    dir_norm = torch.sqrt(dir_norm_sq + 1e-12)
    rand_tensors = [torch.randn_like(p.grad) for _, p in named]
    rand_norm = torch.sqrt(torch.stack([(r**2).sum() for r in rand_tensors]).sum() + 1e-12)
    sign = 1.0 if mode == "aligned" else -1.0
    for (_, p), d, r in zip(named, dirs, rand_tensors):
        correlated = sign * alpha * grad_norm * d / dir_norm
        isotropic = iso_alpha * grad_norm * r / rand_norm
        p.grad.add_(correlated + isotropic)
# --------------------------- end verbatim ---------------------------

def train_teacher(p, base_state, loader, criterion, max_steps=3500, lr=2e-3):
    """Verbatim training loop of the original train_reference_model."""
    model = ModularMLP(p).to(device); model.load_state_dict(base_state)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    it = cycle_loader(loader)
    for _ in range(max_steps):
        model.train(); batch = next(it)
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(batch["x"]), batch["y"]); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    return {k: v.detach().clone().to(device) for k, v in model.state_dict().items()}

# condition -> (inject mode, alpha multiplier). iso = aligned with alpha forced 0.
COND_MAP = {"none": ("none", 1.0), "aligned": ("aligned", 1.0),
            "anti": ("anti", 1.0), "iso": ("aligned", 0.0)}

def train_condition(condition, ref_state, p, base_state, train_eval_loader, val_loader,
                    train_ds, criterion, hp, run_seed, eval_interval=250):
    mode, amul = COND_MAP[condition]
    torch.manual_seed(run_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(run_seed)
    train_loader = make_loader(train_ds, hp["batch_size"], shuffle=True, loader_seed=run_seed + 777)
    model = ModularMLP(p).to(device); model.load_state_dict(base_state)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    it = cycle_loader(train_loader)
    fit_step, grok_step = None, None
    val_curve = []
    for step in range(1, hp["max_steps"] + 1):
        model.train(); batch = next(it)
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(batch["x"]), batch["y"]); loss.backward()
        inject_gradient_noise(model, ref_state, mode=mode,
                              alpha=hp["noise_alpha"] * amul, iso_alpha=hp["iso_alpha"])
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if step == 1 or step % eval_interval == 0 or step == hp["max_steps"]:
            _, train_acc = evaluate(model, train_eval_loader, criterion)
            _, val_acc = evaluate(model, val_loader, criterion)
            if fit_step is None and train_acc >= 0.99: fit_step = step
            if grok_step is None and fit_step is not None and val_acc >= 0.95: grok_step = step
            val_curve.append([step, round(val_acc, 4)])
    _, fv = evaluate(model, val_loader, criterion)
    _, ft = evaluate(model, train_eval_loader, criterion)
    return {"grok_step": grok_step, "fit_step": fit_step,
            "final_val_acc": round(fv, 4), "final_train_acc": round(ft, 4),
            "val_curve": val_curve}

def build_data(p, train_frac, split_seed):
    pairs, _ = all_mod_pairs(p)
    rng = np.random.default_rng(split_seed)
    pairs = pairs[rng.permutation(len(pairs))]
    labels = np.array([(int(a) + int(b)) % p for a, b in pairs], dtype=np.int64)
    n_train = int(train_frac * len(labels))
    return (pairs[:n_train], labels[:n_train], pairs[n_train:], labels[n_train:], pairs, labels)

def derive_seed(ms, cond):
    return (ms * 100003 + sum(ord(c) for c in cond)) % 2_000_000

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=59)
    ap.add_argument("--train_frac", type=float, default=0.30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--max_steps", type=int, default=14000)
    ap.add_argument("--out", default="d1_results.json")
    args = ap.parse_args()
    hp = {"name": "bs1024_lr8e-4_wd1e-1", "batch_size": 1024, "lr": 8e-4,
          "weight_decay": 1e-1, "max_steps": args.max_steps,
          "noise_alpha": 0.08, "iso_alpha": 0.01}
    crit = nn.CrossEntropyLoss(); p = args.p
    ref_types = ["REPLICATE", "DECOY", "TRAIN_ONLY"]
    print(f"device={device} torch={torch.__version__} seeds={args.seeds} max_steps={args.max_steps}", flush=True)
    results, ref_diag_all = [], {}
    t0 = time.time()
    for ms in args.seeds:
        tr_p, tr_l, va_p, va_l, full_p, full_l = build_data(p, args.train_frac, split_seed=ms)
        train_ds = ModAddDataset(tr_p, tr_l); val_ds = ModAddDataset(va_p, va_l)
        full_ds = ModAddDataset(full_p, full_l)
        train_eval_loader = make_loader(train_ds, 2048, False)
        val_loader = make_loader(val_ds, 2048, False)
        torch.manual_seed(ms)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(ms)
        base = ModularMLP(p).to(device)
        base_state = {k: v.detach().clone().to(device) for k, v in base.state_dict().items()}
        # --- build the three reference teachers (identical training, different data) ---
        refs = {}
        refs["REPLICATE"] = train_teacher(p, base_state,
            make_loader(full_ds, 1024, True, loader_seed=ms + 999), crit)
        rngp = np.random.default_rng(10000 + ms); perm = rngp.permutation(p)
        while np.all(perm == np.arange(p)): perm = rngp.permutation(p)
        decoy_ds = ModAddDataset(full_p, perm[full_l])
        refs["DECOY"] = train_teacher(p, base_state,
            make_loader(decoy_ds, 1024, True, loader_seed=ms + 998), crit)
        refs["TRAIN_ONLY"] = train_teacher(p, base_state,
            make_loader(train_ds, 1024, True, loader_seed=ms + 997), crit)
        # diagnostic: each teacher's accuracy on the TRUE val labels (the smoking gun)
        diag = {}
        for rt in ref_types:
            m = ModularMLP(p).to(device); m.load_state_dict(refs[rt])
            _, tv = evaluate(m, val_loader, crit); _, tt = evaluate(m, train_eval_loader, crit)
            diag[rt] = {"true_val_acc": round(tv, 4), "true_train_acc": round(tt, 4)}
        ref_diag_all[ms] = diag
        print(f"\nseed={ms} TEACHER true_val_acc: "
              f"REPLICATE={diag['REPLICATE']['true_val_acc']} "
              f"DECOY={diag['DECOY']['true_val_acc']} "
              f"TRAIN_ONLY={diag['TRAIN_ONLY']['true_val_acc']}", flush=True)
        # --- baselines (ref-agnostic): none, iso ---
        for cond in ["none", "iso"]:
            r = train_condition(cond, refs["REPLICATE"], p, base_state,
                                 train_eval_loader, val_loader, train_ds, crit, hp, derive_seed(ms, cond))
            rec = {"seed": ms, "ref_type": "BASELINE", "condition": cond,
                   "grok_step": r["grok_step"], "fit_step": r["fit_step"],
                   "final_val_acc": r["final_val_acc"], "final_train_acc": r["final_train_acc"],
                   "ref_true_val_acc": None, "val_curve": r["val_curve"]}
            results.append(rec)
            print(f"  seed={ms} {'BASELINE':10s} {cond:8s} grok_step={r['grok_step']} "
                  f"val_acc={r['final_val_acc']} [{time.time()-t0:.0f}s]", flush=True)
        # --- directional conditions x reference types (shared run_seed per condition) ---
        for cond in ["aligned", "anti"]:
            rs = derive_seed(ms, cond)
            for rt in ref_types:
                r = train_condition(cond, refs[rt], p, base_state,
                                     train_eval_loader, val_loader, train_ds, crit, hp, rs)
                rec = {"seed": ms, "ref_type": rt, "condition": cond,
                       "grok_step": r["grok_step"], "fit_step": r["fit_step"],
                       "final_val_acc": r["final_val_acc"], "final_train_acc": r["final_train_acc"],
                       "ref_true_val_acc": diag[rt]["true_val_acc"], "val_curve": r["val_curve"]}
                results.append(rec)
                print(f"  seed={ms} {rt:10s} {cond:8s} grok_step={r['grok_step']} "
                      f"val_acc={r['final_val_acc']} (ref_true_val={diag[rt]['true_val_acc']}) "
                      f"[{time.time()-t0:.0f}s]", flush=True)
        # incremental save so a crash never loses completed seeds
        json.dump({"meta": {"p": p, "train_frac": args.train_frac, "seeds": args.seeds,
                            "hp": hp, "device": str(device), "elapsed_s": round(time.time()-t0, 1),
                            "seeds_done": [s for s in args.seeds if s <= ms]},
                   "ref_diag": ref_diag_all, "results": results},
                  open(args.out, "w"), indent=1)
    # ----------------------------- summary -----------------------------
    def summ(rt, cond):
        rs = [r for r in results if r["ref_type"] == rt and r["condition"] == cond]
        if not rs: return None
        gr = float(np.mean([1.0 if r["grok_step"] is not None else 0.0 for r in rs]))
        fin = [r["grok_step"] for r in rs if r["grok_step"] is not None]
        vals = [r["final_val_acc"] for r in rs]
        return gr, (int(np.median(fin)) if fin else None), float(np.mean(vals)), float(np.std(vals))
    print("\n========== D1 SUMMARY (over %d seeds) ==========" % len(args.seeds))
    print(f"{'ref_type':12s}{'condition':9s}{'grok_rate':10s}{'med_grok_step':15s}{'val_acc mean±sd':18s}")
    for rt, cond in [("BASELINE", "none"), ("BASELINE", "iso"),
                     ("REPLICATE", "aligned"), ("DECOY", "aligned"), ("TRAIN_ONLY", "aligned"),
                     ("REPLICATE", "anti"), ("DECOY", "anti"), ("TRAIN_ONLY", "anti")]:
        s = summ(rt, cond)
        if s: print(f"{rt:12s}{cond:9s}{s[0]:<10.2f}{str(s[1]):15s}{s[2]:.3f}±{s[3]:.3f}")
    print("\nTeacher true-val-acc by seed (the leakage diagnostic):")
    for ms in args.seeds:
        d = ref_diag_all.get(ms, {})
        print(f"  seed={ms}: " + "  ".join(f"{rt}={d.get(rt,{}).get('true_val_acc')}" for rt in ref_types))
    print(f"\n[saved] {args.out}  total_elapsed={time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
