#!/usr/bin/env python
"""D1b: decompose the DECOY residual -- feature-distillation vs answer-leakage.

Main D1 (d1_confound_ablation.py) showed `aligned` groks under REPLICATE (ref =
val answers; 5/5, ~2750 steps) and DECOY (ref = permuted-output teacher; 5/5,
~12750 steps, 4.6x slower) but NOT under TRAIN_ONLY / none / iso / anti. The DECOY
residual is hypothesized to be distillation of the *generalizing modular features*
(the circular/Fourier embeddings), which the permuted-output teacher still learns
(only its readout is wrong).

This adds DECOY_RANDOM: a teacher trained on the FULL table (identical val-row
exposure to REPLICATE/DECOY) but with a FIXED RANDOM label table -> it can only
memorize and learns NO generalizing structure. All three teachers see every val
input pair; they differ only in whether the labels carry modular structure.

  REPLICATE    : full table, TRUE modular labels      (true features + true readout)
  DECOY_PERM   : full table, PERMUTED modular labels   (true features + wrong readout)
  DECOY_RANDOM : full table, RANDOM label table        (no generalizing structure)

Prediction (confound = generalizing-feature distillation): aligned grok rate is
REPLICATE(fast) > DECOY_PERM(slow) >> DECOY_RANDOM(none). If DECOY_RANDOM does NOT
grok, "mere exposure to the val rows" is ruled out -- the teacher must have learned
the generalizing function. run_seed matches the main run so REPLICATE/DECOY_PERM
reproduce their main-run numbers exactly.
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
    tl, tc, tn = 0.0, 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        logits = model(batch["x"]); loss = criterion(logits, batch["y"])
        preds = logits.argmax(dim=-1)
        tl += loss.item() * batch["y"].size(0); tc += (preds == batch["y"]).sum().item(); tn += batch["y"].size(0)
    return tl / max(tn, 1), tc / max(tn, 1)

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
    if mode == "none": return
    named = [(name, p) for name, p in model.named_parameters() if p.grad is not None]
    if not named: return
    grads = [p.grad.detach() for _, p in named]
    grad_norm = torch.sqrt(torch.stack([(g**2).sum() for g in grads]).sum() + 1e-12)
    dirs = []; dn = torch.tensor(0.0, device=device)
    for name, p in named:
        d = p.detach() - ref_state[name].to(device); dirs.append(d); dn += (d**2).sum()
    dir_norm = torch.sqrt(dn + 1e-12)
    rand_tensors = [torch.randn_like(p.grad) for _, p in named]
    rand_norm = torch.sqrt(torch.stack([(r**2).sum() for r in rand_tensors]).sum() + 1e-12)
    sign = 1.0 if mode == "aligned" else -1.0
    for (_, p), d, r in zip(named, dirs, rand_tensors):
        p.grad.add_(sign * alpha * grad_norm * d / dir_norm + iso_alpha * grad_norm * r / rand_norm)
# --------------------------- end verbatim ---------------------------

def train_teacher(p, base_state, loader, criterion, max_steps=3500, lr=2e-3):
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

def train_aligned(ref_state, p, base_state, train_eval_loader, val_loader, train_ds,
                  criterion, hp, run_seed, eval_interval=250):
    torch.manual_seed(run_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(run_seed)
    train_loader = make_loader(train_ds, hp["batch_size"], shuffle=True, loader_seed=run_seed + 777)
    model = ModularMLP(p).to(device); model.load_state_dict(base_state)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    it = cycle_loader(train_loader)
    fit_step = grok_step = None
    for step in range(1, hp["max_steps"] + 1):
        model.train(); batch = next(it)
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(batch["x"]), batch["y"]); loss.backward()
        inject_gradient_noise(model, ref_state, mode="aligned",
                              alpha=hp["noise_alpha"], iso_alpha=hp["iso_alpha"])
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if step == 1 or step % eval_interval == 0 or step == hp["max_steps"]:
            _, ta = evaluate(model, train_eval_loader, criterion)
            _, va = evaluate(model, val_loader, criterion)
            if fit_step is None and ta >= 0.99: fit_step = step
            if grok_step is None and fit_step is not None and va >= 0.95: grok_step = step
    _, fv = evaluate(model, val_loader, criterion)
    return {"grok_step": grok_step, "final_val_acc": round(fv, 4)}

def build_data(p, train_frac, split_seed):
    pairs, _ = all_mod_pairs(p)
    rng = np.random.default_rng(split_seed)
    pairs = pairs[rng.permutation(len(pairs))]
    labels = np.array([(int(a) + int(b)) % p for a, b in pairs], dtype=np.int64)
    n_train = int(train_frac * len(labels))
    return pairs[:n_train], labels[:n_train], pairs[n_train:], labels[n_train:], pairs, labels

def derive_seed(ms, cond):  # identical to main run -> REPLICATE/DECOY_PERM reproduce
    return (ms * 100003 + sum(ord(c) for c in cond)) % 2_000_000

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=59)
    ap.add_argument("--train_frac", type=float, default=0.30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--max_steps", type=int, default=14000)
    ap.add_argument("--out", default="d1b_results.json")
    args = ap.parse_args()
    hp = {"batch_size": 1024, "lr": 8e-4, "weight_decay": 1e-1,
          "max_steps": args.max_steps, "noise_alpha": 0.08, "iso_alpha": 0.01}
    crit = nn.CrossEntropyLoss(); p = args.p
    refs_order = ["REPLICATE", "DECOY_PERM", "DECOY_RANDOM"]
    print(f"device={device} torch={torch.__version__} seeds={args.seeds}", flush=True)
    results, diag_all = [], {}
    t0 = time.time()
    for ms in args.seeds:
        tr_p, tr_l, va_p, va_l, full_p, full_l = build_data(p, args.train_frac, ms)
        train_ds = ModAddDataset(tr_p, tr_l); val_ds = ModAddDataset(va_p, va_l)
        train_eval_loader = make_loader(train_ds, 2048, False); val_loader = make_loader(val_ds, 2048, False)
        torch.manual_seed(ms)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(ms)
        base = ModularMLP(p).to(device)
        base_state = {k: v.detach().clone().to(device) for k, v in base.state_dict().items()}
        # teachers
        refs = {}
        refs["REPLICATE"] = train_teacher(p, base_state, make_loader(ModAddDataset(full_p, full_l), 1024, True, ms + 999), crit)
        rngp = np.random.default_rng(10000 + ms); perm = rngp.permutation(p)
        while np.all(perm == np.arange(p)): perm = rngp.permutation(p)
        refs["DECOY_PERM"] = train_teacher(p, base_state, make_loader(ModAddDataset(full_p, perm[full_l]), 1024, True, ms + 998), crit)
        rngr = np.random.default_rng(20000 + ms)
        rand_labels = rngr.integers(0, p, size=len(full_l)).astype(np.int64)
        refs["DECOY_RANDOM"] = train_teacher(p, base_state, make_loader(ModAddDataset(full_p, rand_labels), 1024, True, ms + 996), crit)
        # diagnostics: each teacher on TRUE val labels + on its OWN training labels
        diag = {}
        own = {"REPLICATE": full_l, "DECOY_PERM": perm[full_l], "DECOY_RANDOM": rand_labels}
        for rt in refs_order:
            m = ModularMLP(p).to(device); m.load_state_dict(refs[rt])
            _, tv = evaluate(m, val_loader, crit)
            _, ot = evaluate(m, make_loader(ModAddDataset(full_p, own[rt]), 2048, False), crit)
            diag[rt] = {"true_val_acc": round(tv, 4), "own_label_acc": round(ot, 4)}
        diag_all[ms] = diag
        print(f"\nseed={ms} teachers: " + "  ".join(
            f"{rt}(trueval={diag[rt]['true_val_acc']},memorized_own={diag[rt]['own_label_acc']})" for rt in refs_order), flush=True)
        rs = derive_seed(ms, "aligned")
        for rt in refs_order:
            r = train_aligned(refs[rt], p, base_state, train_eval_loader, val_loader, train_ds, crit, hp, rs)
            rec = {"seed": ms, "ref_type": rt, "condition": "aligned",
                   "grok_step": r["grok_step"], "final_val_acc": r["final_val_acc"],
                   "ref_true_val_acc": diag[rt]["true_val_acc"]}
            results.append(rec)
            print(f"  seed={ms} {rt:13s} aligned grok_step={r['grok_step']} val_acc={r['final_val_acc']} "
                  f"(ref_true_val={diag[rt]['true_val_acc']}) [{time.time()-t0:.0f}s]", flush=True)
        json.dump({"meta": {"p": p, "seeds": args.seeds, "hp": hp, "device": str(device),
                            "seeds_done": [s for s in args.seeds if s <= ms], "elapsed_s": round(time.time()-t0, 1)},
                   "diag": diag_all, "results": results}, open(args.out, "w"), indent=1)
    print("\n========== D1b SUMMARY (aligned, over %d seeds) ==========" % len(args.seeds))
    print(f"{'ref_type':14s}{'grok_rate':10s}{'med_grok_step':15s}{'val_acc mean±sd':18s}")
    for rt in refs_order:
        rsr = [r for r in results if r["ref_type"] == rt]
        gr = float(np.mean([1.0 if r["grok_step"] is not None else 0.0 for r in rsr]))
        fin = [r["grok_step"] for r in rsr if r["grok_step"] is not None]
        vals = [r["final_val_acc"] for r in rsr]
        print(f"{rt:14s}{gr:<10.2f}{str(int(np.median(fin)) if fin else None):15s}{np.mean(vals):.3f}±{np.std(vals):.3f}")
    print(f"\n[saved] {args.out}  total_elapsed={time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
