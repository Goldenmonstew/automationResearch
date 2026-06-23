#!/usr/bin/env python
"""D1c: confound ablation for noise_alignment_double_descent (idx2).

Original "aligned noise" = gradient of CE on a CLEAN anchor set, added to the
30%-label-noise training gradient -> it injects clean supervision the noisy
training set lacks. Faithful minimal change: keep MLP/SGD/widths/conditions and
the 30%-label-noise setup; only vary the ANCHOR TYPE that the aligned/anti
"noise" points along:

  CLEAN    : anchor CE on CLEAN labels (= the original aligned/anti)
  SHUFFLED : anchor CE on SHUFFLED anchor labels (a gradient, but no clean signal)
  RANDOMDIR: a fixed random unit direction (pure undirected noise)

Confound prediction: aligned beats baseline / anti degrades it ONLY for CLEAN;
under SHUFFLED and RANDOMDIR the aligned/anti arms collapse to baseline -> the
"double-descent peak control" effect is clean-supervision injection, not noise
alignment. Uses sklearn digits (what the original effectively fell back to: fast,
deterministic, no download). Pure torch.
"""
import os, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DS(Dataset):
    def __init__(self, x, y_clean, y_train=None):
        self.x = x.float(); self.yc = y_clean.long()
        self.yt = (y_clean if y_train is None else y_train).long()
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return {"x": self.x[i], "y": self.yt[i], "clean_y": self.yc[i]}


class MLP(nn.Module):
    def __init__(self, d, w, k):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(d, w), nn.ReLU(),
                                 nn.Linear(w, w), nn.ReLU(), nn.Linear(w, k))
    def forward(self, x): return self.net(x)


def set_seed(sd):
    np.random.seed(sd); torch.manual_seed(sd)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(sd)


@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval(); ls = 0.0; c = 0; n = 0
    for b in loader:
        b = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}
        lo = model(b["x"]); ls += crit(lo, b["clean_y"]).item() * b["x"].size(0)
        c += (lo.argmax(1) == b["clean_y"]).sum().item(); n += b["x"].size(0)
    return ls / max(n, 1), c / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--out", default="d1c_results.json")
    args = ap.parse_args()

    d = load_digits()
    X = torch.tensor(d.images[:, None, :, :], dtype=torch.float32) / 16.0
    Y = torch.tensor(d.target, dtype=torch.long)
    itr, ite = train_test_split(np.arange(len(Y)), test_size=0.25, stratify=Y.numpy(), random_state=0)
    xtr, xte, ytr, yte = X[itr], X[ite], Y[itr], Y[ite]
    m, s = xtr.mean(), xtr.std().clamp_min(1e-6); xtr = (xtr - m) / s; xte = (xte - m) / s
    num_classes = 10; input_dim = int(np.prod(xtr.shape[1:]))

    g = torch.Generator().manual_seed(123); N = len(ytr); perm = torch.randperm(N, generator=g)
    tn = int(N * 0.8); tr_i, an_i = perm[:tn], perm[tn:]
    xt, ytc = xtr[tr_i], ytr[tr_i]; xa, ya = xtr[an_i], ytr[an_i]
    noise = torch.rand(tn, generator=g) < 0.30; ytn = ytc.clone()
    off = torch.randint(1, num_classes, (int(noise.sum()),), generator=g)
    ytn[noise] = (ytn[noise] + off) % num_classes

    widths = [16, 32, 64, 128, 256, 512, 1024]
    anchor_types = ["CLEAN", "SHUFFLED", "RANDOMDIR"]
    noise_strength = 0.35; crit = nn.CrossEntropyLoss()
    results = []; t0 = time.time()
    print(f"device={device} torch={torch.__version__} seeds={args.seeds} (sklearn digits, 30% label noise)", flush=True)

    for sd in args.seeds:
        rng = np.random.default_rng(1000 + sd)
        ya_shuf = torch.tensor(rng.permutation(ya.numpy()), dtype=torch.long)
        blocks = [("CLEAN", "baseline")] + [(at, c) for at in anchor_types for c in ("aligned", "anti")]
        for at, cond in blocks:
            an_lab = ya_shuf if at == "SHUFFLED" else ya
            tl = DataLoader(DS(xt, ytc, ytn), batch_size=128, shuffle=True, num_workers=0)
            al = DataLoader(DS(xa, an_lab), batch_size=min(256, len(an_lab)), shuffle=True, num_workers=0)
            vl = DataLoader(DS(xte, yte), batch_size=256, shuffle=False, num_workers=0)
            sign = 0.0 if cond == "baseline" else (1.0 if cond == "aligned" else -1.0)
            for w in widths:
                set_seed(sd)
                model = MLP(input_dim, w, num_classes).to(device)
                opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
                ab = next(iter(al)); ab = {k: v.to(device) for k, v in ab.items() if isinstance(v, torch.Tensor)}
                randdir = None
                for _ in range(args.epochs):
                    model.train()
                    for b in tl:
                        b = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}
                        opt.zero_grad(set_to_none=True)
                        loss = crit(model(b["x"]), b["y"]); loss.backward()
                        if sign != 0.0:
                            params = [p for p in model.parameters() if p.requires_grad]
                            cn = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in params if p.grad is not None) + 1e-12)
                            if at == "RANDOMDIR":
                                if randdir is None: randdir = [torch.randn_like(p) for p in params]
                                dn = torch.sqrt(sum((r ** 2).sum() for r in randdir) + 1e-12)
                                sc = float((sign * noise_strength * cn / dn).detach().cpu())
                                with torch.no_grad():
                                    for p, r in zip(params, randdir):
                                        if p.grad is not None: p.grad.add_(r, alpha=sc)
                            else:
                                aloss = crit(model(ab["x"]), ab["clean_y"])
                                ag = torch.autograd.grad(aloss, params, allow_unused=True)
                                an = torch.sqrt(sum((a.detach() ** 2).sum() for a in ag if a is not None) + 1e-12)
                                sc = float((sign * noise_strength * cn / an).detach().cpu())
                                with torch.no_grad():
                                    for p, a in zip(params, ag):
                                        if a is not None and p.grad is not None: p.grad.add_(a.detach(), alpha=sc)
                        opt.step()
                _, vacc = evaluate(model, vl, crit)
                results.append({"seed": sd, "anchor_type": at, "condition": cond, "width": w,
                                "val_acc": round(vacc, 4), "val_err": round(1 - vacc, 4)})
            print(f"  seed={sd} {at:9s} {cond:8s} done [{time.time()-t0:.0f}s]", flush=True)
        json.dump({"meta": {"seeds": args.seeds, "widths": widths, "device": str(device),
                            "epochs": args.epochs, "elapsed_s": round(time.time()-t0, 1)},
                   "results": results}, open(args.out, "w"), indent=1)

    def curve(at, cond):
        out = []
        for w in widths:
            vs = [r["val_err"] for r in results if r["anchor_type"] == at and r["condition"] == cond and r["width"] == w]
            out.append(round(float(np.mean(vs)), 3) if vs else None)
        return out
    print("\n===== D1c SUMMARY (mean val_err by width; peak = argmax width) =====")
    base = curve("CLEAN", "baseline")
    print("%-22s %s peak_w=%s" % ("baseline", base, widths[int(np.argmax(base))]))
    for at in anchor_types:
        for cond in ("aligned", "anti"):
            c = curve(at, cond)
            print("%-22s %s peak_w=%s" % (at + "/" + cond, c, widths[int(np.argmax(c))]))
    print("\nConfound test: CLEAN/anti should be far worse than baseline; SHUFFLED & RANDOMDIR arms should collapse to baseline.")
    print(f"[saved] {args.out} elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
