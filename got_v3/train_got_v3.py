#!/usr/bin/env python3
"""
GOT-v3 — Geometry-first Geometric Operator Autoencoder

3-phase training pipeline:

  Phase 1 — coefficient geometry (no Δζ(3) signal):
    python3 got_v3/train_got_v3.py --data data/cf_large \
        --config got_v3/configs/ae_pretrain.yaml

  Phase 2 — arithmetic heads with frozen encoder:
    python3 got_v3/train_got_v3.py --data data/cf_large \
        --config got_v3/configs/target_train.yaml \
        --load ckpt/got_v3_ae_pretrain_k3.pt

  Phase 3 — joint fine-tuning:
    python3 got_v3/train_got_v3.py --data data/cf_large \
        --config got_v3/configs/joint.yaml \
        --load ckpt/got_v3_target_train_k3.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent))
from dataset import CFDataset
from models  import GOTv3
from losses  import compute_losses


# ─── Loss-weight presets ──────────────────────────────────────────────────────

WEIGHT_PRESETS: Dict[str, Dict[str, float]] = {
    "ae_pretrain": {
        "recon": 1.0, "nbr": 1.0, "op": 0.1,   # op uses unit-sphere delta → bounded [0,2]
        "contrast": 0.2, "mix": 0.2,
        "delta": 0.0, "conv": 0.0, "plateau": 0.0,
        "z_reg": 0.005,   # balances VICReg var_reg → |z| stable at ~2-3
        "var_reg": 0.1,
    },
    "target_train": {
        "recon": 0.0, "nbr": 0.0, "op": 0.0,
        "contrast": 0.0, "mix": 0.1,
        "delta": 1.0, "conv": 0.3, "plateau": 0.5,
        "z_reg": 0.0005,
        "var_reg": 0.0,
    },
    "joint": {
        "recon": 0.2, "nbr": 0.2, "op": 0.5,
        "contrast": 0.1, "mix": 0.2,
        "delta": 1.0, "conv": 0.3, "plateau": 0.5,
        "z_reg": 0.0005,
        "var_reg": 0.05,   # softer anti-collapse during joint fine-tuning
    },
}


# ─── Training utilities ───────────────────────────────────────────────────────

def _move(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def train_one_epoch(
    model:     GOTv3,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    str,
    weights:   Dict[str, float],
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    model.train()
    totals: Dict[str, float] = {}
    n = 0
    for batch in loader:
        batch = _move(batch, device)
        out   = model(batch["an"], batch["bn"])
        loss, logs = compute_losses(model, batch, out, weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        for k, v in logs.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate(
    model:   GOTv3,
    loader:  DataLoader,
    device:  str,
    weights: Dict[str, float],
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    n = 0
    d_preds, d_trues = [], []
    c_preds, c_trues = [], []
    p_preds, p_trues = [], []

    for batch in loader:
        batch = _move(batch, device)
        out   = model(batch["an"], batch["bn"])
        _, logs = compute_losses(model, batch, out, weights)
        for k, v in logs.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1
        d_preds.append(out["delta_pred"].cpu())
        d_trues.append(batch["delta"].cpu())
        c_preds.append(out["component_logits"].argmax(1).cpu())
        c_trues.append(batch["component"].cpu())
        p_preds.append(torch.sigmoid(out["plateau_logit"]).cpu())
        p_trues.append(batch["plateau"].cpu())

    logs = {k: v / max(n, 1) for k, v in totals.items()}

    dp, dt = torch.cat(d_preds).numpy(), torch.cat(d_trues).numpy()
    ss_res = float(((dt - dp) ** 2).sum())
    ss_tot = float(((dt - dt.mean()) ** 2).sum()) + 1e-9
    logs["delta_r2"]      = 1.0 - ss_res / ss_tot
    logs["delta_mse"]     = float(((dp - dt) ** 2).mean())

    cp, ct = torch.cat(c_preds).numpy(), torch.cat(c_trues).numpy()
    logs["component_acc"] = float((cp == ct).mean())

    pp, pt = torch.cat(p_preds).numpy(), torch.cat(p_trues).numpy()
    logs["plateau_acc"]   = float(((pp > 0.5) == (pt > 0.5)).mean())

    return logs


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GOT-v3 training")
    p.add_argument("--data",           required=True,
                   help="Path to data directory (an_coeffs.npy etc.)")
    p.add_argument("--mode",           default="ae_pretrain",
                   choices=["ae_pretrain", "target_train", "joint"])
    p.add_argument("--config",         default=None,
                   help="YAML config file (CLI args override config values)")
    p.add_argument("--k",              type=int,   default=3)
    p.add_argument("--d_model",        type=int,   default=128)
    p.add_argument("--layers",         type=int,   default=4)
    p.add_argument("--heads",          type=int,   default=8)
    p.add_argument("--dropout",        type=float, default=0.1)
    p.add_argument("--n_components",   type=int,   default=20)
    p.add_argument("--epochs",         type=int,   default=80)
    p.add_argument("--batch",          type=int,   default=512)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--val_frac",       type=float, default=0.1)
    p.add_argument("--load",           default=None,
                   help="Checkpoint to load (strict=False)")
    p.add_argument("--save",           default=None,
                   help="Override checkpoint save path")
    p.add_argument("--freeze_encoder", action="store_true",
                   help="Freeze encoder+bottleneck (recommended for target_train)")
    p.add_argument("--ablate_k",       action="store_true",
                   help="Run ae_pretrain for k in {2,3,4,5,6,7} and save summary")
    p.add_argument("--ablate_ks",      type=str, default=None,
                   help="Comma-separated custom k list, e.g. '3,4,5,6,7'")
    p.add_argument("--device",         default="auto")
    p.add_argument("--ablate_epochs",  type=int, default=None,
                   help="Epoch count override for each ablation run")

    # Two-pass: first get --config, set YAML defaults, then full parse
    pre, _ = p.parse_known_args()
    if pre.config is not None:
        try:
            import yaml
            cfg = yaml.safe_load(Path(pre.config).read_text())
            valid = {a.dest for a in p._actions}
            p.set_defaults(**{k: v for k, v in cfg.items() if k in valid})
        except ImportError:
            print("[warn] PyYAML not installed — ignoring --config")

    return p.parse_args()


def _resolve_device(arg: str) -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args=None):
    if args is None:
        args = parse_args()
    device = _resolve_device(args.device)

    ckpt_dir   = Path("ckpt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_path  = Path(args.save) if args.save else \
                 ckpt_dir / f"got_v3_{args.mode}_k{args.k}.pt"

    print("=" * 72)
    print("GOT-v3 — Geometry-first Geometric Operator Autoencoder")
    print("=" * 72)
    print(f"  mode={args.mode}  k={args.k}  d={args.d_model}  "
          f"L={args.layers}  heads={args.heads}  device={device}")

    ds    = CFDataset(args.data, n_components=args.n_components)
    n_val = int(len(ds) * args.val_frac)
    train_ds, val_ds = random_split(
        ds, [len(ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              drop_last=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              drop_last=False, num_workers=0)

    model = GOTv3(
        n_a=ds.an.shape[1], n_b=ds.bn.shape[1],
        k=args.k, d_model=args.d_model, n_heads=args.heads,
        n_layers=args.layers, dropout=args.dropout,
        n_components=args.n_components,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params={n_params:,}")

    if args.load is not None:
        state = torch.load(args.load, map_location=device)
        model.load_state_dict(
            state["model"] if "model" in state else state, strict=False
        )
        print(f"  loaded: {args.load}")

    if args.freeze_encoder:
        model.freeze_encoder()

    weights   = WEIGHT_PRESETS[args.mode].copy()
    print(f"  weights: { {k:v for k,v in weights.items() if v > 0} }")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    log_every = max(1, args.epochs // 20)
    best_val  = float("inf")
    history   = []

    for epoch in range(1, args.epochs + 1):
        t = train_one_epoch(model, train_loader, optimizer, device, weights)
        v = evaluate(model, val_loader, device, weights)
        scheduler.step()

        flag = ""
        if v["total"] < best_val:
            best_val = v["total"]
            torch.save({
                "model":    model.state_dict(),
                "args":     vars(args),
                "weights":  weights,
                "epoch":    epoch,
                "val_logs": v,
            }, save_path)
            flag = "✓"

        history.append({"epoch": epoch, "train": t, "val": v})

        if epoch == 1 or epoch % log_every == 0 or epoch == args.epochs:
            print(
                f"  ep {epoch:04d}/{args.epochs} "
                f"Δr²={v['delta_r2']:+.3f}  "
                f"mix_acc={v['component_acc']:.3f}  "
                f"recon={v.get('recon', 0):.4f}  "
                f"nbr={v.get('nbr', 0):.4f}  "
                f"var={v.get('var_reg', 0):+.3f}  "
                f"op={v.get('op', 0):.4f}  "
                f"total={v['total']:.4f}  {flag}"
            )

    hist_path = save_path.with_suffix(".history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f)

    print(f"\n  checkpoint → {save_path}")
    print(f"  history    → {hist_path}")


def run_ablate_k(args, device: str):
    """Run ae_pretrain for k in {2,3,4,5,6,7} and save a summary JSON."""
    ckpt_dir = Path("ckpt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    epochs   = args.ablate_epochs or args.epochs
    summary  = []

    # Allow custom k list via --ablate_ks "3,4,5,6,7"
    if getattr(args, "ablate_ks", None):
        k_grid = [int(x) for x in args.ablate_ks.split(",")]
    else:
        k_grid = [2, 3, 4, 5, 6, 7]

    ds    = CFDataset(args.data, n_components=args.n_components)
    n_val = int(len(ds) * args.val_frac)
    train_ds, val_ds = random_split(
        ds, [len(ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              drop_last=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              drop_last=False, num_workers=0)

    for k in k_grid:
        print(f"\n{'='*60}\nAblation k={k} ({epochs} epochs)\n{'='*60}")
        save_path = ckpt_dir / f"got_v3_ae_pretrain_k{k}.pt"
        model = GOTv3(
            n_a=ds.an.shape[1], n_b=ds.bn.shape[1],
            k=k, d_model=args.d_model, n_heads=args.heads,
            n_layers=args.layers, dropout=args.dropout,
            n_components=args.n_components,
        ).to(device)

        weights   = WEIGHT_PRESETS["ae_pretrain"].copy()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs,
        )
        best_val = float("inf")
        log_every = max(1, epochs // 10)

        for epoch in range(1, epochs + 1):
            train_one_epoch(model, train_loader, optimizer, device, weights)
            v = evaluate(model, val_loader, device, weights)
            scheduler.step()
            if v["total"] < best_val:
                best_val = v["total"]
                torch.save({"model": model.state_dict(), "args": vars(args),
                            "weights": weights, "epoch": epoch, "val_logs": v},
                           save_path)
            if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
                print(f"  k={k} ep {epoch:03d}/{epochs}  "
                      f"mix_acc={v['component_acc']:.3f}  "
                      f"recon={v.get('recon',0):.4f}  "
                      f"var={v.get('var_reg',0):+.3f}  "
                      f"nbr={v.get('nbr',0):.4f}")

        summary.append({"k": k, "mix_acc": v["component_acc"],
                        "recon": v.get("recon", 0), "nbr": v.get("nbr", 0),
                        "var_reg": v.get("var_reg", 0), "best_val": best_val,
                        "ckpt": str(save_path)})

    out = ckpt_dir / "ablate_k_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAblation summary → {out}")
    for row in summary:
        print(f"  k={row['k']}  mix={row['mix_acc']:.3f}  "
              f"recon={row['recon']:.4f}  nbr={row['nbr']:.4f}  "
              f"var={row['var_reg']:+.3f}")


if __name__ == "__main__":
    _args   = parse_args()
    _device = _resolve_device(_args.device)
    if _args.ablate_k:
        run_ablate_k(_args, _device)
    else:
        main(_args)
