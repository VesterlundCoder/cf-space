#!/usr/bin/env python3
"""
Latent space diagnostics for GOT-v3.

Computes:
    coeff TwoNN dim    — intrinsic dimension of coefficient space (target ≈ 3.0)
    latent TwoNN dim   — intrinsic dimension of z-space (target ≈ 2.5–3.5)
    trustworthiness    — sklearn metric (target > 0.90)
    kNN overlap @k     — fraction of k-NN preserved from coeff → latent (target > 0.40)
    z_var              — mean per-dim variance of z (sanity check; not exploded)

Usage:
    python3 got_v3/eval_latent.py \
        --data  data/cf_large \
        --ckpt  ckpt/got_v3_ae_pretrain_k3.pt \
        --out   results/latent_ae_k3

    # Subsample 10k points for speed (recommended for 286k corpus):
    python3 got_v3/eval_latent.py --data data/cf_large \
        --ckpt ckpt/got_v3_ae_pretrain_k3.pt --n 10000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from dataset import CFDataset
from models  import GOTv3


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def twonn_id(X: np.ndarray, eps: float = 1e-12) -> float:
    """
    TwoNN intrinsic dimension estimator (Facco et al., 2017).

    ID = 1 / mean(log(r2/r1))  where r1, r2 are distances to 1st and 2nd NN.
    """
    from sklearn.neighbors import NearestNeighbors
    nbrs       = NearestNeighbors(n_neighbors=3).fit(X)
    dists, _   = nbrs.kneighbors(X)
    r1         = dists[:, 1] + eps
    r2         = dists[:, 2] + eps
    mu         = r2 / r1
    mu         = mu[np.isfinite(mu) & (mu > 1.0)]
    return 1.0 / float(np.mean(np.log(mu)))


def knn_overlap(X: np.ndarray, Z: np.ndarray, k: int = 15) -> float:
    """
    Mean fraction of k-nearest neighbours shared between coeff-space and latent.
    1.0 = perfect preservation, 0.0 = random.
    """
    from sklearn.neighbors import NearestNeighbors

    def get_idx(M: np.ndarray) -> np.ndarray:
        return NearestNeighbors(n_neighbors=k + 1).fit(M) \
                   .kneighbors(M, return_distance=False)[:, 1:]

    idx_x = get_idx(X)
    idx_z = get_idx(Z)
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(idx_x, idx_z)]))


# ─────────────────────────────────────────────────────────────────────────────
# Encoding
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_all(
    model:      GOTv3,
    ds:         CFDataset,
    device:     str,
    batch_size: int = 2048,
):
    model.eval()
    Z, X_all, deltas, comps = [], [], [], []
    for i in range(0, len(ds), batch_size):
        batch  = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
        an     = torch.stack([b["an"]    for b in batch]).to(device)
        bn     = torch.stack([b["bn"]    for b in batch]).to(device)
        _, z   = model.encode(an, bn)
        Z.append(z.cpu().numpy())
        X_all.append(np.stack([b["coeff"].numpy()   for b in batch]))
        deltas.append(np.array([float(b["delta"])   for b in batch]))
        comps.append( np.array([int(b["component"]) for b in batch]))
    return (
        np.concatenate(Z),
        np.concatenate(X_all),
        np.concatenate(deltas),
        np.concatenate(comps),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="GOT-v3 latent diagnostics")
    p.add_argument("--data",   required=True)
    p.add_argument("--ckpt",   required=True)
    p.add_argument("--out",    default="results/latent_eval")
    p.add_argument("--k_nn",   type=int,   default=15,   help="k for kNN overlap")
    p.add_argument("--n",      type=int,   default=None, help="Subsample N points")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available() else
              "mps"  if torch.backends.mps.is_available() else "cpu") \
             if args.device == "auto" else args.device

    ds   = CFDataset(args.data)
    ckpt = torch.load(args.ckpt, map_location=device)
    ma   = ckpt["args"]

    model = GOTv3(
        n_a=ds.an.shape[1],  n_b=ds.bn.shape[1],
        k=ma["k"],           d_model=ma["d_model"],
        n_heads=ma["heads"], n_layers=ma["layers"],
        dropout=ma["dropout"], n_components=ma["n_components"],
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)

    print(f"encoding {len(ds):,} CFs on {device} …")
    Z, X, delta, comp = encode_all(model, ds, device)

    if args.n is not None and args.n < len(Z):
        idx = np.random.default_rng(0).choice(len(Z), args.n, replace=False)
        Z, X, delta, comp = Z[idx], X[idx], delta[idx], comp[idx]
        print(f"subsampled to {len(Z):,} points")

    print(f"TwoNN on coeff ({X.shape[1]}D) …")
    id_x = twonn_id(X)

    print(f"TwoNN on latent ({Z.shape[1]}D) …")
    id_z = twonn_id(Z)

    print(f"kNN overlap (k={args.k_nn}) …")
    overlap = knn_overlap(X, Z, k=args.k_nn)

    tw = float("nan")
    try:
        from sklearn.manifold import trustworthiness
        print("trustworthiness …")
        tw = float(trustworthiness(X, Z, n_neighbors=args.k_nn))
    except Exception as e:
        print(f"[warn] trustworthiness failed: {e}")

    z_var = float(Z.var(0).mean())

    print()
    print("=" * 62)
    print("  Latent diagnostics")
    print("=" * 62)
    print(f"  coeff TwoNN dim   : {id_x:.3f}   (reference)")
    print(f"  latent TwoNN dim  : {id_z:.3f}   (target ≈ 2.5–3.5)")
    print(f"  trustworthiness   : {tw:.3f}   (target > 0.90)")
    print(f"  kNN overlap @{args.k_nn:2d}   : {overlap:.3f}   (target > 0.40)")
    print(f"  z_var (mean)      : {z_var:.4f}")
    print("=" * 62)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "Z.npy",         Z)
    np.save(out_dir / "coeff.npy",     X)
    np.save(out_dir / "delta.npy",     delta)
    np.save(out_dir / "component.npy", comp)

    metrics = {
        "coeff_twonn":      round(id_x, 4),
        "latent_twonn":     round(id_z, 4),
        "trustworthiness":  round(tw,   4) if not np.isnan(tw) else None,
        "knn_overlap":      round(overlap, 4),
        "k_nn":             args.k_nn,
        "z_var":            round(z_var, 6),
        "n_points":         len(Z),
        "checkpoint":       str(args.ckpt),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n  arrays  → {out_dir}/Z.npy, coeff.npy, delta.npy, component.npy")
    print(f"  metrics → {out_dir}/metrics.json")

    # ── Plots ────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        Z2 = PCA(n_components=2).fit_transform(Z) if Z.shape[1] > 2 else Z

        for arr, label, fname, cmap in [
            (delta,            "log10|v − ζ(3)|", "latent_delta.png",      "viridis"),
            (comp.astype(float), "component",     "latent_components.png", "tab20"),
        ]:
            plt.figure(figsize=(9, 7))
            sc = plt.scatter(Z2[:, 0], Z2[:, 1], c=arr, s=2, alpha=0.35, cmap=cmap)
            plt.colorbar(sc, label=label)
            plt.title(
                f"GOT-v3 latent  (TwoNN d̂={id_z:.2f}  kNN={overlap:.2f}  "
                f"n={len(Z):,})"
            )
            plt.tight_layout()
            plt.savefig(out_dir / fname, dpi=180)
            plt.close()
        print(f"  plots   → {out_dir}/latent_delta.png, latent_components.png")
    except ImportError:
        print("  (matplotlib/sklearn not available — skipping plots)")


if __name__ == "__main__":
    main()
