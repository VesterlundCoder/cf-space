#!/usr/bin/env python3
"""
compare_operator_geometry.py
═══════════════════════════════════════════════════════════════════════════════
Systematic comparison of GOT-v3 operator geometry across latent dimensions
k ∈ {3, 4, 5, 6, 7}.

Computes 10 metrics (A–G) from algebra_report.json files produced by
latent_operator_algebra.py, generates a markdown/JSON comparison table,
and produces 5 comparison figures.

Usage
─────
  python3 compare_operator_geometry.py \\
    --inputs \\
      results/operator_algebra_k3/algebra_report.json \\
      results/operator_algebra_k4/algebra_report.json \\
      results/operator_algebra_k5/algebra_report.json \\
      results/operator_algebra_k6/algebra_report.json \\
      results/operator_algebra_k7/algebra_report.json \\
    --ckpts \\
      ckpt/got_v3_ae_pretrain_k3.pt \\
      ckpt/got_v3_ae_pretrain_k4.pt \\
      ckpt/got_v3_ae_pretrain_k5.pt \\
      ckpt/got_v3_ae_pretrain_k6.pt \\
      ckpt/got_v3_ae_pretrain_k7.pt \\
    --out results/operator_geometry_k_ablation/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import seaborn as sns

# ─── operator index table ────────────────────────────────────────────────────
OPS = ["identity","shift","backshift","scale_up","scale_down",
       "apery_pos","apery_neg","sign_flip_b","sign_flip_a","random_sm"]
LABS = ["I","shift","bshift","s↑","s↓","a+","a−","sf_b","sf_a","rand"]
OP_IDX = {op: i for i, op in enumerate(OPS)}

SMOOTH_OPS = {"scale_up","scale_down","apery_pos","apery_neg","sign_flip_a","sign_flip_b"}
SCALE_APERY = {"scale_up","scale_down","apery_pos","apery_neg"}

STYLE = {"font.family":"serif","font.size":10,"axes.labelsize":11,
         "axes.titlesize":12,"figure.dpi":150}
plt.rcParams.update(STYLE)
sns.set_style("whitegrid")
BLUE="#2563EB"; RED="#DC2626"; GREEN="#16A34A"; AMB="#D97706"; GREY="#6B7280"
K_COLORS = {3:"#1D4ED8",4:"#7C3AED",5:"#16A34A",6:"#D97706",7:"#DC2626"}


# ═══════════════════════════════════════════════════════════════════════════
# METRIC EXTRACTORS
# ═══════════════════════════════════════════════════════════════════════════

def metric_A_cosine_stability(report: dict, ref_report: Optional[dict]) -> dict:
    """A. Cosine stability vs reference k (default: k=6)."""
    cm = np.array(report["cos_matrix"])
    n = cm.shape[0]
    # specific canonical pairs
    i_ap = OP_IDX["apery_pos"]; i_an = OP_IDX["apery_neg"]
    i_su = OP_IDX["scale_up"];  i_sd = OP_IDX["scale_down"]
    i_rd = OP_IDX["random_sm"]

    apery_pair_cos  = float(cm[i_ap, i_an])
    scale_pair_cos  = float(cm[i_su, i_sd])
    random_mean_cos = float(np.abs(cm[i_rd, :]).mean())

    # stability vs reference
    if ref_report is not None:
        ref_cm = np.array(ref_report["cos_matrix"])
        stability_vs_ref = float(np.abs(cm - ref_cm).mean())
    else:
        stability_vs_ref = 0.0

    return {
        "apery_pair_cos" : apery_pair_cos,
        "scale_pair_cos" : scale_pair_cos,
        "random_mean_cos": random_mean_cos,
        "stability_vs_ref": stability_vs_ref,
    }


def metric_B_commutator(report: dict) -> dict:
    """B. Commutator norms."""
    cn = np.array(report["comm_norm"])
    n = cn.shape[0]
    # off-diagonal mean (exclude diagonal)
    off_diag = cn[~np.eye(n, dtype=bool)]
    mean_offdiag = float(off_diag.mean())
    max_comm     = float(cn.max())

    i_sh  = OP_IDX["shift"]
    i_sfb = OP_IDX["sign_flip_b"]
    i_bsh = OP_IDX["backshift"]
    comm_shift_sfb    = float(cn[i_sh,  i_sfb])
    comm_shift_bshift = float(cn[i_sh,  i_bsh])

    return {
        "mean_offdiag"        : mean_offdiag,
        "max_comm"            : max_comm,
        "comm_shift_signflipb": comm_shift_sfb,
        "comm_shift_backshift": comm_shift_bshift,
    }


def metric_C_manifold_closure(report: dict) -> dict:
    """C. Manifold closure ratios."""
    dm = {e["op"]: e for e in report.get("dist_manifold", [])}

    smooth_ratios = [dm[op]["ratio"] for op in SMOOTH_OPS if op in dm]
    mean_smooth   = float(np.mean(smooth_ratios)) if smooth_ratios else float("nan")
    ratio_shift   = float(dm["shift"]["ratio"])   if "shift"   in dm else float("nan")
    ratio_bshift  = float(dm["backshift"]["ratio"]) if "backshift" in dm else float("nan")
    n_on_manifold = sum(1 for op in SMOOTH_OPS if op in dm and dm[op]["ratio"] < 3.0)

    return {
        "mean_smooth_closure": mean_smooth,
        "closure_ratio_shift" : ratio_shift,
        "closure_ratio_bshift": ratio_bshift,
        "n_smooth_on_manifold": n_on_manifold,
    }


def metric_D_lie_closure(report: dict) -> dict:
    """D. Lie-like bracket closure."""
    lc = report.get("lie_closure", [])
    cos_90 = sum(1 for e in lc if e.get("best_cos", 0) > 0.90)
    cos_95 = sum(1 for e in lc if e.get("best_cos", 0) > 0.95)

    # within scale+apery subset
    sa_brackets = [
        e for e in lc
        if e["t"] in SCALE_APERY and e["s"] in SCALE_APERY
    ]
    sa_best_cos = float(np.mean([e["best_cos"] for e in sa_brackets])) if sa_brackets else float("nan")

    return {
        "n_closed_brackets_90": cos_90,
        "n_closed_brackets_95": cos_95,
        "sa_mean_best_cos"     : sa_best_cos,
    }


def metric_E_shift_disruptiveness(report: dict) -> dict:
    """E. Shift disruptiveness (raw values; z-score composite computed across k later)."""
    dm  = {e["op"]: e for e in report.get("dist_manifold", [])}
    lip = {e["op"]: e for e in report.get("lipschitz", [])}
    cn  = np.array(report["comm_norm"])
    add = report.get("additivity", [])
    eig = {e["op"]: e for e in report.get("eigenoperator", [])}

    closure_ratio_shift   = float(dm["shift"]["ratio"])   if "shift"   in dm  else float("nan")
    max_L_shift           = float(lip["shift"]["max_L"])  if "shift"   in lip else float("nan")
    comm_shift_sfb        = float(cn[OP_IDX["shift"], OP_IDX["sign_flip_b"]])
    eigen_cos_shift       = float(eig["shift"]["cos_vT_z"]) if "shift" in eig else float("nan")

    # additivity error for (shift, shift)
    add_err_ss = next(
        (e["additivity_err"]["mean"] for e in add if e["t"]=="shift" and e["s"]=="shift"),
        float("nan")
    )

    return {
        "closure_ratio_shift" : closure_ratio_shift,
        "max_L_shift"         : max_L_shift,
        "comm_shift_sfb"      : comm_shift_sfb,
        "eigen_cos_shift"     : eigen_cos_shift,
        "additivity_err_shift": add_err_ss,
    }


def metric_F_zeta3_alignment(report: dict) -> dict:
    """F. ζ(3) scalar-field alignment."""
    al = {e["op"]: e["align_core_far"] for e in report.get("align", [])}
    align_sd  = float(al.get("scale_down", float("nan")))
    align_ap  = float(al.get("apery_pos",  float("nan")))

    valid = {op: v for op, v in al.items() if not np.isnan(v)}
    max_op  = max(valid, key=valid.get) if valid else "?"
    min_op  = min(valid, key=valid.get) if valid else "?"

    return {
        "align_scale_down"   : align_sd,
        "align_apery_pos"    : align_ap,
        "max_pos_align_op"   : max_op,
        "max_pos_align_val"  : float(valid.get(max_op, float("nan"))),
        "min_neg_align_op"   : min_op,
        "min_neg_align_val"  : float(valid.get(min_op, float("nan"))),
    }


def metric_G_training(ckpt_path: Optional[str]) -> dict:
    """G. Training metrics from checkpoint history file."""
    default = {"recon": float("nan"), "nbr": float("nan"),
               "mix_acc": float("nan"), "op_loss": float("nan")}
    if ckpt_path is None or not Path(ckpt_path).exists():
        return default
    hist_path = Path(ckpt_path).with_suffix(".history.json")
    if not hist_path.exists():
        # try reading best val from ckpt
        try:
            import torch
            state = torch.load(ckpt_path, map_location="cpu")
            v = state.get("val_logs", {})
            return {
                "recon"   : float(v.get("recon",   float("nan"))),
                "nbr"     : float(v.get("nbr",     float("nan"))),
                "mix_acc" : float(v.get("component_acc", float("nan"))),
                "op_loss" : float(v.get("op",      float("nan"))),
            }
        except Exception:
            return default
    try:
        hist = json.loads(hist_path.read_text())
        # find epoch with best val total
        best = min(hist, key=lambda e: e["val"]["total"])
        v = best["val"]
        return {
            "recon"   : float(v.get("recon",   float("nan"))),
            "nbr"     : float(v.get("nbr",     float("nan"))),
            "mix_acc" : float(v.get("component_acc", float("nan"))),
            "op_loss" : float(v.get("op",      float("nan"))),
        }
    except Exception:
        return default


# ═══════════════════════════════════════════════════════════════════════════
# COMPOSITE SHIFT SCORE  (z-score based, computed across all k)
# ═══════════════════════════════════════════════════════════════════════════

def compute_shift_disruptiveness_composite(all_E: List[dict]) -> List[float]:
    keys = ["closure_ratio_shift","max_L_shift","comm_shift_sfb","additivity_err_shift"]
    mat = np.array([[e[k] for k in keys] for e in all_E], dtype=np.float32)
    # replace nan with column mean
    col_means = np.nanmean(mat, axis=0)
    for j in range(mat.shape[1]):
        nan_mask = np.isnan(mat[:, j])
        mat[nan_mask, j] = col_means[j]
    col_std = mat.std(0) + 1e-9
    z = (mat - mat.mean(0)) / col_std
    return list(z.sum(1))


# ═══════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════

def _save(name: str, out_dir: Path):
    for ext in ("pdf","png"):
        plt.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  fig: {name}")


def fig_metric_vs_k(ks: List[int], rows: dict, out_dir: Path):
    """Line plot of 6 key scalar metrics vs k."""
    metrics = [
        ("apery_pair_cos",       "cos(apery+, apery−)",       "A",  False),
        ("mean_smooth_closure",  "mean smooth closure ratio",  "C",  False),
        ("n_closed_brackets_90", "# Lie brackets cos>0.90",    "D",  True),
        ("max_L_shift",          "max Lipschitz (shift)",      "E",  False),
        ("align_scale_down",     "ζ(3)-alignment scale_down",  "F",  False),
        ("mix_acc",              "mix_acc (train)",             "G",  True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, (key, label, section, higher_is_better) in zip(axes.flat, metrics):
        vals = [rows[k].get(key, float("nan")) for k in ks]
        ax.plot(ks, vals, "o-", color=BLUE, lw=2, markersize=8)
        for ki, vi in zip(ks, vals):
            if not np.isnan(vi):
                ax.annotate(f"{vi:.2f}", (ki, vi),
                            textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_xticks(ks)
        ax.set_xlabel("k (latent dim)")
        ax.set_title(f"[{section}] {label}", fontsize=9)
        arrow = "↑ better" if higher_is_better else "↓ better"
        ax.set_ylabel(arrow, fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Operator Geometry Metrics vs Latent Dimension k", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save("figC1_metrics_vs_k", out_dir)


def fig_cosine_stability(ks: List[int], rows: dict, out_dir: Path):
    """Bar chart: cosine stability score and specific pairs per k."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # apery pair
    vals = [rows[k].get("apery_pair_cos", float("nan")) for k in ks]
    colors = [K_COLORS.get(ki, GREY) for ki in ks]
    axes[0].bar(ks, vals, color=colors, width=0.6, edgecolor="white")
    axes[0].axhline(-0.998, color=RED, lw=1.5, linestyle="--", label="target k=3")
    axes[0].set_title("[A] cos(apery+, apery−)\n(target ≈ −0.998)")
    axes[0].set_xticks(ks); axes[0].set_xlabel("k"); axes[0].legend(fontsize=8)

    # scale pair
    vals2 = [rows[k].get("scale_pair_cos", float("nan")) for k in ks]
    axes[1].bar(ks, vals2, color=colors, width=0.6, edgecolor="white")
    axes[1].axhline(-0.8, color=RED, lw=1.5, linestyle="--", label="target k=3")
    axes[1].set_title("[A] cos(scale_up, scale_down)\n(should be negative)")
    axes[1].set_xticks(ks); axes[1].set_xlabel("k"); axes[1].legend(fontsize=8)

    # stability vs ref
    vals3 = [rows[k].get("stability_vs_ref", float("nan")) for k in ks]
    axes[2].bar(ks, vals3, color=colors, width=0.6, edgecolor="white")
    axes[2].set_title("[A] Cosine matrix instability\nvs k=6 reference (lower=better)")
    axes[2].set_xticks(ks); axes[2].set_xlabel("k")

    fig.suptitle("Cosine Structure Stability Across k", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save("figC2_cosine_stability", out_dir)


def fig_commutator_shift(ks: List[int], rows: dict, out_dir: Path):
    """Bar chart of commutator structure per k."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = [K_COLORS.get(ki, GREY) for ki in ks]

    for ax, (key, title) in zip(axes, [
        ("mean_offdiag",         "Mean off-diag ||[T,S]||\n(lower = more commutative)"),
        ("comm_shift_signflipb", "||[shift, sign_flip_b]||\n(should stay large)"),
        ("max_comm",             "Max ||[T,S]||\n"),
    ]):
        vals = [rows[k].get(key, float("nan")) for k in ks]
        axes[list(["mean_offdiag","comm_shift_signflipb","max_comm"]).index(key)].bar(
            ks, vals, color=colors, width=0.6, edgecolor="white")
        axes[list(["mean_offdiag","comm_shift_signflipb","max_comm"]).index(key)].set_title(f"[B] {title}")
        axes[list(["mean_offdiag","comm_shift_signflipb","max_comm"]).index(key)].set_xticks(ks)
        axes[list(["mean_offdiag","comm_shift_signflipb","max_comm"]).index(key)].set_xlabel("k")

    fig.suptitle("Commutator Structure Across k", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save("figC3_commutator_structure", out_dir)


def fig_shift_disruptiveness(ks: List[int], rows: dict, composite: List[float], out_dir: Path):
    """Multi-panel shift disruptiveness figure."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    colors = [K_COLORS.get(ki, GREY) for ki in ks]

    panels = [
        ("closure_ratio_shift", "Manifold closure ratio (shift)\nhigher = more disruptive"),
        ("max_L_shift",         "Max Lipschitz L (shift)\nhigher = more singular"),
        ("comm_shift_sfb",      "||[shift, sign_flip_b]||\nhigher = more non-commutative"),
        ("eigen_cos_shift",     "Eigenoperator cos(v_shift, z)\n|cos| > 0.7 = eigenlike"),
    ]
    for ax, (key, title) in zip(axes.flat, panels):
        vals = [rows[k].get(key, float("nan")) for k in ks]
        ax.bar(ks, vals, color=colors, width=0.6, edgecolor="white")
        ax.set_title(f"[E] {title}", fontsize=9)
        ax.set_xticks(ks); ax.set_xlabel("k")
        for ki, vi in zip(ks, vals):
            if not np.isnan(vi):
                ax.text(ki, vi * 1.01, f"{vi:.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Shift Operator Disruptiveness Across k", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save("figC4_shift_disruptiveness", out_dir)


def fig_heatmap_all_metrics(ks: List[int], rows: dict, out_dir: Path):
    """Heatmap: metric × k, normalized 0–1 within each metric."""
    metric_keys = [
        ("apery_pair_cos",       "cos(a+,a−)",     False),  # lower_is_better for abs distance from -1
        ("scale_pair_cos",       "cos(s↑,s↓)",     False),
        ("stability_vs_ref",     "A: instability", False),   # lower better
        ("mean_offdiag",         "B: mean comm",   False),   # lower better
        ("comm_shift_signflipb", "B: shift×sfb",   True),    # higher = consistently disruptive
        ("mean_smooth_closure",  "C: smooth close",False),   # lower better
        ("closure_ratio_shift",  "C: shift close", True),    # higher = shift disruptive
        ("n_closed_brackets_95", "D: Lie close@95",True),    # higher better
        ("sa_mean_best_cos",     "D: SA bracket",  True),    # higher better
        ("max_L_shift",          "E: max L shift", True),    # higher = disruptive (fine)
        ("align_scale_down",     "F: ζ align sd",  True),    # higher better
        ("mix_acc",              "G: mix_acc",      True),   # higher better
    ]
    mat = np.array([
        [rows[k].get(mkey, float("nan")) for k in ks]
        for mkey, _, _ in metric_keys
    ], dtype=np.float32)

    # normalize each row (metric) to [0,1]; flip lower-is-better
    mat_norm = np.zeros_like(mat)
    for i, (_, _, higher_better) in enumerate(metric_keys):
        row = mat[i]
        valid = row[~np.isnan(row)]
        if len(valid) < 2:
            mat_norm[i] = 0.5
            continue
        rng = valid.max() - valid.min() + 1e-9
        n = (row - valid.min()) / rng
        mat_norm[i] = n if higher_better else (1.0 - n)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat_norm, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"k={ki}" for ki in ks], fontsize=10)
    ax.set_yticks(range(len(metric_keys)))
    ax.set_yticklabels([lbl for _, lbl, _ in metric_keys], fontsize=9)
    ax.set_title("Operator Geometry Quality Heatmap\n(green = best within metric, red = worst)",
                 fontsize=11, fontweight="bold")

    # annotate with raw values
    for i in range(len(metric_keys)):
        for j, ki in enumerate(ks):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5, color="black")

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="normalised score (green=best)")
    plt.tight_layout()
    _save("figC5_heatmap_all_metrics", out_dir)


# ═══════════════════════════════════════════════════════════════════════════
# VERDICT TABLE
# ═══════════════════════════════════════════════════════════════════════════

def print_verdict(ks: List[int], rows: dict, composite: List[float]):
    W = 100
    print("\n" + "═" * W)
    print("  OPERATOR GEOMETRY COMPARISON — k-ABLATION VERDICT")
    print("═" * W)
    print(f"  {'Metric':<38} " + " ".join(f"k={ki:>4}" for ki in ks) + "  Best")
    print("─" * W)

    def fv(v):
        if isinstance(v, float) and np.isnan(v): return " --- "
        if isinstance(v, float): return f"{v:>6.3f}"
        return f"{str(v):>6}"

    sections = [
        ("A — Cosine Stability", [
            ("apery_pair_cos",    "cos(apery+, apery−)     [→ -0.998]"),
            ("scale_pair_cos",    "cos(scale+, scale−)     [→ -0.80]"),
            ("stability_vs_ref",  "instability vs k=6  ↓"),
        ]),
        ("B — Commutator Norms", [
            ("mean_offdiag",         "mean off-diag ||[T,S]||  ↓"),
            ("comm_shift_signflipb", "||[shift, sf_b]||         ↑ disruptive"),
            ("max_comm",             "max ||[T,S]||"),
        ]),
        ("C — Manifold Closure", [
            ("mean_smooth_closure",  "mean smooth ratio        ↓"),
            ("closure_ratio_shift",  "shift ratio             ↑ disruptive"),
            ("n_smooth_on_manifold", "n_smooth on manifold    ↑"),
        ]),
        ("D — Lie-like Closure", [
            ("n_closed_brackets_90", "# brackets cos>0.90    ↑"),
            ("n_closed_brackets_95", "# brackets cos>0.95    ↑"),
            ("sa_mean_best_cos",     "scale+apery mean cos    ↑"),
        ]),
        ("E — Shift Disruptiveness", [
            ("max_L_shift",          "max Lipschitz (shift)"),
            ("eigen_cos_shift",      "eigenoperator cos"),
            ("additivity_err_shift", "additivity error (shift,shift)"),
        ]),
        ("F — ζ(3) Alignment", [
            ("align_scale_down", "alignment scale_down    ↑"),
            ("align_apery_pos",  "alignment apery_pos"),
            ("max_pos_align_val","max positive alignment  ↑"),
        ]),
        ("G — Training Metrics", [
            ("recon",   "reconstruction loss     ↓"),
            ("nbr",     "neighbourhood loss      ↓"),
            ("mix_acc", "component accuracy      ↑"),
        ]),
    ]

    for sec_name, fields in sections:
        print(f"\n  {sec_name}")
        for key, label in fields:
            vals = [rows[ki].get(key, float("nan")) for ki in ks]
            str_vals = [fv(v) for v in vals]
            # find best (heuristic: label contains ↑ → max, ↓ → min)
            if "↑" in label:
                valid = [(v, ki) for v, ki in zip(vals, ks) if not np.isnan(v)]
                best_k = max(valid, key=lambda x: x[0])[1] if valid else "?"
            elif "↓" in label:
                valid = [(v, ki) for v, ki in zip(vals, ks) if not np.isnan(v)]
                best_k = min(valid, key=lambda x: x[0])[1] if valid else "?"
            else:
                best_k = "-"
            print(f"    {label:<40} " + " ".join(str_vals) + f"  k={best_k}")

    # Composite shift disruptiveness
    print(f"\n  E — Composite Shift Score (z-sum, higher = consistently disruptive)")
    print(f"    {'composite shift disruptiveness':<40} " +
          " ".join(f"{v:>6.2f}" for v in composite))

    print("\n" + "─" * W)
    print("  RECOMMENDATIONS")
    print("─" * W)
    print("  [Minimal geometry]   k with best mix_acc / lowest recon, prefer small k")
    print("  [Operator algebra]   k with best Lie closure + stable apery cosine + smooth manifold")
    print("  [Navigation]         k with low smooth closure + high Lie closure")
    print("  [Best compromise]    trade-off: Lie@95 + smooth closure + mix_acc")
    print("═" * W)


def build_markdown_table(ks: List[int], rows: dict, composite: List[float]) -> str:
    lines = ["# Operator Geometry k-Ablation Comparison\n"]
    lines.append("| Metric | " + " | ".join(f"k={ki}" for ki in ks) + " | Best |")
    lines.append("|" + "---|" * (len(ks) + 2))

    def fv(v):
        if isinstance(v, float) and np.isnan(v): return "—"
        if isinstance(v, float): return f"{v:.3f}"
        return str(v)

    rows_data = [
        ("A: cos(apery+, apery−)",      "apery_pair_cos",       "max"),
        ("A: cos(scale+, scale−)",      "scale_pair_cos",       "max"),
        ("A: instability vs k=6",       "stability_vs_ref",     "min"),
        ("B: mean off-diag [T,S]",      "mean_offdiag",         "min"),
        ("B: [shift, sf_b]",            "comm_shift_signflipb", "-"),
        ("C: mean smooth closure ratio","mean_smooth_closure",  "min"),
        ("C: shift closure ratio",      "closure_ratio_shift",  "-"),
        ("C: n smooth on manifold",     "n_smooth_on_manifold", "max"),
        ("D: # brackets cos>0.90",      "n_closed_brackets_90", "max"),
        ("D: # brackets cos>0.95",      "n_closed_brackets_95", "max"),
        ("D: SA subset mean cos",       "sa_mean_best_cos",     "max"),
        ("E: max L (shift)",            "max_L_shift",          "-"),
        ("E: eigen cos (shift)",        "eigen_cos_shift",      "-"),
        ("F: ζ(3) align scale_down",    "align_scale_down",     "max"),
        ("F: ζ(3) align apery_pos",     "align_apery_pos",      "-"),
        ("G: recon loss",               "recon",                "min"),
        ("G: nbr loss",                 "nbr",                  "min"),
        ("G: mix accuracy",             "mix_acc",              "max"),
    ]

    for label, key, best_dir in rows_data:
        vals = [rows[ki].get(key, float("nan")) for ki in ks]
        str_vals = [fv(v) for v in vals]
        if best_dir == "max":
            valid = [(v, ki) for v, ki in zip(vals, ks) if not np.isnan(v)]
            best = f"k={max(valid,key=lambda x:x[0])[1]}" if valid else "—"
        elif best_dir == "min":
            valid = [(v, ki) for v, ki in zip(vals, ks) if not np.isnan(v)]
            best = f"k={min(valid,key=lambda x:x[0])[1]}" if valid else "—"
        else:
            best = "—"
        lines.append(f"| {label} | " + " | ".join(str_vals) + f" | {best} |")

    # composite shift row
    lines.append(
        "| E: composite shift z-score | " +
        " | ".join(f"{v:.2f}" for v in composite) +
        " | — |"
    )
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Compare operator geometry across k")
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="algebra_report.json files, one per k")
    ap.add_argument("--ks",    nargs="+", type=int, default=None,
                    help="Explicit k values (auto-detected from ckpt names if omitted)")
    ap.add_argument("--ckpts", nargs="*", default=None,
                    help="Optional checkpoint paths for training metrics (same order as --inputs)")
    ap.add_argument("--ref_k", type=int, default=6,
                    help="Reference k for cosine stability comparison (default: 6)")
    ap.add_argument("--out", default="results/operator_geometry_k_ablation/")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # auto-detect k from filenames e.g. operator_algebra_k5/algebra_report.json
    if args.ks:
        ks = args.ks
    else:
        ks = []
        for path in args.inputs:
            p = Path(path)
            # look for _k{N}/ in path
            for part in p.parts:
                if "_k" in part:
                    try:
                        ks.append(int(part.split("_k")[-1].rstrip("/")))
                        break
                    except ValueError:
                        pass
            else:
                ks.append(len(ks) + 3)  # fallback

    ckpts = args.ckpts or [None] * len(args.inputs)

    reports = []
    for path in args.inputs:
        print(f"  loading {path}")
        reports.append(json.loads(Path(path).read_text()))

    # find reference report
    ref_report = None
    if args.ref_k in ks:
        ref_report = reports[ks.index(args.ref_k)]

    # ── compute all metrics ─────────────────────────────────────────────
    print("Computing metrics …")
    rows: Dict[int, dict] = {}
    for ki, rep, ckpt in zip(ks, reports, ckpts):
        r: dict = {}
        r.update(metric_A_cosine_stability(rep, ref_report))
        r.update(metric_B_commutator(rep))
        r.update(metric_C_manifold_closure(rep))
        r.update(metric_D_lie_closure(rep))
        r.update(metric_E_shift_disruptiveness(rep))
        r.update(metric_F_zeta3_alignment(rep))
        r.update(metric_G_training(ckpt))
        rows[ki] = r

    all_E = [metric_E_shift_disruptiveness(rep) for rep in reports]
    composite = compute_shift_disruptiveness_composite(all_E)

    # ── figures ─────────────────────────────────────────────────────────
    print("Generating figures …")
    fig_metric_vs_k(ks, rows, out_dir)
    fig_cosine_stability(ks, rows, out_dir)
    fig_commutator_shift(ks, rows, out_dir)
    fig_shift_disruptiveness(ks, rows, composite, out_dir)
    fig_heatmap_all_metrics(ks, rows, out_dir)

    # ── text report ──────────────────────────────────────────────────────
    print_verdict(ks, rows, composite)

    # ── save outputs ─────────────────────────────────────────────────────
    def _nan_safe(obj):
        if isinstance(obj, float) and np.isnan(obj): return None
        if isinstance(obj, dict): return {k: _nan_safe(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_nan_safe(x) for x in obj]
        return obj

    with open(out_dir / "comparison_metrics.json", "w") as f:
        json.dump(_nan_safe({"ks": ks, "rows": {str(k): rows[k] for k in ks},
                             "composite_shift": composite}), f, indent=2)
    print(f"  JSON → {out_dir}/comparison_metrics.json")

    md = build_markdown_table(ks, rows, composite)
    with open(out_dir / "comparison_table.md", "w") as f:
        f.write(md)
    print(f"  MD  → {out_dir}/comparison_table.md")

    print(f"\nDone. All outputs in {out_dir}/")


if __name__ == "__main__":
    main()
