"""
Generate all figures for the preprint paper.
Run from the repo root:  python3 paper/generate_figures.py
Outputs:  paper/figures/fig_*.pdf  (and .png for quick preview)
"""
import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm, LogNorm
import seaborn as sns

# ── paths ────────────────────────────────────────────────────────────────────
HERE   = os.path.dirname(__file__)
ROOT   = os.path.dirname(HERE)
REPORT = os.path.join(ROOT, "results", "operator_algebra", "algebra_report.json")
OUTDIR = os.path.join(HERE, "figures")
os.makedirs(OUTDIR, exist_ok=True)

with open(REPORT) as f:
    d = json.load(f)

OPS  = ["identity","shift","backshift","scale_up","scale_down",
        "apery_pos","apery_neg","sign_flip_b","sign_flip_a","random_sm"]
LABS = ["I","shift","bshift","s↑","s↓","a+","a−","sf_b","sf_a","rand"]

STYLE = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.dpi": 150,
}
plt.rcParams.update(STYLE)
sns.set_style("whitegrid")

ACCENT = "#2563EB"   # blue
WARM   = "#DC2626"   # red
GREY   = "#6B7280"

def save(name):
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(OUTDIR, f"{name}.{ext}"),
                    bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved {name}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Cosine similarity heatmap
# ─────────────────────────────────────────────────────────────────────────────
cos = np.array(d["cos_matrix"])

fig, ax = plt.subplots(figsize=(7.5, 6.2))
norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
im = ax.imshow(cos, cmap="RdBu_r", norm=norm, aspect="auto")

ax.set_xticks(range(len(OPS))); ax.set_xticklabels(LABS, rotation=45, ha="right")
ax.set_yticks(range(len(OPS))); ax.set_yticklabels(LABS)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("cosine similarity")

for i in range(len(OPS)):
    for j in range(len(OPS)):
        v = cos[i, j]
        if abs(v) > 0.001:
            color = "white" if abs(v) > 0.6 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=7.5, color=color, fontweight="bold" if abs(v) > 0.9 else "normal")

ax.set_title("Fig. 1 — Operator Cosine Similarity Matrix $\\cos(v_T, v_S)$")
ax.set_xlabel("Operator $S$")
ax.set_ylabel("Operator $T$")

# annotate key pairs
for (i, j, note) in [(5, 6, "−0.998"), (2, 6, ""), (3, 4, "−0.799")]:
    rect = mpatches.FancyBboxPatch((j-0.47, i-0.47), 0.94, 0.94,
                                    linewidth=2, edgecolor=WARM,
                                    boxstyle="round,pad=0.05", fill=False)
    ax.add_patch(rect)

plt.tight_layout()
save("fig1_cosine_heatmap")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Commutator norm heatmap (log scale)
# ─────────────────────────────────────────────────────────────────────────────
cn_raw = np.array(d["comm_norm"])
cn = cn_raw + 1e-4   # offset for log

fig, ax = plt.subplots(figsize=(7.5, 6.2))
im = ax.imshow(cn, cmap="YlOrRd", norm=LogNorm(vmin=1e-4, vmax=cn.max()), aspect="auto")

ax.set_xticks(range(len(OPS))); ax.set_xticklabels(LABS, rotation=45, ha="right")
ax.set_yticks(range(len(OPS))); ax.set_yticklabels(LABS)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("$\\|[T,S]\\|$ (log scale)")

# label cells > 0.01
for i in range(len(OPS)):
    for j in range(len(OPS)):
        v = cn_raw[i, j]
        if v > 0.01:
            color = "white" if v > 1.0 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7.5, color=color)

ax.set_title("Fig. 2 — Commutator Norm Matrix $\\|[T,S]\\|$ (log scale)")
ax.set_xlabel("Operator $S$")
ax.set_ylabel("Operator $T$")
plt.tight_layout()
save("fig2_commutator_heatmap")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Manifold closure: distance ratio per operator (horizontal bars)
# ─────────────────────────────────────────────────────────────────────────────
closure_ops = ["identity","scale_up","scale_down","apery_pos","apery_neg",
               "sign_flip_b","sign_flip_a","random_sm","backshift","shift"]
ratios = []
means  = []
base   = d["dist_manifold"][0]["baseline_dist"]

# build lookup
dm_map = {e["op"]: e for e in d["dist_manifold"]}
dm_map["identity"] = {"mean_dist": base, "ratio": 1.0}

for op in closure_ops:
    e = dm_map[op]
    ratios.append(e["ratio"])
    means.append(e["mean_dist"])

colors = [WARM if r > 10 else ACCENT if r > 2 else GREY for r in ratios]
labels = [op.replace("_", "\\_") for op in closure_ops]

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.barh(range(len(closure_ops)), ratios, color=colors, edgecolor="white", height=0.6)

ax.set_xscale("log")
ax.set_yticks(range(len(closure_ops)))
ax.set_yticklabels([o.replace("_", " ") for o in closure_ops], fontsize=9.5)
ax.axvline(1.0, color="black", linewidth=1.5, linestyle="--", alpha=0.6, label="baseline")
ax.axvline(10.0, color=WARM, linewidth=1.0, linestyle=":", alpha=0.4)
ax.set_xlabel("Distance ratio relative to identity (log scale)")
ax.set_title("Fig. 3 — Manifold Closure: Distance after $T(x)$ / Baseline")

for i, (bar, r) in enumerate(zip(bars, ratios)):
    ax.text(max(r * 1.05, 1.3), i, f"{r:.0f}×", va="center", fontsize=8.5,
            color=WARM if r > 10 else ACCENT)

# legend patches
ax.legend(handles=[
    mpatches.Patch(color=GREY,  label="on manifold  (ratio < 2)"),
    mpatches.Patch(color=ACCENT, label="slightly off  (ratio 2–10)"),
    mpatches.Patch(color=WARM,  label="off manifold  (ratio > 10)"),
], loc="lower right", fontsize=9)

plt.tight_layout()
save("fig3_manifold_closure")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — Lipschitz constants: mean ± std + max marker
# ─────────────────────────────────────────────────────────────────────────────
lip_ops   = [e["op"] for e in d["lipschitz"]]
lip_mean  = [e["mean_L"] for e in d["lipschitz"]]
lip_med   = [e["median_L"] for e in d["lipschitz"]]
lip_max   = [e["max_L"] for e in d["lipschitz"]]

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(lip_ops))
bars = ax.bar(x, lip_mean, color=ACCENT, alpha=0.8, label="mean $L$", width=0.5)
ax.plot(x, lip_med, "o", color="white", markeredgecolor=ACCENT,
        markeredgewidth=1.5, markersize=6, label="median $L$", zorder=5)
ax.plot(x, lip_max, "^", color=WARM, markersize=7, label="max $L$", zorder=5)

ax.set_xticks(x)
ax.set_xticklabels([o.replace("_", " ") for o in lip_ops], rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Lipschitz constant $L$")
ax.set_title("Fig. 4 — Lipschitz Constants per Operator\n"
             "(mean, median, max)")
ax.legend(fontsize=9)

# annotate shift max
shift_idx = lip_ops.index("shift")
ax.annotate(f"max = {lip_max[shift_idx]:.1f}",
            xy=(shift_idx, lip_max[shift_idx]),
            xytext=(shift_idx + 0.3, lip_max[shift_idx] * 0.95),
            fontsize=8.5, color=WARM,
            arrowprops=dict(arrowstyle="->", color=WARM, lw=1.2))

plt.tight_layout()
save("fig4_lipschitz")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — Conservation laws grouped bar chart
# ─────────────────────────────────────────────────────────────────────────────
cons = d["conservation"]
props = ["deg_a", "deg_b", "sign_a0", "norm_class"]
prop_labels = ["deg($a$)", "deg($b$)", "sign($a_0$)", "norm class"]

ops_c = list(cons.keys())
n_ops = len(ops_c)
n_props = len(props)
x = np.arange(n_ops)
width = 0.18
offsets = np.linspace(-(n_props-1)/2, (n_props-1)/2, n_props) * width

palette = ["#1D4ED8","#059669","#D97706","#7C3AED"]

fig, ax = plt.subplots(figsize=(10, 4.5))
for k, (prop, lbl, color) in enumerate(zip(props, prop_labels, palette)):
    vals = [cons[op][prop] * 100 for op in ops_c]
    ax.bar(x + offsets[k], vals, width, label=lbl, color=color, alpha=0.85, edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels([o.replace("_", " ") for o in ops_c], rotation=30, ha="right", fontsize=8.5)
ax.set_ylabel("Conservation rate (%)")
ax.set_ylim(0, 112)
ax.axhline(100, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
ax.set_title("Fig. 5 — Conservation Laws per Operator\n"
             "(fraction of generators preserving structural property)")
ax.legend(ncol=4, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.0))
plt.tight_layout()
save("fig5_conservation")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 — Algebraic law pass/fail summary grid
# ─────────────────────────────────────────────────────────────────────────────
# Build pass/fail for each operator across laws
law_names = ["Identity", "Involution\n$T^2{=}I$", "Idempotent\n$T^2{=}T$",
             "Period-2", "Manifold\nclosure", "Additivity\nin $z$",
             "Jacobi\n(smooth)", "Lie\nclosure", "Braid\nrelation",
             "deg($b$)\nconserved", "sign($a_0$)\nconserved"]

# hand-coded from results (1=pass, 0.5=approx, 0=fail)
grid = {
    "identity":    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "shift":       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0],
    "backshift":   [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0.5],
    "scale_up":    [1, 0, 0.5, 0.5, 1, 1, 1, 1, 0, 1, 1],
    "scale_down":  [1, 0, 0.5, 0.5, 1, 1, 1, 1, 0, 1, 1],
    "apery_pos":   [1, 0, 0.5, 0.5, 1, 1, 1, 1, 0.5, 1, 1],
    "apery_neg":   [1, 0, 0.5, 0.5, 1, 1, 1, 1, 0.5, 1, 1],
    "sign_flip_b": [1, 1, 0.5, 1, 1, 1, 0, 0, 0, 1, 1],
    "sign_flip_a": [1, 1, 0.5, 1, 1, 1, 1, 0, 0.5, 1, 1],
    "random_sm":   [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.5],
}

mat = np.array([grid[op] for op in OPS])

fig, ax = plt.subplots(figsize=(10, 5))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "traffic", ["#EF4444","#F59E0B","#22C55E"], N=256)
im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")

ax.set_xticks(range(len(law_names)))
ax.set_xticklabels(law_names, fontsize=8, rotation=20, ha="right")
ax.set_yticks(range(len(OPS)))
ax.set_yticklabels([o.replace("_", " ") for o in OPS], fontsize=9)
ax.set_title("Fig. 6 — Algebraic Law Pass/Fail Summary\n"
             "(green = pass, amber = approximate, red = fail)")

for i in range(len(OPS)):
    for j in range(len(law_names)):
        v = mat[i, j]
        sym = "+" if v == 1 else "~" if v == 0.5 else "-"
        color = "white" if v < 0.3 else "black" if v > 0.7 else "black"
        ax.text(j, i, sym, ha="center", va="center", fontsize=11, color=color)

cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02,
                    ticks=[0, 0.5, 1])
cbar.ax.set_yticklabels(["fail", "approx", "pass"])
plt.tight_layout()
save("fig6_law_summary")

print(f"\nAll figures saved to {OUTDIR}/")
