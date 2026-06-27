"""
analyse_results.py  —  Figure generation (publication-ready revision)

Design notes:
  - Okabe-Ito palette, publication typography, spine cleanup
  - fig1: grouped horizontal bars, legend below, capsize improved
  - fig2: actual delta bars, star annotation, sorted correctly
  - fig3: line chart with grey background + top-5 highlight (replaces heatmap)
  - fig4: Pareto frontier, manual label placement, legend outside
  - fig5: lollipop CD diagram (new)
  - fig6: horizontal bars with value annotations, full model names
  - figS1: k-formatted x-axis, panel label S1, legend outside

Usage (run from the repository root):
    python scripts/analyse_results.py

Paths default to the repository-relative OUTPUTS/ and FIGURES/ directories and can
be overridden with the OUTPUTS_DIR / FIGURES_DIR environment variables.
"""

import os
import sys
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
# Resolve the repository root (this file lives in scripts/figures/), overridable via env.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUTS = os.environ.get("OUTPUTS_DIR", os.path.join(_REPO_ROOT, "OUTPUTS"))
FIGURES = os.environ.get("FIGURES_DIR", os.path.join(_REPO_ROOT, "FIGURES"))
os.makedirs(FIGURES, exist_ok=True)

# ── matplotlib backend (must be set before importing pyplot) ─────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Shared style from plot_style.py (lives in scripts/, the parent of figures/) ──
_script_dir = os.path.dirname(os.path.abspath(__file__))
_scripts_root = os.path.dirname(_script_dir)
for _p in (_script_dir, _scripts_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from plot_style import PALETTE, MARKERS, LINESTYLES, apply_style, style_ax

apply_style()   # apply global rcParams defined in plot_style.py

# ── Model name cleanup ────────────────────────────────────────────────────────
PREFIXES = [
    "sentence-transformers/", "BAAI/", "intfloat/", "juanpablomesa/",
    "NeuML/", "menadsa/", "pritamdeka/", "dmis-lab/", "sultan/",
    "microsoft/", "emilyalsentzer/", "medicalai/", "allenai/",
    "cambridgeltl/", "BioMistral/",
]

def short(name: str) -> str:
    """Strip HuggingFace org prefix; keep full model name (no truncation)."""
    for pfx in PREFIXES:
        name = name.replace(pfx, "")
    return name

def save(fig, name):
    """Save as PDF and PNG at 300 DPI into FIGURES directory."""
    for ext in ["pdf", "png"]:
        fig.savefig(f"{FIGURES}/{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURES}/{name}.pdf / .png")

# ── Data loading ──────────────────────────────────────────────────────────────
def load_exp1():
    """Load summary results for all datasets and merge BM25."""
    files = glob.glob(f"{OUTPUTS}/exp1_ontology_results_*.csv")
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    bm25_rows = []
    for dataset, raw_file in [
        ("mimic_setA", f"{OUTPUTS}/bm25_raw_mimic_setA.csv"),
        ("mimic_setB", f"{OUTPUTS}/bm25_raw_mimic_setB.csv"),
    ]:
        if not os.path.exists(raw_file):
            continue
        raw = pd.read_csv(raw_file)
        for m in raw["model"].unique():
            sub = raw[raw["model"] == m]
            row = {
                "model": m, "category": "baseline", "dataset": dataset,
                "n_queries": len(sub),
                "hits@1_mean":  sub["hits@1"].mean(),  "hits@1_std":  sub["hits@1"].std(),
                "hits@5_mean":  sub["hits@5"].mean(),  "hits@5_std":  sub["hits@5"].std(),
                "hits@10_mean": sub["hits@10"].mean(), "hits@10_std": sub["hits@10"].std(),
                "mrr_mean":     sub["mrr"].mean(),     "mrr_std":     sub["mrr"].std(),
                "ndcg@10_mean": sub["ndcg@10"].mean(), "ndcg@10_std": sub["ndcg@10"].std(),
                "time_ms_mean": sub["time_ms"].mean(), "time_ms_std": sub["time_ms"].std(),
            }
            bm25_rows.append(row)
    if bm25_rows:
        df = pd.concat([df, pd.DataFrame(bm25_rows)], ignore_index=True)
    return df


def load_exp1_raw():
    """Load per-query raw results and merge BM25 raw."""
    dfs = []
    for f in glob.glob(f"{OUTPUTS}/exp1_raw_*.csv"):
        d = pd.read_csv(f)
        if "dataset" not in d.columns:
            ds = os.path.basename(f).replace("exp1_raw_", "").replace(".csv", "")
            d["dataset"] = ds
        dfs.append(d)
    for f in glob.glob(f"{OUTPUTS}/bm25_raw_*.csv"):
        d = pd.read_csv(f)
        if "dataset" not in d.columns:
            ds = os.path.basename(f).replace("bm25_raw_", "").replace(".csv", "")
            d["dataset"] = ds
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True) if dfs else None


# ────────────────────────────────────────────────────────────────────────────
# Fig 1 — Grouped horizontal bar chart: Set A + Set B (MIMIC only)
# ────────────────────────────────────────────────────────────────────────────
def fig_embedding_comparison(df):
    """
    Fig 1: Grouped horizontal bars, two bars per model (Set A darker, Set B lighter).
    Sorted by Set B performance descending (best at top).
    """
    sub = df[df["dataset"].isin(["mimic_setA", "mimic_setB"])].copy()
    sub["label"] = sub["model"].apply(short)

    pivot     = sub.pivot_table(index="label", columns="dataset", values="hits@1_mean").fillna(0)
    pivot_std = sub.pivot_table(index="label", columns="dataset", values="hits@1_std").fillna(0)

    # Sort by Set B descending → best at top (after invert_yaxis)
    if "mimic_setB" in pivot.columns:
        pivot     = pivot.sort_values("mimic_setB", ascending=True)
        pivot_std = pivot_std.loc[pivot.index]

    cat_map = df.drop_duplicates("model").set_index("model")["category"].to_dict()
    cat_map = {short(k): v for k, v in cat_map.items()}

    datasets = [c for c in ["mimic_setA", "mimic_setB"] if c in pivot.columns]
    n  = len(pivot)
    bw = 0.35
    y  = np.arange(n)

    ds_meta = {
        "mimic_setA": ("Set A — all-caps",    0.75),
        "mimic_setB": ("Set B — normalised",  1.00),
    }

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)

    for i, ds in enumerate(datasets):
        offset = (i - (len(datasets) - 1) / 2.0) * bw
        vals   = pivot[ds].values if ds in pivot.columns else np.zeros(n)
        errs   = pivot_std[ds].values if ds in pivot_std.columns else np.zeros(n)
        colors = [PALETTE.get(cat_map.get(m, "baseline"), "#999999") for m in pivot.index]
        label, alpha = ds_meta[ds]
        ax.barh(y + offset, vals, bw * 0.92, xerr=errs,
                color=colors, alpha=alpha, zorder=3,
                label=label,
                error_kw={"elinewidth": 1.2, "capsize": 4, "ecolor": "black"})

    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index, fontsize=8.5, ha="right")
    ax.set_xlabel("Hits@1 (95% bootstrap CI)")

    # X-limit: cover the furthest error bar tip + small padding
    max_tip = max(
        (pivot[ds] + pivot_std[ds]).max()
        for ds in datasets if ds in pivot.columns
    )
    ax.set_xlim(0, max_tip + 0.04)

    ax.axvline(0.5, color="#AAAAAA", lw=1.0, ls="--", alpha=0.7, label="0.5 reference")
    style_ax(ax, grid_axis="y")

    ax.set_title("Embedding Model Comparison: MIMIC-IV Ontology Matching")

    # Category legend (colours)
    cat_patches = [mpatches.Patch(color=c, label=cat.capitalize())
                   for cat, c in PALETTE.items()]
    # Dataset legend (opacity)
    ds_patches = [
        mpatches.Patch(facecolor="#888888", alpha=0.75, label="Set A — all-caps"),
        mpatches.Patch(facecolor="#888888", alpha=1.00, label="Set B — normalised"),
    ]
    # Reference line
    from matplotlib.lines import Line2D
    ref_line = Line2D([0], [0], color="#AAAAAA", lw=1.0, ls="--", label="Hits@1 = 0.5 reference")

    all_handles = cat_patches + ds_patches + [ref_line]
    ax.legend(handles=all_handles, title="Category / Dataset",
              loc="upper center", bbox_to_anchor=(0.5, -0.09),
              ncol=4, frameon=False, fontsize=8)

    save(fig, "fig1_embedding_comparison")


# ────────────────────────────────────────────────────────────────────────────
# Fig 2 — Casing sensitivity: Δ Hits@1 (Set B − Set A)
# ────────────────────────────────────────────────────────────────────────────
def fig_casing_delta(df):
    """
    Fig 2: Stacked delta bar chart.
    Each model has two bar segments stacked horizontally:
      - Grey segment: Set A H@1 (the baseline absolute score)
      - Category-coloured segment: delta (Set B − Set A, i.e., the casing gain)
    Total bar length = Set B H@1.
    Sorted by delta descending (most casing-sensitive at top).
    ★ marks models where Set A = 0.000.
    """
    from matplotlib.lines import Line2D

    sub_a = df[df["dataset"] == "mimic_setA"].set_index("model")
    sub_b = df[df["dataset"] == "mimic_setB"].set_index("model")
    if sub_a.empty or sub_b.empty:
        return

    common = sub_a.index.intersection(sub_b.index)
    setA   = sub_a.loc[common, "hits@1_mean"]
    setB   = sub_b.loc[common, "hits@1_mean"]
    delta  = (setB - setA).sort_values(ascending=True)  # ascending → most sensitive at top after invert

    cats = (df.drop_duplicates("model").set_index("model")["category"]
              .reindex(delta.index).fillna("baseline"))

    fig, ax = plt.subplots(figsize=(9, 6.5), constrained_layout=True)
    y = np.arange(len(delta))

    for i, model in enumerate(delta.index):
        a_val  = setA[model]
        d_val  = delta[model]
        cat    = cats[model]
        color  = PALETTE.get(cat, "#999999")
        star   = " ★" if a_val == 0.0 else ""

        # Grey segment: Set A baseline
        if a_val > 0:
            ax.barh(i, a_val, height=0.65, color="#CCCCCC",
                    edgecolor="none", zorder=3)

        # Coloured segment: casing gain (delta), starts at Set A
        if d_val > 0:
            ax.barh(i, d_val, left=a_val, height=0.65,
                    color=color, alpha=0.90, edgecolor="none", zorder=3)

        # Tip annotation: show delta value
        tip = a_val + d_val
        ax.text(tip + 0.008, i, f"Δ{d_val:.2f}{star}",
                va="center", ha="left", fontsize=7.5, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels([short(m) for m in delta.index], fontsize=8.5)
    ax.set_xlabel("Hits@1")
    ax.set_xlim(0, 1.12)
    ax.axvline(0, color="#CCCCCC", lw=0.6, zorder=0)
    ax.invert_yaxis()   # most sensitive at top
    style_ax(ax, grid_axis="x")
    ax.set_title("Casing Sensitivity: Set A Baseline + Δ Gain to Set B")

    cat_patches = [mpatches.Patch(color=c, label=cat.capitalize())
                   for cat, c in PALETTE.items()]
    base_patch  = mpatches.Patch(color="#CCCCCC", label="Set A (ALL-CAPS) score")
    star_patch  = mpatches.Patch(facecolor="none", edgecolor="none",
                                 label="★ = Set A score = 0.000")
    ax.legend(handles=cat_patches + [base_patch, star_patch],
              title="Category / Segment",
              bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)

    save(fig, "fig2_casing_delta")


# ────────────────────────────────────────────────────────────────────────────
# Fig 3 — Hits@k line chart: grey background + top-5 highlight
# ────────────────────────────────────────────────────────────────────────────
def fig_hits_at_k(df):
    """
    Fig 3: Heatmap of Hits@k (k=1,5,10) for all models.
    Two panels: Set A (all-caps) | Set B (normalised).
    Models sorted by Set B H@1 descending. Y-tick labels coloured by category.
    Cell values annotated; shared colorbar on the right.
    """
    import matplotlib.colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap

    datasets  = ["mimic_setA", "mimic_setB"]
    k_cols    = ["hits@1_mean", "hits@5_mean", "hits@10_mean"]
    k_labels  = ["H@1", "H@5", "H@10"]
    ds_titles = {"mimic_setA": "Set A — all-caps", "mimic_setB": "Set B — normalised"}

    sub = df[df["dataset"].isin(datasets)].copy()
    sub["label"] = sub["model"].apply(short)

    # Sort order: by H@1 on Set B descending
    ref = (sub[sub["dataset"] == "mimic_setB"]
           .sort_values("hits@1_mean", ascending=False))
    model_order  = ref["label"].tolist()
    cat_map      = ref.set_index("label")["category"].to_dict()

    n_models = len(model_order)
    row_h    = 0.38
    fig_h    = max(4.0, n_models * row_h + 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(10, fig_h), constrained_layout=True)

    # Colormap consistent with project Okabe-Ito palette: white (0) → dark blue (1)
    cmap = LinearSegmentedColormap.from_list(
        "oki_blue", ["#f7f7f7", "#003D6B"], N=256
    )
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    im   = None

    for ax, ds in zip(axes, datasets):
        panel = (sub[sub["dataset"] == ds]
                 .set_index("label")
                 .reindex(model_order)[k_cols])
        mat = panel.values  # (n_models, 3)

        im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm,
                       interpolation="none")

        # Cell value annotations
        for r in range(n_models):
            for c in range(3):
                val = mat[r, c]
                if np.isnan(val):
                    continue
                txt_color = "white" if (val > 0.75 or val < 0.20) else "black"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=txt_color)

        ax.set_xticks(range(3))
        ax.set_xticklabels(k_labels)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(model_order, fontsize=8.0)

        # Colour y-tick labels by category
        for tick, model in zip(ax.get_yticklabels(), model_order):
            tick.set_color(PALETTE.get(cat_map.get(model, "baseline"), "#333333"))

        ax.set_title(ds_titles[ds])
        ax.tick_params(axis="both", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)

    # Hide y-tick labels on the right panel (shared with left)
    axes[1].set_yticklabels([])

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes, orientation="vertical",
                        fraction=0.025, pad=0.02, shrink=0.9)
    cbar.set_label("Hits@k", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Category legend
    cat_patches = [mpatches.Patch(color=c, label=cat.capitalize())
                   for cat, c in PALETTE.items()]
    fig.legend(handles=cat_patches, title="Category", frameon=False,
               fontsize=8, title_fontsize=8,
               loc="lower center", ncol=4, bbox_to_anchor=(0.45, -0.04))

    fig.suptitle("Hits@k — MIMIC-IV Ontology Matching", fontsize=11, fontweight="bold")
    save(fig, "fig3_hits_at_k")


# ────────────────────────────────────────────────────────────────────────────
# Fig 4 — Speed vs. Accuracy scatter with Pareto frontier
# ────────────────────────────────────────────────────────────────────────────
def _pareto_front(latencies, accuracies):
    """Return indices of Pareto-optimal points (lower latency, higher accuracy)."""
    pts = list(zip(latencies, accuracies, range(len(latencies))))
    pareto = []
    for lat, acc, idx in sorted(pts, key=lambda x: x[0]):  # sort by latency
        if not pareto or acc > pareto[-1][1]:
            pareto.append((lat, acc, idx))
    return pareto


def fig_speed_accuracy(df):
    """
    Fig 4: Two-panel horizontal bar chart — Accuracy (H@1 Set B) | Latency (ms).
    Models sorted by H@1 descending (best at top). Pareto-optimal models marked ★.
    RapidFuzz latency capped at MAX_MS with annotation.
    """
    from matplotlib.lines import Line2D

    sub = df[df["dataset"] == "mimic_setB"].copy()
    sub["label"] = sub["model"].apply(short)
    sub = sub.sort_values("hits@1_mean", ascending=False).reset_index(drop=True)

    # ── Pareto-optimal: no other model is both faster AND more accurate ───────
    lats = sub["time_ms_mean"].values
    accs = sub["hits@1_mean"].values
    pareto_set = set()
    for i in range(len(sub)):
        dominated = any(
            j != i and lats[j] <= lats[i] and accs[j] >= accs[i]
            and (lats[j] < lats[i] or accs[j] > accs[i])
            for j in range(len(sub))
        )
        if not dominated:
            pareto_set.add(i)

    # ── Clip extreme latency (RapidFuzz ~5 s) ────────────────────────────────
    MAX_MS = 1500
    sub["time_plot"] = sub["time_ms_mean"].clip(upper=MAX_MS)
    clipped_mask = sub["time_ms_mean"] > MAX_MS

    colors = [PALETTE.get(sub.loc[i, "category"], "#999999") for i in sub.index]
    n = len(sub)
    y = np.arange(n)

    # ── Y-axis labels: add ★ for Pareto models ───────────────────────────────
    ylabels = [
        f"{sub.loc[i, 'label']} ★" if i in pareto_set else sub.loc[i, "label"]
        for i in range(n)
    ]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 6), constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1]}
    )

    # ── Left: Accuracy bars ───────────────────────────────────────────────────
    ax1.barh(y, sub["hits@1_mean"].values,
             color=colors, edgecolor="none", height=0.7, zorder=3)
    for yi, val in enumerate(sub["hits@1_mean"].values):
        ax1.text(val + 0.005, yi, f"{val:.3f}",
                 va="center", ha="left", fontsize=7.0, color="#333333")

    ax1.set_yticks(y)
    ax1.set_yticklabels(ylabels, fontsize=8.5)
    ax1.set_xlabel("Hits@1 (Set B — normalised)")
    ax1.set_xlim(0, 1.08)
    ax1.set_title("Accuracy", fontsize=10, fontweight="bold")
    ax1.invert_yaxis()
    style_ax(ax1, grid_axis="x")

    # ── Right: Latency bars ───────────────────────────────────────────────────
    ax2.barh(y, sub["time_plot"].values,
             color=colors, edgecolor="none", height=0.7, zorder=3, alpha=0.75)
    for yi, (val, is_clip, true_val) in enumerate(
            zip(sub["time_plot"].values, clipped_mask, sub["time_ms_mean"].values)):
        label = f"→ {true_val / 1000:.1f}k ms" if is_clip else f"{val:.0f} ms"
        ax2.text(val + 15, yi, label,
                 va="center", ha="left", fontsize=7.0, color="#333333")

    ax2.set_yticks(y)
    ax2.set_yticklabels([])          # shared model order visible on left panel
    ax2.set_xlabel("Query Latency (ms / query, capped at 1 500 ms)")
    ax2.set_xlim(0, MAX_MS * 1.25)
    ax2.set_title("Speed", fontsize=10, fontweight="bold")
    ax2.invert_yaxis()
    style_ax(ax2, grid_axis="x")

    fig.suptitle("Speed vs. Accuracy Trade-off — MIMIC-IV Set B",
                 fontsize=11, fontweight="bold")

    cat_patches = [mpatches.Patch(color=c, label=cat.capitalize())
                   for cat, c in PALETTE.items()]
    star_entry = mpatches.Patch(facecolor="none", edgecolor="none",
                                label="★ = Pareto-optimal")
    ax2.legend(handles=cat_patches + [star_entry], title="Category",
               bbox_to_anchor=(1.01, 1), loc="upper left",
               ncol=2, frameon=False)

    save(fig, "fig4_speed_accuracy")


# ────────────────────────────────────────────────────────────────────────────
# Fig 5 — Critical Difference lollipop chart (average rank across datasets)
# ────────────────────────────────────────────────────────────────────────────
def fig_cd_diagram(df):
    """
    Fig 5: Lollipop chart of average rank across all available datasets.
    BM25 excluded (non-embedding baseline). Axis annotated '← Better'.
    """
    # Use k=3 datasets as blocks (Demšar 2006): microbiome, mimic_setA, mimic_setB.
    # synonym_setB excluded from Friedman test (different task structure).
    rank_ds = [d for d in ("microbiome", "mimic_setA", "mimic_setB")
               if d in df["dataset"].values]
    if not rank_ds:
        rank_ds = list(df["dataset"].unique())

    # Exclude BM25 (lexical baseline, excluded from Friedman test)
    sub = df[~df["model"].str.contains("BM25", na=False)].copy()
    sub = sub[sub["dataset"].isin(rank_ds)]

    pivot = sub.pivot_table(index="model", columns="dataset",
                            values="hits@1_mean").dropna(how="all")
    if pivot.empty:
        return

    # Rank per dataset (rank 1 = best); sort ascending so best model is at top
    ranks       = pivot.rank(ascending=False, method="average")
    avg_rank    = ranks.mean(axis=1).sort_values(ascending=True)
    median_rank = avg_rank.median()

    cat_map = df.drop_duplicates("model").set_index("model")["category"].to_dict()
    labels  = [short(m) for m in avg_rank.index]
    colors  = [PALETTE.get(cat_map.get(m, "baseline"), "#999999") for m in avg_rank.index]

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    y = np.arange(len(avg_rank))
    # Stems from 0 → rank value; with inverted x-axis 0 is on the RIGHT,
    # so best model (rank 1) has a short stem near the right edge.
    for yi, (rank, color) in enumerate(zip(avg_rank.values, colors)):
        ax.hlines(yi, 0, rank, color="#CCCCCC", lw=1.5, zorder=2)
    ax.scatter(avg_rank.values, y, color=colors, s=80, zorder=3,
               edgecolors="white", linewidths=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.axvline(median_rank, color="#AAAAAA", lw=1.0, ls="--")

    # Invert x-axis: rank 1 (best) on the RIGHT — "further right = better"
    ax.invert_xaxis()

    style_ax(ax, grid_axis="x")
    ax.set_xlabel("Average rank across datasets")
    ax.set_title("Critical Difference — Average Rank Across Datasets")

    from matplotlib.lines import Line2D
    cat_patches = [mpatches.Patch(color=c, label=cat.capitalize())
                   for cat, c in PALETTE.items()]
    median_line = Line2D([0], [0], color="#AAAAAA", lw=1.0, ls="--",
                         label=f"Median rank ({median_rank:.1f})")
    ax.legend(handles=cat_patches + [median_line],
              title="Category", bbox_to_anchor=(1.01, 1),
              loc="upper left", frameon=False)

    # Source note next to median line at the top of the axes
    ax.text(median_rank + 0.15, 0.99,
            f"Friedman k={len(rank_ds)} dataset blocks, N=13 models",
            transform=ax.get_xaxis_transform(),
            fontsize=7, color="#777777", va="top", ha="left")

    save(fig, "fig5_cd_diagram")


# ────────────────────────────────────────────────────────────────────────────
# Fig 6 — Average rank: MIMIC-IV combined (Set A + Set B), includes BM25
# ────────────────────────────────────────────────────────────────────────────
def fig_avg_rank(df):
    """
    Fig 6: Connected dot plot — Set A rank vs Set B rank per model.
    Model names on y-axis (sorted by average rank, best at top).
    X-axis = rank 1–N, inverted so rank 1 is on the right.
    Circle = Set A rank, diamond = Set B rank, connected by a thin line.
    Category colours. Includes BM25.
    """
    from matplotlib.lines import Line2D

    mimic_ds = [d for d in ["mimic_setA", "mimic_setB"] if d in df["dataset"].values]
    if len(mimic_ds) < 2:
        return

    pivot = df[df["dataset"].isin(mimic_ds)].pivot_table(
        index="model", columns="dataset", values="hits@1_mean"
    ).dropna(how="all")
    if pivot.empty:
        return

    ranks    = pivot.rank(ascending=False, method="average")
    avg_rank = ranks.mean(axis=1).sort_values(ascending=True)  # best at top
    n        = len(avg_rank)

    cat_map = df.drop_duplicates("model").set_index("model")["category"].to_dict()
    labels  = [short(m) for m in avg_rank.index]
    y       = np.arange(n)

    fig, ax = plt.subplots(figsize=(9, 6.5), constrained_layout=True)

    for i, model in enumerate(avg_rank.index):
        r_a   = ranks.loc[model, "mimic_setA"]
        r_b   = ranks.loc[model, "mimic_setB"]
        cat   = cat_map.get(model, "baseline")
        color = PALETTE.get(cat, "#999999")

        # Connecting line
        ax.hlines(i, r_a, r_b, color=color, lw=1.8, alpha=0.6, zorder=2)
        # Set A: circle
        ax.scatter(r_a, i, color=color, marker="o", s=70,
                   edgecolors="white", linewidths=0.5, zorder=4)
        # Set B: diamond
        ax.scatter(r_b, i, color=color, marker="D", s=55,
                   edgecolors="white", linewidths=0.5, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Rank  (rank 1 = best, on the right)")
    ax.set_xlim(n + 0.8, 0.2)    # invert: rank 1 on right
    ax.set_xticks(range(1, n + 1))
    style_ax(ax, grid_axis="x")
    ax.set_title("Rank Comparison — MIMIC-IV Set A vs Set B")

    cat_handles = [mpatches.Patch(color=c, label=cat.capitalize())
                   for cat, c in PALETTE.items()]
    marker_handles = [
        Line2D([0], [0], marker="o", color="grey", linestyle="none",
               markersize=6, label="Set A (ALL-CAPS)"),
        Line2D([0], [0], marker="D", color="grey", linestyle="none",
               markersize=5, label="Set B (normalised)"),
    ]
    ax.legend(handles=cat_handles + marker_handles,
              title="Category / Condition",
              bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)

    save(fig, "fig6_avg_rank")


# ────────────────────────────────────────────────────────────────────────────
# Fig S1 — MIMIC-IV organism frequency overview (supplementary)
# ────────────────────────────────────────────────────────────────────────────
def fig_mimic_organisms():
    """
    Fig S1: Top 20 MIMIC-IV organisms by frequency.
    Bars coloured by resolution status (resolved=blue, unresolved=grey).
    X-axis formatted as Xk, sorted by frequency descending, panel label S1.
    """
    org_path = f"{OUTPUTS}/mimic_organisms.csv"
    if not os.path.exists(org_path):
        print("  mimic_organisms.csv not found — skipping figS1")
        return

    org   = pd.read_csv(org_path)
    top20 = org.nlargest(20, "frequency").copy()

    # Standardize organism name casing: title-case on normalized name
    top20["display_name"] = top20["org_name_normalized"].str.title()

    status_colors = {
        "resolved":   "#0072B2",  # blue
        "unresolved": "#999999",  # grey
    }
    colors = [status_colors.get(s, "#999999") for s in top20["resolution_status"]]

    # Diverging values: resolved → right (+), unresolved → left (−)
    top20["freq_resolved"]   = top20["frequency"].where(
        top20["resolution_status"] == "resolved", 0)
    top20["freq_unresolved"] = top20["frequency"].where(
        top20["resolution_status"] == "unresolved", 0)

    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)

    y = np.arange(len(top20))
    # Resolved bars → right (positive)
    ax.barh(y,  top20["freq_resolved"].values,
            color=status_colors["resolved"], edgecolor="none",
            height=0.7, zorder=3, label="Resolved")
    # Unresolved bars → left (negative)
    ax.barh(y, -top20["freq_unresolved"].values,
            color=status_colors["unresolved"], edgecolor="none",
            height=0.7, zorder=3, label="Unresolved")

    # Tip annotations
    for yi, row in enumerate(top20.itertuples()):
        if row.freq_resolved > 0:
            ax.text(row.freq_resolved + top20["frequency"].max() * 0.01, yi,
                    f"{int(row.freq_resolved / 1000)}k",
                    va="center", ha="left", fontsize=9.0, color="#333333")
        if row.freq_unresolved > 0:
            ax.text(-(row.freq_unresolved + top20["frequency"].max() * 0.01), yi,
                    f"{int(row.freq_unresolved / 1000)}k",
                    va="center", ha="right", fontsize=9.0, color="#333333")

    ax.set_yticks(y)
    ax.set_yticklabels(top20["display_name"].values, fontsize=8.5)
    ax.invert_yaxis()

    # Symmetric x-axis formatted as Xk
    x_max = top20["frequency"].max() * 1.18
    ax.set_xlim(-x_max, x_max)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(abs(x) / 1000)}k")
    )
    ax.axvline(0, color="#333333", lw=0.8, zorder=4)
    ax.set_xlabel("← Unresolved   |   Frequency (isolates)   |   Resolved →")
    style_ax(ax, grid_axis="y")

    ax.set_title("Top 20 Organisms in MIMIC-IV — Ontology Resolution Status")

    # Panel label S1
    ax.text(0.01, 0.99, "S1", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top")

    patches = [mpatches.Patch(color=c, label=s.capitalize())
               for s, c in status_colors.items()]
    ax.legend(handles=patches, title="Resolution Status",
              bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False)

    save(fig, "figS1_mimic_organisms")


# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────
def main():
    print("\nFigure Generation (publication-ready revision)")
    print("=" * 70)
    print(f"  OUTPUTS: {OUTPUTS}")
    print(f"  FIGURES: {FIGURES}\n")

    exp1 = load_exp1()

    if exp1 is not None:
        print(f"  Loaded {len(exp1)} rows across {exp1['dataset'].nunique()} datasets")
        print(f"  Models: {exp1['model'].nunique()}\n")
        print("Generating figures...")
        fig_embedding_comparison(exp1)
        fig_casing_delta(exp1)
        fig_hits_at_k(exp1)
        fig_speed_accuracy(exp1)
        fig_cd_diagram(exp1)
        fig_avg_rank(exp1)
    else:
        print("  No Exp1 results found — check OUTPUTS path.")

    fig_mimic_organisms()

    print(f"\nAll figures saved to: {FIGURES}/")
    print("Done.\n")


if __name__ == "__main__":
    main()
