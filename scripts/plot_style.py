"""
plot_style.py — Shared publication style for the benchmark figures.

Nature/Bioinformatics-grade standards.
Import at the top of every figure script:

    from plot_style import PALETTE, MARKERS, LINESTYLES, apply_style, style_ax, save_fig
"""

import matplotlib.pyplot as plt

# ── Okabe-Ito colorblind-safe palette ────────────────────────────────────────
PALETTE = {
    "general":    "#0072B2",  # blue
    "biomedical": "#E69F00",  # orange
    "clinical":   "#009E73",  # green
    "baseline":   "#999999",  # grey
}

# ── Markers per category (for B&W printability) ─────────────────────────────
MARKERS = {
    "general":    "o",   # circle
    "biomedical": "s",   # square
    "clinical":   "D",   # diamond
    "baseline":   "^",   # triangle up
}

# ── Line styles per category ─────────────────────────────────────────────────
LINESTYLES = {
    "general":    "-",    # solid
    "biomedical": "--",   # dashed
    "clinical":   ":",    # dotted
    "baseline":   "-.",   # dash-dot
}

# ── Global rcParams (Nature/Bioinformatics typography) ───────────────────────
RC_PARAMS = {
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica Neue", "DejaVu Sans", "Arial"],
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   8.5,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.linewidth":    0.8,
    "lines.linewidth":   1.0,
    "axes.spines.top":   False,
    "axes.spines.right": False,
}


def apply_style():
    """Apply global rcParams. Call once at the top of every figure script."""
    plt.rcParams.update(RC_PARAMS)


def style_ax(ax, grid_axis="y"):
    """
    Apply standard spine cleanup and gridlines to an axis.
    - Removes top and right spines
    - Adds light grey gridlines (zorder=0) on the requested axis
    - Draws bars/points above gridlines (set_axisbelow)
    """
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    if grid_axis in ("y", "both"):
        ax.yaxis.grid(True, color="#EEEEEE", zorder=0, linewidth=0.8)
    if grid_axis in ("x", "both"):
        ax.xaxis.grid(True, color="#EEEEEE", zorder=0, linewidth=0.8)


def save_fig(fig, path_stem):
    """Save figure as both PDF and PNG at 300 DPI with tight bounding box."""
    fig.savefig(f"{path_stem}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{path_stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path_stem}.pdf / .png")
