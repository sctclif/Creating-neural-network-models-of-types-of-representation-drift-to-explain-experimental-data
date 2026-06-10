"""
Shared matplotlib style for all thesis figures.

Usage — add to every figure notebook imports cell:
    from plot_style import apply_style, COLORS, CMAPS, FIG_SIZES, FONT
Then call apply_style() once before building any figure.
"""
import matplotlib as mpl


# ── Colours ───────────────────────────────────────────────────────────────────
COLORS = {
    "input":          "steelblue",      # stimulus trace fill
    "highlight":      "crimson",        # selected neuron / emphasis
    "neutral":        "gray",           # secondary lines, day markers (light bg)
    "day_line":       "white",          # day boundary lines on dark heatmaps
    "day_line_light": "gray",           # day boundary lines on light backgrounds
}

# ── Colormaps ─────────────────────────────────────────────────────────────────
CMAPS = {
    "firing_rate":    "cividis",
    "excitability":   "viridis",
    "weights":        "binary",
    "diverging":      "RdBu_r",
    "heatmap":        "inferno",
}

# LaTeX A4 linewidth: 455.24411pt ÷ 72.27pt/in
LINEWIDTH_IN = 455.24411 / 72.27   # ≈ 6.30 inches — use for scale=1 exports

# ── Figure sizes (width × height, inches) ─────────────────────────────────────
FIG_SIZES = {
    "full":      (14, 11),          # 4-panel summary figures (screen/draft)
    "full_a4":   (LINEWIDTH_IN, 7.5),  # 4-panel, fits A4 linewidth at scale=1
    "wide":      (12,  4),          # single-row multi-panel
    "wide_a4":   (LINEWIDTH_IN, 3.2),  # single-row, A4 linewidth at scale=1
    "half":      ( 7,  5),          # half-page single panel
    "half_a4":   (LINEWIDTH_IN / 2, 2.5),  # half-linewidth panel at scale=1
    "single_a4": (LINEWIDTH_IN, 4.0),      # full-linewidth single panel at scale=1
    "square":    ( 6,  6),          # square single panel
    "tall":      ( 7,  9),          # tall two-panel column
}

# ── Font sizes (points) ───────────────────────────────────────────────────────
FONT = {
    "panel_label":  13,
    "title":        10,
    "axis_label":    9,
    "tick":          8,
    "colorbar":      8,
    "annotation":    8,
}

# ── Line weights ──────────────────────────────────────────────────────────────
LW = {
    "trace":      1.0,
    "day_line":   0.8,
    "spine":      0.8,
}


def apply_style():
    """Apply shared rcParams. Call once per notebook before building figures."""
    mpl.rcParams.update({
        # Font
        "font.family":           "sans-serif",
        "font.size":             FONT["axis_label"],
        "axes.labelsize":        FONT["axis_label"],
        "axes.titlesize":        FONT["title"],
        "xtick.labelsize":       FONT["tick"],
        "ytick.labelsize":       FONT["tick"],
        # Spines
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "axes.linewidth":        LW["spine"],
        # Ticks
        "xtick.major.width":     LW["spine"],
        "ytick.major.width":     LW["spine"],
        "xtick.direction":       "out",
        "ytick.direction":       "out",
        # Save
        "figure.dpi":            150,
        "savefig.dpi":           300,
        "savefig.bbox":          "tight",
    })
