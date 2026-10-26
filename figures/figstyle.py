"""Shared figure style: palette, font sizes and drawing helpers. Plot scripts take every colour and size from here."""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------------------------------------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------------------------------------------
# The four branch colours are validated as an ordered set (colour-vision deficiency, chroma, contrast): worst adjacent
# pair dE 13.5 (protanopia), normal vision dE 17.1, all >= 3:1 against the chart surface. Keep the order; a fifth
# series is folded into "Other" or faceted.
BRANCH = {
    "flat": "#0A918C",          # teal: our model
    "sequence": "#C2681A",      # ochre
    "adapter": "#B0407A",       # magenta
    "structural": "#3A63CC",    # blue
}
BRANCH_ORDER = ["flat", "sequence", "adapter", "structural"]

# External methods are always neutral grey ("what we compare against"). A second external variant is a lighter
# step of the same grey: 3.9:1 against BASELINE, 2.1:1 against the Flat teal.
BASELINE = "#4A5A54"
BASELINE_ALT = "#B6C0BB"

# Single-hue ramp for continuous quantities, light to dark (never rainbow or viridis).
TEAL_RAMP = ["#D8ECEA", "#A9D6D2", "#6FBDB8", "#2CA39D", "#0A918C", "#07736F", "#05504D"]

# Text and surfaces. Labels never wear a series colour.
INK = "#14231F"
INK_SOFT = "#4A5A54"
INK_FAINT = "#5C6961"
GRID = "#D5DCD6"
SURFACE = "#F7F8F6"
SURFACE_ALT = "#E9EDE8"
WARN = "#8A4F22"            # caveats, never a series colour
CARTOON = "#C9D0CA"         # cartoon grey of the structure renders (figures/pymol)
SURFACE_DARK = "#14231F"


def sequential(n: int) -> list[str]:
    """n evenly spaced steps of the teal ramp, light to dark."""
    if n < 2:
        return [TEAL_RAMP[4]]
    step = (len(TEAL_RAMP) - 1) / (n - 1)
    return [TEAL_RAMP[round(i * step)] for i in range(n)]


def darken(hex_color: str, factor: float = 0.75) -> str:
    """Darker step of a colour (1.0 = unchanged, 0.0 = black)."""
    return mcolors.to_hex([max(0.0, c * factor) for c in mcolors.to_rgb(hex_color)])


# ---------------------------------------------------------------------------------------------------------------
# Font
# ---------------------------------------------------------------------------------------------------------------
# IBM Plex Sans (SIL Open Font License) if installed, else DejaVu Sans:
#   mkdir -p ~/.fonts && cd ~/.fonts && curl -LO https://github.com/IBM/plex/releases/latest/download/TrueType.zip
#   unzip -j TrueType.zip '*IBMPlexSans-*' -d . && fc-cache -f
def _register_fonts(extra_dir: str | None = None) -> str:
    candidates = [Path.home() / ".fonts", Path("/usr/share/fonts")]
    if extra_dir:
        candidates.insert(0, Path(extra_dir))
    for d in candidates:
        if d.exists():
            for ttf in d.rglob("IBMPlexSans*.ttf"):
                try:
                    font_manager.fontManager.addfont(str(ttf))
                except Exception:
                    pass
    if "IBM Plex Sans" in {f.name for f in font_manager.fontManager.ttflist}:
        return "IBM Plex Sans"
    warnings.warn("IBM Plex Sans not found, falling back to DejaVu Sans.")
    return "DejaVu Sans"


FONT = _register_fonts()

# Font sizes in pt. With figsize = slide pixels / 100, 1 pt is about 1.39 px on the slide, so 18 pt is ~25 px,
# above the 24 px floor. Nothing smaller.
FS_TICK = 18
FS_LABEL = 20
FS_ANNOT = 22
FS_DIRECT = 22

mpl.rcParams.update({
    "font.family": FONT,
    "mathtext.fontset": "custom",           # formulas ($p_t$) in the same font as the axis titles
    "mathtext.rm": FONT,
    "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold",
    "mathtext.cal": f"{FONT}:italic",
    "font.size": FS_TICK,
    "text.color": INK_SOFT,
    "axes.labelcolor": INK_SOFT,
    "axes.edgecolor": GRID,
    "axes.labelsize": FS_LABEL,
    "axes.titlesize": FS_LABEL,
    "axes.linewidth": 0.8,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.fontsize": FS_TICK,
    "lines.linewidth": 2.0,
    "lines.markersize": 9,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.transparent": True,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

# ---------------------------------------------------------------------------------------------------------------
# Figures and axes
# ---------------------------------------------------------------------------------------------------------------
DPI = 200          # layout resolution (figsize = pixels / 100)
SAVE_DPI = 400     # pixel density of the saved PNG


def figure(width_px: int, height_px: int, **kwargs):
    """Figure sized to a placeholder of width_px x height_px on a 1920 x 1080 slide."""
    return plt.subplots(figsize=(width_px / 100, height_px / 100), dpi=DPI, **kwargs)


def finish(ax, *, xlabel: str = "", ylabel: str = "", grid_axis: str = "y", zero_line: bool | str = False):
    """Remove top/right spines, light grid on grid_axis ('x', 'y', 'both', 'none'), axis titles; zero_line True/'v'
    draws a vertical dashed zero line (differences on x), 'h' a horizontal one."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    if grid_axis in ("y", "both"):
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    if grid_axis in ("x", "both"):
        ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=12)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=12)
    if zero_line:
        draw = ax.axhline if zero_line == "h" else ax.axvline
        draw(0, color=BASELINE, linewidth=1.5, linestyle=(0, (4, 3)), zorder=1)
    return ax


BETTER = {"higher": "↑ higher is better", "lower": "↓ lower is better"}


def better(ax, direction: str = "higher"):
    """Reading direction of the metric as a small note above the axes, top left."""
    return ax.annotate(BETTER[direction], (0, 1), xycoords="axes fraction", xytext=(0, 10),
                       textcoords="offset points", ha="left", va="bottom", fontsize=FS_TICK, color=INK_SOFT)


def ci_errorbar_v(ax, x: float, ci: tuple[float, float], color: str):
    """Vertical confidence interval without a point marker (the value sits on a bar)."""
    lo, hi = ci
    centre = (lo + hi) / 2
    ax.errorbar(x, centre, yerr=[[centre - lo], [hi - centre]], fmt="none", ecolor=color, elinewidth=1.8,
                capsize=4, capthick=1.8, zorder=4)


def ci_errorbar_h(ax, y: float, ci: tuple[float, float], color: str, *, point: float | None = None):
    """Horizontal confidence interval with its point estimate (for differences)."""
    lo, hi = ci
    centre = point if point is not None else (lo + hi) / 2
    ax.errorbar(centre, y, xerr=[[centre - lo], [hi - centre]], fmt="o", color=color, ecolor=color, elinewidth=1.8,
                capsize=4, capthick=1.8, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)


def rounded_bar(ax, x: float, height: float, width: float, color: str, alpha: float = 1.0):
    return ax.bar(x, height, width=width, color=color, alpha=alpha, align="center", zorder=3)[0]


def rounded_barh(ax, y: float, width: float, height: float, color: str, alpha: float = 1.0):
    return ax.barh(y, width, height=height, color=color, alpha=alpha, align="center", zorder=3)[0]


def value_label(ax, x, y, text, *, va="bottom", ha="center", dy=0.0):
    """Value next to its mark, always in text colour."""
    return ax.annotate(text, (x, y), xytext=(0, dy), textcoords="offset points", ha=ha, va=va,
                       fontsize=FS_DIRECT, color=INK, zorder=5)


def direct_label(ax, x, y, text, color_dot=None, *, dx=12):
    """Series name at the end of its line instead of a legend; an optional coloured dot carries the identity."""
    if color_dot:
        ax.plot([x], [y], "o", color=color_dot, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
    return ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points", ha="left", va="center",
                       fontsize=FS_DIRECT, color=INK, zorder=5)


def num(x, nd: int = 3, signed: bool = False) -> str:
    """Number for labels: decimal point, true minus sign (U+2212), optional explicit plus."""
    t = f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"
    return t.replace("-", "−")


def panel(ax, letter: str, title: str = ""):
    """Bold panel letter and optional short panel title, top left above the axes."""
    text = f"$\\bf{{{letter}}}$  {title}" if title else f"$\\bf{{{letter}}}$"
    return ax.annotate(text, (0, 1), xycoords="axes fraction", xytext=(0, 14), textcoords="offset points",
                       ha="left", va="bottom", fontsize=FS_LABEL, color=INK)


def save(fig, filename: str, outdir: str = "figures/output"):
    """Save a transparent PNG at SAVE_DPI."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    fig.savefig(path, dpi=SAVE_DPI, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"written: {path}")
    return path
