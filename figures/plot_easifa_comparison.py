"""EasIFA's 82 M-CSA test proteins, the same proteins for all three models. (A) macro F1, MCC and AUPRC; (B) the
paired difference ours minus each EasIFA variant with its 95% CI (bootstrap over proteins). The CI belongs to the
difference, so panel A has no error bars. Data: figures/data/easifa_comparison.json.

  python figures/plot_easifa_comparison.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.patches as mpatches

import figstyle as fs

DATA = "figures/data/easifa_comparison.json"
OUT = "easifa_comparison.png"
W_PX, H_PX = 1300, 470
BAR_H = 0.24
GAP = 0.02
# as an axis title below rather than a panel title above: the legend of panel A sits on top
DELTA_LABEL = "Δ (ours − EasIFA), 95% CI"


def de(value, digits=3):
    return fs.num(value, digits)


def signed(value, digits=3):
    return fs.num(value, digits, signed=True)


def color_for(key):
    if key == "baseline":
        return fs.BASELINE
    if key == "baseline_alt":
        return fs.BASELINE_ALT
    return fs.BRANCH[key]


def main():
    with open(DATA) as f:
        d = json.load(f)
    fig, (ax, axd) = fs.figure(W_PX, H_PX, ncols=2, sharey=True,
                               gridspec_kw={"width_ratios": [2.5, 1], "wspace": 0.06})

    metrics, series = d["metrics"], d["series"]
    n_series = len(series)
    handles = []
    for s_i, s in enumerate(series):
        color = color_for(s["color_key"])
        # first series on top within a group, hence the sign
        offset = ((n_series - 1) / 2 - s_i) * (BAR_H + GAP)
        for m_i, m in enumerate(metrics):
            y = (len(metrics) - 1 - m_i) + offset
            value = m["values"][s_i]
            fs.rounded_barh(ax, y, value, BAR_H, color)
            fs.value_label(ax, value + 0.012, y, de(value), ha="left", va="center")
        handles.append(mpatches.Patch(facecolor=color, edgecolor="none", label=s["name"]))

    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([m["name"] for m in reversed(metrics)])
    ax.set_xlim(0, d["xmax"])          # bar axes start at 0
    ax.set_ylim(-0.6, len(metrics) - 0.4)
    fs.finish(ax, xlabel="Score (macro average over 82 proteins)", grid_axis="x")
    fs.panel(ax, "A", "Absolute scores")
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.13, 1, 0.12),
              ncol=n_series, handlelength=1.2, columnspacing=1.4, borderaxespad=0.0)

    # --- panel B: the paired difference, which carries the significance ---
    # two points per metric: filled = vs ESM-bin, hollow = vs SaProt-bin
    D_OFF = 0.17
    marks = []
    for m_i, m in enumerate(metrics):
        y0 = len(metrics) - 1 - m_i
        for j, dd in enumerate(m["deltas"]):
            y = y0 + (D_OFF if j == 0 else -D_OFF)
            # CI crossing zero -> muted, else the branch colour: colour codes whether the effect is supported
            proven = dd["ci"][0] > 0
            colour = fs.BRANCH["flat"] if proven else fs.INK_SOFT
            axd.plot(dd["ci"], [y, y], color=colour, linewidth=2.0, solid_capstyle="butt", zorder=3)
            axd.plot([dd["delta"]], [y], marker="o", markersize=11, linestyle="none", zorder=4,
                     markerfacecolor=colour if j == 0 else fs.SURFACE, markeredgecolor=colour if j else fs.SURFACE,
                     markeredgewidth=2)
            fs.value_label(axd, dd["ci"][1], y, "  " + signed(dd["delta"]), ha="left", va="center")
    for j, vs in enumerate(metrics[0]["deltas"]):
        marks.append(axd.plot([], [], marker="o", markersize=11, linestyle="none",
                              markerfacecolor=fs.BRANCH["flat"] if j == 0 else fs.SURFACE,
                              markeredgecolor=fs.SURFACE if j == 0 else fs.BRANCH["flat"], markeredgewidth=2,
                              label=f"vs. {vs['vs']}")[0])
    # below the axis title: the top row belongs to the bar legend of panel A
    axd.legend(handles=marks, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, handlelength=1.0,
               handletextpad=0.3, columnspacing=0.8, borderaxespad=0.0)

    axd.set_xlim(*d["delta_xlim"])
    fs.finish(axd, xlabel=DELTA_LABEL, grid_axis="x", zero_line=True)
    fs.panel(axd, "B", "Paired difference")
    fs.save(fig, OUT)


if __name__ == "__main__":
    main()
