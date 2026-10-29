"""EasIFA's own ablation on its Swiss-Prot benchmark: macro F1 gained from the 3D-structure and the reaction branch,
per identity band to EasIFA's training set, with paired 95% CIs. Both series are grey: two components of another
model. Data: figures/data/easifa_ablation.json.

  python figures/plot_easifa_ablation.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.patches as mpatches

import figstyle as fs

DATA = "figures/data/easifa_ablation.json"
OUT = "easifa_ablation.png"
W_PX, H_PX = 1600, 440      # wide enough that neighbouring value labels do not touch
BAR_W = 0.34
GAP = 0.02
XLABEL = "Sequence identity to the EasIFA training set"


def de(value):
    return fs.num(value, 3, signed=True)


def color_for(key):
    return fs.BASELINE_ALT if key == "baseline_alt" else fs.BASELINE


def main():
    with open(DATA) as f:
        d = json.load(f)
    fig, ax = fs.figure(W_PX, H_PX)

    groups, series = d["groups"], d["series"]
    n_series = len(series)
    handles = []
    for s_i, s in enumerate(series):
        color = color_for(s["color_key"])
        offset = (s_i - (n_series - 1) / 2) * (BAR_W + GAP)
        for g_i, _ in enumerate(groups):
            x = g_i + offset
            value = s["values"][g_i]
            fs.rounded_bar(ax, x, value, BAR_W, color)
            fs.ci_errorbar_v(ax, x, tuple(s["ci"][g_i]), fs.darken(color))
            # value above the upper CI end, not above the bar: otherwise it sits in the error bar
            fs.value_label(ax, x, s["ci"][g_i][1], de(value), dy=8)
        handles.append(mpatches.Patch(facecolor=color, edgecolor="none", label=s["name"]))

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g}\n(n = {n})" for g, n in zip(groups, d["n"])])
    ax.set_xlim(-0.5, len(groups) - 0.5)
    # some room below 0 so a CI crossing zero stays visible
    ax.set_ylim(-0.012, d["ymax"])
    fs.finish(ax, xlabel=XLABEL, ylabel=d["ylabel"], zero_line="h")
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0, 1, 0.12),
              ncol=n_series, handlelength=1.2, columnspacing=1.6, borderaxespad=0.0)
    fs.save(fig, OUT)


if __name__ == "__main__":
    main()
