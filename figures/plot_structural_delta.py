"""Does 3D structure help on top of the sequence embedding? Per training size the paired difference in pooled
AUPRC, Structural minus Flat: filled = Flat on the same ESMC-6B features and data (controlled), hollow = the main
model (not controlled: language model and data differ). CIs excluding zero in the structural colour.
Data: figures/data/structural_delta.json.

  python figures/plot_structural_delta.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import figstyle as fs

DATA = "figures/data/structural_delta.json"
OUT = "structural_delta.png"
W_PX, H_PX = 1600, 520
OFF = 0.17


def main():
    d = json.load(open(DATA))
    rows = d["rows"]
    color = fs.BRANCH["structural"]
    fig, ax = fs.figure(W_PX, H_PX)
    n = len(rows)
    for i, r in enumerate(rows):
        y = n - 1 - i
        for filled, (delta, ci, yy) in ((True, (r["delta"], r["ci"], y + OFF)),
                                        (False, (r["vs_main"]["delta"], r["vs_main"]["ci"], y - OFF))):
            c = color if (ci[0] > 0 or ci[1] < 0) else fs.INK_SOFT
            ax.plot(ci, [yy, yy], color=c, linewidth=2.0, solid_capstyle="butt", zorder=3)
            ax.plot([delta], [yy], marker="o", markersize=11, linestyle="none", zorder=4,
                    markerfacecolor=c if filled else fs.SURFACE,
                    markeredgecolor=fs.SURFACE if filled else c, markeredgewidth=2)
            if delta >= 0:      # label on the side away from the zero line
                fs.value_label(ax, ci[1], yy, "  " + fs.num(delta, 3, signed=True), ha="left", va="center")
            else:
                fs.value_label(ax, ci[0], yy, fs.num(delta, 3, signed=True) + "  ", ha="right", va="center")
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{r['label']}" for r in reversed(rows)])
    ax.set_xlim(-0.14, 0.07)
    ax.set_ylim(-0.6, n - 0.4)
    fs.finish(ax, xlabel=d["xlabel"], grid_axis="x", zero_line=True)
    ax.tick_params(axis="y", length=0)
    marks = [ax.plot([], [], marker="o", markersize=11, linestyle="none", markerfacecolor=color,
                     markeredgecolor=fs.SURFACE, markeredgewidth=2,
                     label="vs Flat on the same ESMC-6B features")[0],
             ax.plot([], [], marker="o", markersize=11, linestyle="none", markerfacecolor=fs.SURFACE,
                     markeredgecolor=color, markeredgewidth=2,
                     label="vs main model (Flat, ESM-2 650M layer 30)")[0]]
    ax.legend(handles=marks, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, handlelength=1.0,
              handletextpad=0.4, columnspacing=1.6, borderaxespad=0.0)
    fs.save(fig, OUT)


if __name__ == "__main__":
    main()
