"""The same Flat model on four frozen embeddings, 5-fold over 962 M-CSA proteins: (A) macro AUPRC, (B) macro F1,
(C) paired difference of each variant minus the main model with 95% CI (filled = AUPRC, hollow = F1).
Data: figures/data/plm_series.json.

  python figures/plot_plm_series.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import figstyle as fs

DATA = "figures/data/plm_series.json"
OUT = "plm_series.png"
W_PX, H_PX = 1600, 520
BAR_H = 0.62


def de(x, nd=3):
    return fs.num(x, nd)


def main():
    d = json.load(open(DATA))
    rows = d["rows"]
    n = len(rows)
    fig, (a1, a2, a3) = fs.figure(W_PX, H_PX, ncols=3, sharey=True,
                                  gridspec_kw={"width_ratios": [1.2, 1.2, 1.4], "wspace": 0.10})
    for ax, key, title, letter in ((a1, "macro_auprc", "Macro AUPRC", "A"), (a2, "macro_f1", "Macro F1", "B")):
        for i, r in enumerate(rows):
            y = n - 1 - i
            color = fs.BRANCH["flat"] if r["main"] else fs.BASELINE_ALT
            fs.rounded_barh(ax, y, r[key], BAR_H, color)
            fs.value_label(ax, r[key] + 0.015, y, de(r[key]), ha="left", va="center")
        ax.set_xlim(0, 1.0)                 # bar axis starts at 0
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])   # no tick at the panel edge (collided with the next panel's 0)
        fs.finish(ax, xlabel=title, grid_axis="x")
        fs.panel(ax, letter, title)   # "5-fold, 962 proteins" goes into the caption
    a1.set_yticks(range(n))
    a1.set_yticklabels([r["label"] for r in reversed(rows)])
    a1.set_ylim(-0.6, n - 0.4)

    # paired difference to the main model: the CI carries the significance
    for i, r in enumerate(rows):
        if r["main"]:
            continue
        y = n - 1 - i
        for j, (dk, ck) in enumerate((("delta_auprc", "ci_auprc"), ("delta_f1", "ci_f1"))):
            yy = y + (0.19 if j == 0 else -0.19)
            proven = r[ck][1] < 0 or r[ck][0] > 0
            colour = fs.INK_SOFT if not proven else fs.BASELINE
            a3.plot(r[ck], [yy, yy], color=colour, linewidth=2.0, solid_capstyle="butt", zorder=3)
            a3.plot([r[dk]], [yy], marker="o", markersize=11, linestyle="none", zorder=4,
                    markerfacecolor=colour if j == 0 else fs.SURFACE,
                    markeredgecolor=fs.SURFACE if j == 0 else colour, markeredgewidth=2)
            fs.value_label(a3, r[ck][0], yy, fs.num(r[dk], 3, signed=True) + "  ", ha="right", va="center")
    a3.set_xlim(-0.22, 0.02)
    fs.finish(a3, xlabel="Δ (variant − layer 30), 95% CI", grid_axis="x", zero_line=True)
    fs.panel(a3, "C", "Paired difference")
    marks = [a3.plot([], [], marker="o", markersize=11, linestyle="none", markerfacecolor=fs.BASELINE,
                     markeredgecolor=fs.SURFACE, markeredgewidth=2, label="AUPRC")[0],
             a3.plot([], [], marker="o", markersize=11, linestyle="none", markerfacecolor=fs.SURFACE,
                     markeredgecolor=fs.BASELINE, markeredgewidth=2, label="F1")[0]]
    a3.legend(handles=marks, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, handlelength=1.0,
              handletextpad=0.3, columnspacing=0.8, borderaxespad=0.2)
    fs.save(fig, OUT)


if __name__ == "__main__":
    main()
