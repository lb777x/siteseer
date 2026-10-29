"""Same training data as Squidly: Flat trained only on CataloDB train against the released Squidly 3B ensemble, on
Squidly's CataloDB test set. (A) macro AUPRC and F1, (B) paired difference Flat minus Squidly with 95% CI (crossing
zero in INK_SOFT). Data: figures/data/squidly_same_data.json.

  python figures/plot_squidly_same_data.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.patches as mpatches

import figstyle as fs

DATA = "figures/data/squidly_same_data.json"
OUT = "squidly_same_data.png"
W_PX, H_PX = 1400, 380
BAR_H = 0.34
GAP = 0.03


def de(x, nd=3):
    return fs.num(x, nd)


def signed(x, nd=3):
    return fs.num(x, nd, signed=True)


def main():
    d = json.load(open(DATA))
    rows = d["rows"]
    n = len(rows)
    fig, (ax, axd) = fs.figure(W_PX, H_PX, ncols=2, sharey=True,
                               gridspec_kw={"width_ratios": [2.2, 1], "wspace": 0.06})
    series = (("squidly", "Squidly 3B (ensemble)", fs.BASELINE), ("flat", "Flat (this work)", fs.BRANCH["flat"]))
    for i, r in enumerate(rows):
        y = n - 1 - i
        for s_i, (key, _, color) in enumerate(series):
            yy = y + (BAR_H + GAP) / 2 * (1 if s_i == 0 else -1)
            fs.rounded_barh(ax, yy, r[key], BAR_H, color)
            fs.value_label(ax, r[key] + 0.012, yy, de(r[key]), ha="left", va="center")
        proven = r["ci"][0] > 0 or r["ci"][1] < 0
        colour = fs.BRANCH["flat"] if proven else fs.INK_SOFT
        fs.ci_errorbar_h(axd, y, tuple(r["ci"]), colour, point=r["delta"])
        fs.value_label(axd, r["ci"][1], y, "  " + signed(r["delta"]), ha="left", va="center")
    ax.set_yticks(range(n))
    one_set = len({r["set"] for r in rows}) == 1        # then the set goes into the panel title, not every row
    ax.set_yticklabels([r["metric"] if one_set else f"{r['set']} (n = {r['n']})\n{r['metric']}" for r in reversed(rows)])
    ax.set_xlim(0, d["xmax"])            # bar axes start at 0
    ax.set_ylim(-0.6, n - 0.4)
    fs.finish(ax, xlabel="Score (macro average)", grid_axis="x")
    fs.panel(ax, "A", f"{rows[0]['set']} (n = {rows[0]['n']}), same training data" if one_set
             else "Absolute scores, same training data")
    ax.legend(handles=[mpatches.Patch(facecolor=c, edgecolor="none", label=l) for _, l, c in series],
              loc="lower left", bbox_to_anchor=(0, 1.13, 1, 0.12), ncol=2, handlelength=1.2, columnspacing=1.4,
              borderaxespad=0.0)
    axd.set_xlim(*d["delta_xlim"])
    axd.set_xticks([0, 0.05, 0.1])        # no tick left of 0: it collided with the "1.0" of the left panel
    fs.finish(axd, xlabel="Δ (Flat − Squidly), 95% CI", grid_axis="x", zero_line=True)
    fs.panel(axd, "B", "Paired difference")
    fs.save(fig, OUT)


if __name__ == "__main__":
    main()
