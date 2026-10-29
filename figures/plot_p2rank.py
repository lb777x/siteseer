"""Sequence-only prediction against P2Rank on the same proteins and AlphaFold models, residue-level macro AUPRC:
non-enzymes with BioLiP ligands (all, metal, organic), Goliath test enzymes, M-CSA (main model; otherwise the
site-type model). (A) absolute, (B) paired difference ours minus P2Rank with 95% CI: above zero teal, below zero
BASELINE, crossing zero INK_SOFT. Data: figures/data/p2rank_comparison.json.

  python figures/plot_p2rank.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.patches as mpatches

import figstyle as fs

DATA = "figures/data/p2rank_comparison.json"
OUT = "p2rank.png"
W_PX, H_PX = 1400, 640
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
    series = (("p2rank", "P2Rank (structure-based)", fs.BASELINE), ("ours", "Sequence-based (this work)", fs.BRANCH["flat"]))
    for i, r in enumerate(rows):
        y = n - 1 - i
        for s_i, (key, _, color) in enumerate(series):
            yy = y + (BAR_H + GAP) / 2 * (1 if s_i == 0 else -1)
            fs.rounded_barh(ax, yy, r[key], BAR_H, color)
            fs.value_label(ax, r[key] + 0.012, yy, de(r[key]), ha="left", va="center")
        colour = fs.BRANCH["flat"] if r["ci"][0] > 0 else fs.BASELINE if r["ci"][1] < 0 else fs.INK_SOFT
        fs.ci_errorbar_h(axd, y, tuple(r["ci"]), colour, point=r["delta"])
        if r["delta"] >= 0:
            fs.value_label(axd, r["ci"][1], y, "  " + signed(r["delta"]), ha="left", va="center")
        else:
            fs.value_label(axd, r["ci"][0], y, signed(r["delta"]) + "  ", ha="right", va="center")
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{r['label']}\n(n = {r['n']:,})" for r in reversed(rows)])
    ax.set_xlim(0, d["xmax"])            # bar axis starts at 0; room on the right for "0.925"
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(-0.6, n - 0.4)
    fs.finish(ax, xlabel="Macro AUPRC (per protein, residue level)", grid_axis="x")
    fs.panel(ax, "A", "Absolute scores")
    ax.legend(handles=[mpatches.Patch(facecolor=c, edgecolor="none", label=l) for _, l, c in series],
              loc="lower left", bbox_to_anchor=(0, 1.13, 1, 0.12), ncol=2, handlelength=1.2, columnspacing=1.4,
              borderaxespad=0.0)
    axd.set_xlim(*d["delta_xlim"])
    axd.set_xticks([0, 0.25, 0.5])        # no tick left of 0: it would collide with the left panel's last tick
    fs.finish(axd, xlabel="Δ (ours − P2Rank), 95% CI", grid_axis="x", zero_line=True)
    fs.panel(axd, "B", "Paired difference")
    fs.save(fig, OUT)


if __name__ == "__main__":
    main()
