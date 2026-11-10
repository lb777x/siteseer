"""Macro AUPRC of the main model on M-CSA (5-fold, out of fold) by sequence identity of each protein to the stage-1
training set (A) and to its own M-CSA training fold (B); bars with 95% bootstrap CI over proteins, dashed line at the
value over all proteins. Data: figures/data/identity_bands.json.

  python figures/plot_identity_bands.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import figstyle as fs

DATA = "figures/data/identity_bands.json"
OUT = "identity_bands.png"
W_PX, H_PX = 1600, 600
WIDTH = 0.62


def draw(ax, bands, xlabel, overall):
    ax.axhline(overall, color=fs.BASELINE, linewidth=1.5, linestyle=(0, (4, 3)), zorder=1)
    for i, b in enumerate(bands):
        fs.rounded_bar(ax, i, b["macro_auprc"], WIDTH, fs.BRANCH["flat"])
        fs.ci_errorbar_v(ax, i, b["ci"], fs.INK)
        fs.value_label(ax, i, b["ci"][1], fs.num(b["macro_auprc"], 3), dy=6)
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([f"{b['band']}\nn = {b['n']:,}" for b in bands])
    ax.set_xlim(-0.6, len(bands) - 0.4)
    ax.tick_params(axis="x", length=0)
    fs.finish(ax, xlabel=xlabel)


def main():
    d = json.load(open(DATA))
    fig, (a, b) = fs.figure(W_PX, H_PX, ncols=2, sharey=True,
                            gridspec_kw={"width_ratios": [4, 2.3], "wspace": 0.12})
    draw(a, d["pool"]["bands"], "Identity to the stage-1/1.5 training set", d["macro_auprc_all"])
    draw(b, d["fold"]["bands"], "Identity to its M-CSA training fold", d["macro_auprc_all"])
    a.annotate(f"all proteins {fs.num(d['macro_auprc_all'], 3)}", (0.01, d["macro_auprc_all"]),
               xycoords=("axes fraction", "data"), xytext=(0, 6), textcoords="offset points", ha="left", va="bottom",
               fontsize=fs.FS_TICK, color=fs.INK_SOFT)
    a.set_ylim(0, 1.0)
    a.set_ylabel("Macro AUPRC, M-CSA 5-fold (95% CI)", labelpad=12)
    fs.panel(a, "A", "Large training set")
    fs.panel(b, "B", "M-CSA fine-tuning set")
    fs.save(fig, OUT)


if __name__ == "__main__":
    main()
