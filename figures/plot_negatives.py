"""Negative control: does the model flag proteins without an active site?
  3.18 neg_score_dist.png   protein score (maximum residue probability): confident non-enzymes (set G) against enzymes
  3.19 neg_rejection.png    share classified as enzyme per set, 95% Wilson CI: three enzyme test sets, four
                            non-enzyme sets (set B light grey: an alarm is intended there, binding sites are positive
                            in training)
  3.20 neg_categories.png   set G by functional category
Threshold: Youden on the validation part of G and M-CSA valid; reported on the rest.
Data: figures/data/negatives_control.json (siteseer.negatives.analyze).

  python figures/plot_negatives.py
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import figstyle as fs

ENZ = ["reallabels_test", "goliath_test", "mcsa_test"]
KIND = ["enz", "enz", "enz", "neg", "neg", "neg", "b"]


def whisker(ax, y, lo, hi):
    """95% CI as a plain line: the value sits at the bar end, and a Wilson interval's centre is not the estimate."""
    ax.errorbar((lo + hi) / 2, y, xerr=[[(hi - lo) / 2], [(hi - lo) / 2]], fmt="none", ecolor=fs.INK_SOFT,
                elinewidth=1.8, capsize=4, capthick=1.8, zorder=4)


def de(x, nd=1):
    return fs.num(x, nd)


def score_dist(d, out):
    t, h, sd = d["threshold"], d["score_hist"], d["score_dist"]
    bins, g, e = np.array(h["bins"]), np.array(h["G_pct"]), np.array(h["enzymes_pct"])
    fig, ax = fs.figure(1600, 560)
    ax.hist(bins[:-1], bins=bins, weights=g, color=fs.BASELINE, alpha=0.9, zorder=2)
    ax.hist(bins[:-1], bins=bins, weights=e, histtype="step", linewidth=2.8, color=fs.BRANCH["flat"], zorder=3)
    ax.axvline(t, color=fs.INK_SOFT, linewidth=1.5, linestyle=(0, (4, 3)), zorder=4)
    top = max(g.max(), e.max())
    ax.set_ylim(0, top * 1.32)
    ax.set_xlim(0, 1)
    kept = [sd["enzymes_above_pct"][k] for k in ENZ]
    ax.annotate(f"Threshold {de(t, 2)}", (t, top * 1.27), xytext=(0, 0), textcoords="offset points",
                ha="center", va="top", fontsize=fs.FS_TICK, color=fs.INK_SOFT, zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", fc=fs.SURFACE, ec="none"))
    ax.annotate(f"{de(sd['G_below_pct'], 0)}% below", (t, top * 1.04), xytext=(-12, 0), textcoords="offset points",
                ha="right", va="center", fontsize=fs.FS_TICK, color=fs.INK)
    ax.annotate(f"{de(min(kept), 0)}-{de(max(kept), 0)}% above", (t, top * 1.04), xytext=(12, 0),
                textcoords="offset points", ha="left", va="center", fontsize=fs.FS_TICK, color=fs.INK)
    fs.direct_label(ax, 0.06, top * 0.86, f"Non-enzymes (G, n = {sd['G_n']:,})", fs.BASELINE, dx=12)
    fs.direct_label(ax, 0.06, top * 0.68, f"Enzymes (3 test sets, n = {sd['enzymes_n']:,})", fs.BRANCH["flat"], dx=12)
    fs.finish(ax, xlabel="Protein score (maximum residue probability)", ylabel="Proteins (%)")
    fs.save(fig, out)


def rejection(d, out):
    t = d["threshold"]
    colors = {"enz": fs.BRANCH["flat"], "neg": fs.BASELINE, "b": fs.BASELINE_ALT}
    fig, ax = fs.figure(1600, 640)
    ys = [7.6, 6.6, 5.6, 3.6, 2.6, 1.6, 0.6]                 # gap between enzymes and non-enzymes
    for y, r, kind in zip(ys, d["rejection"], KIND):
        lo, hi = r["ci_pct"]
        fs.rounded_barh(ax, y, r["rate_pct"], 0.62, colors[kind])
        whisker(ax, y, lo, hi)
        ax.annotate(f"{de(r['rate_pct'])}%", (hi, y), xytext=(10, 0), textcoords="offset points", ha="left",
                    va="center", fontsize=fs.FS_TICK, color=fs.INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['label']} (n = {r['n']:,})" for r in d["rejection"]])
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_ylim(0, 8.2)
    fs.finish(ax, xlabel=f"Classified as enzyme (protein score ≥ {de(t, 2)}), %", grid_axis="x")
    ax.tick_params(axis="y", length=0)
    fs.save(fig, out)


def categories(d, out):
    t, stats = d["threshold"], d["categories"]
    fig, ax = fs.figure(1600, 600)
    for y, r in enumerate(stats):
        lo, hi = r["ci_pct"]
        fs.rounded_barh(ax, y, r["rate_pct"], 0.62, fs.BASELINE)
        whisker(ax, y, lo, hi)
        ax.annotate(f"{de(r['rate_pct'])}%", (hi, y), xytext=(10, 0), textcoords="offset points", ha="left",
                    va="center", fontsize=fs.FS_TICK, color=fs.INK)
    # overall rate as a note top right instead of a line: a line would cross the value labels
    ax.annotate(f"G overall: {de(d['G_report_rate_pct'])}%", (1, 1), xycoords="axes fraction", xytext=(0, 10),
                textcoords="offset points", ha="right", va="bottom", fontsize=fs.FS_TICK, color=fs.INK_SOFT)
    ax.set_yticks(range(len(stats)))
    ax.set_yticklabels([f"{r['category']} (n = {r['n']})" for r in stats])
    ax.set_xlim(0, 30)
    ax.set_ylim(-0.6, len(stats) - 0.1)
    fs.finish(ax, xlabel=f"Confident non-enzymes classified as enzyme (protein score ≥ {de(t, 2)}), %", grid_axis="x")
    ax.tick_params(axis="y", length=0)
    fs.better(ax, "lower")
    fs.save(fig, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="figures/data/negatives_control.json")
    d = json.load(open(ap.parse_args().data))
    score_dist(d, "neg_score_dist.png")
    rejection(d, "neg_rejection.png")
    categories(d, "neg_categories.png")


if __name__ == "__main__":
    main()
