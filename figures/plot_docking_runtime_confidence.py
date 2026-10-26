"""Docking with predicted crops: DiffDock confidence and runtime against docking the whole structure.

Two units, two panels: (A) difference in DiffDock's confidence, (B) runtime saving in %, per crop condition, each a
point with its paired 95% bootstrap CI over proteins; the zero line is docking the whole structure. Every condition
ran on the same node per protein, so the runtime comparison is hardware-matched.

Data: figures/data/docking_paired.json, written by compass.docking.paired_bootstrap from the comparison table.

  python figures/plot_docking_runtime_confidence.py
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt

import figstyle as fs

W_PX, H_PX = 1600, 380
ROWS = [("_predicted_cropped", "Cropped"), ("_predicted_cropped_weighted", "Weighted cropped")]
PANELS = [("delta_conf", "delta_conf_ci", "Δ confidence vs. whole-structure docking", 3, "", "Docking confidence"),
          ("time_saving_pct", "time_saving_ci", "Time saving vs. whole-structure docking (%)", 1, "%", "Runtime")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="figures/data/docking_paired.json")
    ap.add_argument("--out", default="docking_runtime_confidence.png")
    args = ap.parse_args()
    d = json.load(open(args.data))
    color = fs.BRANCH["flat"]
    fig, axes = plt.subplots(1, 2, figsize=(W_PX / 100, H_PX / 100), dpi=fs.DPI, gridspec_kw={"wspace": 0.25})
    ys = list(range(len(ROWS)))[::-1]
    for ax, letter, (key, cikey, xlabel, nd, unit, title) in zip(axes, "AB", PANELS):
        vals = [0.0]
        for y, (cond, _) in zip(ys, ROWS):
            r = d["conditions"][cond]
            fs.ci_errorbar_h(ax, y, tuple(r[cikey]), color, point=r[key])
            ax.annotate(fs.num(r[key], nd, signed=True) + unit, (max(r[cikey][1], 0), y), xytext=(12, 0),
                        textcoords="offset points", ha="left", va="center", fontsize=fs.FS_TICK, color=fs.INK)
            vals += list(r[cikey])
        lo, hi = min(vals), max(vals)
        ax.set_xlim(lo - 0.15 * (hi - lo), hi + 0.45 * (hi - lo))
        ax.set_yticks(ys)
        ax.set_yticklabels([lab for _, lab in ROWS] if ax is axes[0] else [])
        ax.set_ylim(-0.6, len(ROWS) - 0.4)
        fs.finish(ax, xlabel=xlabel, grid_axis="x", zero_line=True)
        fs.panel(ax, letter, title)
        ax.tick_params(axis="y", length=0)
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()
