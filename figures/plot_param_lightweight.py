"""Trainable parameters: the Flat head against EasIFA's two variants (log scale). Input requirements are named in
the caption, not on a second axis. Data: figures/data/param_lightweight.json.

  python figures/plot_param_lightweight.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import figstyle as fs

DATA = "figures/data/param_lightweight.json"
OUT = "param_lightweight.png"
W_PX, H_PX = 816, 300
BAR_H = 0.55
XTICKS = [1e5, 1e6, 1e7, 1e8, 1e9]


def color_for(key):
    return fs.BASELINE if key == "baseline" else fs.BRANCH[key]


def main():
    with open(DATA) as f:
        d = json.load(f)
    if d.get("_todo"):
        print(f"[i] offen: {d['_todo']}")
    fig, ax = fs.figure(W_PX, H_PX)

    bars = d["bars"]
    for i, bar in enumerate(bars):
        y = len(bars) - 1 - i          # first entry (Flat) on top
        fs.rounded_barh(ax, y, bar["value"], BAR_H, color_for(bar["color_key"]))
        # label just after the bar end; on a log axis a fixed factor is a constant visual gap
        fs.value_label(ax, bar["value"] * 1.12, y, bar["label"], ha="left", va="center", dy=0)

    ax.set_yticks(range(len(bars)))
    ax.set_yticklabels([b["name"] for b in reversed(bars)])
    ax.set_xscale("log")
    ax.set_xticks(XTICKS)
    ax.set_xticklabels([r"$10^{%d}$" % round(__import__("math").log10(t)) for t in XTICKS])
    ax.minorticks_off()
    # generous right margin so the value labels stay inside the figure
    ax.set_xlim(XTICKS[0], XTICKS[-1] * 6)
    ax.set_ylim(-0.6, len(bars) - 0.4)
    # the axis carries its own 10^n labels
    fs.finish(ax, xlabel=d["xlabel"], grid_axis="x")
    fs.save(fig, OUT)


if __name__ == "__main__":
    main()
