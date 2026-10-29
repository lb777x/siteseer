"""What the frozen ESM-2 embedding already separates. Figure 3.21: UMAP of the same residues in four
representations (layer 33 raw, layer 30 standardised, hidden layer of the site-type and of the Flat head), non-site /
binding / catalytic. Figure 3.22: UMAP of mean-pooled protein embeddings by EC class. The kNN purity in the original
space (under each panel) carries the statement; UMAP only illustrates. Data: figures/data/embedding_map.json.

  python figures/plot_embedding_map.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "figures"))

import numpy as np

import figstyle as fs

DATA = os.path.join(ROOT, "figures", "data", "embedding_map.json")
# short panel titles; "ESM-2 650M" and "hidden layer of the trained MLP head" go into the caption
PANELS = [("layer33", "Layer 33 (raw)"), ("layer30", "Layer 30 (std.)"),
          ("hidden_sitetypes", "Site-type head"), ("hidden_final", "Flat head")]
RES_STYLE = {0: ("Non-site", fs.BASELINE_ALT, 6, 1), 2: ("Binding", fs.BRANCH["structural"], 16, 2),
             1: ("Catalytic", fs.BRANCH["sequence"], 26, 3)}
EC_STYLE = [(1, "EC 1 oxidoreductases", fs.BRANCH["sequence"]), (2, "EC 2 transferases", fs.BRANCH["adapter"]),
            (3, "EC 3 hydrolases", fs.BRANCH["structural"])]


def de(x, nd=2):
    return fs.num(x, nd)


def bare(ax):
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def residues(d):
    maps = d["residues"]["maps"]
    panels = [p for p in PANELS if p[0] in maps]
    fig, axes = fs.figure(1600, 560, ncols=len(panels))
    axes = np.atleast_1d(axes)
    for k_ax, (ax, (key, title)) in enumerate(zip(axes, panels)):
        m = maps[key]
        xy, y = np.asarray(m["xy"]), np.asarray(m["y"])
        for cls in (0, 2, 1):                      # Hintergrund zuerst, katalytisch obenauf
            name, color, size, z = RES_STYLE[cls]
            sel = y == cls
            ax.scatter(xy[sel, 0], xy[sel, 1], s=size, c=color, linewidths=0.6 if cls else 0,
                       edgecolors=fs.SURFACE if cls else "none", zorder=z, rasterized=True)
        bare(ax)
        p = m["purity"]
        fs.panel(ax, "ABCD"[k_ax], title)
        ax.set_xlabel(f"kNN purity {de(p['balanced'])}\ncatalytic {de(p['per_class']['1'])}",
                      color=fs.INK, fontsize=fs.FS_TICK, labelpad=8, linespacing=1.3)
    handles = [axes[0].scatter([], [], s=90, c=RES_STYLE[c][1], edgecolors=fs.SURFACE if c else "none", linewidths=0.6)
               for c in (1, 2, 0)]
    fig.legend(handles, [RES_STYLE[c][0] for c in (1, 2, 0)], loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.13), fontsize=fs.FS_TICK, labelcolor=fs.INK_SOFT, handletextpad=0.3)
    pur0 = maps[panels[0][0]]["purity"]
    fig.text(0.5, -0.10, f"kNN purity: share of the {d['residues']['k']} nearest neighbours (original space) with the "
             f"same class, class-balanced; chance {de(pur0['chance_balanced'])} (catalytic alone "
             f"{de(pur0['chance']['1'])})", ha="center", va="top", fontsize=fs.FS_TICK, color=fs.INK_FAINT)
    fig.subplots_adjust(wspace=0.08)
    return fs.save(fig, "embedding_map_residues.png")


def proteins(d):
    p = d["proteins"]
    xy, ec = np.asarray(p["xy"]), np.asarray(p["ec"])
    fig, ax = fs.figure(800, 560)
    other = ~np.isin(ec, [c for c, _, _ in EC_STYLE])
    ax.scatter(xy[other, 0], xy[other, 1], s=14, c=fs.BASELINE_ALT, linewidths=0, zorder=1, rasterized=True)
    for c, _, color in EC_STYLE:
        sel = ec == c
        ax.scatter(xy[sel, 0], xy[sel, 1], s=18, c=color, edgecolors=fs.SURFACE, linewidths=0.5, zorder=2, rasterized=True)
    # view: main cloud = median +- 2 IQR per axis; UMAP pushes small outlier groups far out and would squeeze the
    # main cloud. The note gives how many proteins fall outside (purity uses all proteins)
    med, iqr = np.median(xy, axis=0), np.subtract(*np.percentile(xy, [75, 25], axis=0))
    lo, hi = med - 2 * iqr, med + 2 * iqr
    pad = 0.03 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]), ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    outside = int((~((xy >= lo - pad) & (xy <= hi + pad)).all(1)).sum())
    bare(ax)
    handles = [ax.scatter([], [], s=90, c=color) for _, _, color in EC_STYLE] + [ax.scatter([], [], s=90, c=fs.BASELINE_ALT)]
    ax.legend(handles, [n for _, n, _ in EC_STYLE] + ["EC 4-7"], loc="upper left", bbox_to_anchor=(1.0, 1.0),
              frameon=False, fontsize=fs.FS_TICK, labelcolor=fs.INK_SOFT, handletextpad=0.3)
    pur = p["purity"]
    ax.set_xlabel(f"{p['n']} proteins, kNN purity {de(pur['balanced'])} (chance {de(pur['chance_balanced'])})"
                  + (f"\n{outside} outlying proteins outside the plotted area" if outside else ""),
                  color=fs.INK, fontsize=fs.FS_TICK, labelpad=8)
    return fs.save(fig, "embedding_map_ec.png")


def main():
    d = json.load(open(DATA))
    print(residues(d))
    print(proteins(d))


if __name__ == "__main__":
    main()
