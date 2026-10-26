"""Share of rank-1 DiffDock poses with a heavy ligand atom within 4 A of an annotated site residue: DiffDock alone,
the two crops, SiteSeer re-ranking of the 10 whole-structure poses, and the best of 10 as upper bound. Differences
with a paired bootstrap CI (2,000 draws). Data: figures/data/docking_site_hits.json (rebuilt with --csv).

  python figures/plot_docking_site_hits.py [--csv site_accuracy.csv]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import figstyle as fs

W_PX, H_PX = 1600, 460
ROWS = [("_global_hit", "DiffDock alone (rank 1)", "base"),
        ("_predicted_cropped_hit", "Cropped", "siteseer"),
        ("_predicted_cropped_weighted_hit", "Weighted cropped", "siteseer"),
        ("_global_rerank_hit", "SiteSeer re-ranking", "siteseer"),
        ("_global_oracle_hit", "Upper bound: best of 10 poses", "ref")]


def summarise(csv_path: str) -> dict:
    m = pd.read_csv(csv_path).dropna(subset=[r[0] for r in ROWS[:4]])
    n = len(m)
    base = m["_global_hit"].astype(float).to_numpy()
    idx = np.random.default_rng(42).integers(0, n, size=(2000, n))
    out = {"n_paired": n, "contact_A": 4.0, "rows": []}
    for col, label, kind in ROWS:
        v = m[col].astype(float).to_numpy()
        row = {"condition": col, "label": label, "kind": kind, "rate": float(v.mean() * 100)}
        if kind == "siteseer":
            d = (v[idx] - base[idx]).mean(1) * 100
            lo, hi = np.percentile(d, [2.5, 97.5])
            row.update({"delta_pp": float((v - base).mean() * 100), "ci_pp": [float(lo), float(hi)]})
        out["rows"].append(row)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default=None, help="rebuild the data file from this site-accuracy table")
    ap.add_argument("--data", default="figures/data/docking_site_hits.json")
    ap.add_argument("--out", default="docking_site_hits.png")
    args = ap.parse_args()
    if args.csv:
        json.dump(summarise(args.csv), open(args.data, "w"), indent=2)
        print(f"[+] {args.data}")
    d = json.load(open(args.data))
    colors = {"base": fs.BASELINE, "siteseer": fs.BRANCH["flat"], "ref": fs.BASELINE_ALT}
    fig, ax = fs.figure(W_PX, H_PX)
    ys = list(range(len(d["rows"])))[::-1]
    for y, row in zip(ys, d["rows"]):
        fs.rounded_barh(ax, y, row["rate"], 0.62, colors[row["kind"]])
        text = f"{fs.num(row['rate'], 1)}%"
        if "delta_pp" in row:
            lo, hi = row["ci_pp"]
            text += (f"   {fs.num(row['delta_pp'], 1, signed=True)} pp [{fs.num(lo, 1, signed=True)}, "
                     f"{fs.num(hi, 1, signed=True)}]" + ("" if lo > 0 or hi < 0 else ", n.s."))
        ax.annotate(text, (row["rate"], y), xytext=(10, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=fs.FS_TICK, color=fs.INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([row["label"] for row in d["rows"]])
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_ylim(-0.6, len(ys) - 0.4)
    fs.finish(ax, xlabel=f"Rank-1 pose at the annotated site (%), n = {d['n_paired']} paired proteins", grid_axis="x")
    ax.tick_params(axis="y", length=0)
    fs.save(fig, args.out)


if __name__ == "__main__":
    main()
