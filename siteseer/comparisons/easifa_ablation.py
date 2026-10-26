"""EasIFA's published ablation on its Swiss-Prot E-RXN ASA test set (892 protein-reaction pairs), per identity band.

Gain of a component = macro F1 of the full model minus macro F1 without it, from EasIFA's own per-pair F1
(f1_scores) in EasIFA_results.zip: full = active-site-position-prediction, no reaction = ablation-experiment-1,
no 3D structure (GearNet) = ablation-experiment-3. 95% CI from 2,000 paired bootstrap draws over pairs (seed 0).
Bands are EasIFA's similarity_index_level (identity to EasIFA's training set).

  python -m siteseer.comparisons.easifa_ablation --results-dir data/external/easifa_results --out-json out.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

FULL = "active-site-position-prediction_results.csv"
COMPONENTS = {"structure": "ablation-experiment-3_results.csv", "reaction": "ablation-experiment-1_results.csv"}
KEY = ["alphafolddb-id", "reaction"]


def f1_table(path):
    return pd.read_csv(path, usecols=KEY + ["similarity_index_level", "f1_scores"])


def gain(delta, rng, n_bootstrap):
    boot = [delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(n_bootstrap)]
    return {"gain": float(delta.mean()), "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
            "n": int(len(delta))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True, help="the extracted results/ folder of EasIFA_results.zip")
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    full = f1_table(os.path.join(args.results_dir, FULL))
    bands = sorted(full["similarity_index_level"].unique())
    rng = np.random.default_rng(args.seed)   # one generator, drawn in the order below (as for the reported figure)
    out = {}
    for name, f in COMPONENTS.items():
        abl = f1_table(os.path.join(args.results_dir, f))
        if not abl[KEY].equals(full[KEY]):   # rows are paired by position; some pairs occur twice
            raise SystemExit(f"{f}: not the same pairs in the same order as {FULL}")
        delta = (full["f1_scores"] - abl["f1_scores"]).to_numpy()
        res = {band: gain(delta[(full["similarity_index_level"] == band).to_numpy()], rng, args.n_bootstrap)
               for band in bands}
        res["all"] = gain(delta, rng, args.n_bootstrap)
        out[name] = res
        for band, r in res.items():
            print(f"{name:9s} {band:7s} n={r['n']:4d}  {r['gain']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]")
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
