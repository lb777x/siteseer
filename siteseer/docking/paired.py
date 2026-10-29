"""Docking on predicted crops against the full structure, paired per protein (figure 3.16): proteins with a pose
in every condition; per crop condition the mean confidence difference and the mean relative runtime saving
(global - crop) / global, each with a 95% percentile bootstrap CI over proteins (1,000 draws, seed 42), and the
share of proteins with higher confidence. All conditions of a protein ran on the same node.

  python -m siteseer.docking.paired --csv results/docking/comparison.csv --out-json results/docking/paired.json
"""
import argparse
import json

import numpy as np
import pandas as pd

CONDITIONS = ["_predicted_cropped", "_predicted_cropped_weighted"]
BASE = "_global"


def boot_ci(values, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    m = df.dropna(subset=[f"{c}_confidence" for c in [BASE] + CONDITIONS]).copy()
    out = {"n_total": int(len(df)), "n_paired": int(len(m)), "conditions": {}}
    for c in CONDITIONS:
        d_conf = m[f"{c}_confidence"] - m[f"{BASE}_confidence"]
        base_s = pd.to_numeric(m[f"{BASE}_seconds"], errors="coerce")
        cond_s = pd.to_numeric(m[f"{c}_seconds"], errors="coerce")
        ok = base_s.notna() & cond_s.notna()
        saving = (base_s[ok] - cond_s[ok]) / base_s[ok] * 100
        out["conditions"][c] = {"delta_conf": float(d_conf.mean()), "delta_conf_ci": boot_ci(d_conf, args.n_bootstrap),
                                "frac_better": float((d_conf > 0).mean()), "time_saving_pct": float(saving.mean()),
                                "time_saving_ci": boot_ci(saving, args.n_bootstrap),
                                "mean_seconds_base": float(base_s[ok].mean()), "mean_seconds": float(cond_s[ok].mean())}
        r = out["conditions"][c]
        print(f"{c}: confidence {r['delta_conf']:+.3f} [{r['delta_conf_ci'][0]:+.3f}, {r['delta_conf_ci'][1]:+.3f}], "
              f"runtime saving {r['time_saving_pct']:.1f}% [{r['time_saving_ci'][0]:.1f}, {r['time_saving_ci'][1]:.1f}]")
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
