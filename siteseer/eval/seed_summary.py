"""Spread of the main model's metrics over training seeds (threshold 0.5).

Per run and prediction set: macro AUPRC, F1, MCC (per protein, averaged) and pooled AUPRC; then mean, SD (n - 1) and
range over runs. A set named kfold_test / kfold_valid pools the five folds' files.

  python -m siteseer.eval.seed_summary --runs main main_seed43 main_seed44 main_seed45 main_seed46 \\
      --sets kfold_test kfold_valid stage2_mcsa_test stage2_easifa82_test --out-json seeds.json
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from siteseer.eval.metrics import binary_metrics


def files_for(pred_dir: str, run: str, name: str) -> list:
    if name.startswith("kfold_"):
        split = name.split("_", 1)[1]
        return [os.path.join(pred_dir, f"{run}_kfold{i}_{split}.pt") for i in range(5)]
    return [os.path.join(pred_dir, f"{run}_{name}.pt")]


def summarise(paths: list) -> dict:
    logits, labels = [], []
    for p in paths:
        d = torch.load(p, weights_only=False)
        logits.extend(d["logits"])
        labels.extend(d["labels"])
    per = [binary_metrics(lo, la) for lo, la in zip(logits, labels)]
    out = {f"macro_{k}": float(np.mean([m[k] for m in per])) for k in ("auprc", "f1", "mcc")}
    out["pooled_auprc"] = float(average_precision_score(torch.cat(labels).numpy(), torch.cat(logits).numpy()))
    out["n_proteins"] = len(per)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="prediction-file prefixes, one per seed")
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--pred-dir", default="results/predictions")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    out = {}
    for name in args.sets:
        per_run = {run: summarise(files_for(args.pred_dir, run, name)) for run in args.runs}
        stats = {}
        for k in ("macro_auprc", "macro_f1", "macro_mcc", "pooled_auprc"):
            v = np.array([per_run[r][k] for r in args.runs])
            stats[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max())}
        out[name] = {"runs": per_run, "summary": stats}
        print(f"--- {name} ({per_run[args.runs[0]]['n_proteins']} proteins, {len(args.runs)} runs) ---")
        for k, s in stats.items():
            print(f"  {k:13s} {s['mean']:.3f} ± {s['sd']:.3f}  ({s['min']:.3f} to {s['max']:.3f})")
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[+] {args.out_json}")


if __name__ == "__main__":
    main()
