"""One row per protein: DiffDock's rank-1 confidence, wall-clock seconds and success per condition.

  python -m siteseer.docking.report --out-dir results/docking/runs --tracker-csv results/docking/tracker_*.csv \\
      --conditions _global _predicted_cropped _predicted_cropped_weighted --output-csv results/docking/comparison.csv
"""
import argparse
import os
import re

import pandas as pd

CONFIDENCE = re.compile(r"rank1_confidence([+-]?\d+\.\d+)\.sdf")


def confidence(folder):
    if os.path.isdir(folder):
        for f in os.listdir(folder):
            m = CONFIDENCE.search(f)
            if m:
                return float(m.group(1))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tracker-csv", nargs="+", required=True)
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--output-csv", required=True)
    args = ap.parse_args()
    tr = pd.concat([pd.read_csv(t) for t in args.tracker_csv if os.path.exists(t)], ignore_index=True)
    seconds, success = dict(zip(tr["complex_name"], tr["seconds"])), dict(zip(tr["complex_name"], tr["success"]))
    rows = []
    for uid in sorted(tr["uid"].unique()):
        row = {"uid": uid}
        for c in args.conditions:
            row[f"{c}_confidence"] = confidence(os.path.join(args.out_dir, f"{uid}{c}"))
            row[f"{c}_seconds"] = seconds.get(f"{uid}{c}")
            row[f"{c}_success"] = success.get(f"{uid}{c}")
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"[+] {len(rows)} proteins -> {args.output_csv}")


if __name__ == "__main__":
    main()
