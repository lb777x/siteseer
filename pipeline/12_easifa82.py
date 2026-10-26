"""Step 12: EasIFA's M-CSA transfer-learning test set (82 proteins, all in our M-CSA test split): the id list and
EasIFA's predictions on it (ESM-bin and SaProt-bin) in our prediction format.

  python pipeline/12_easifa82.py
"""
import argparse
import os

import torch

from common import EXTERNAL
from siteseer.comparisons.easifa import first_rows, to_predictions

RESULTS = f"{EXTERNAL}/easifa_results/results"
VARIANTS = {"esm_bin": "transfer_learning_task_results.csv", "saprot_bin": "transfer_learning_task_SaProt_results.csv"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=RESULTS)
    ap.add_argument("--ids-out", default="dataset/easifa/easifa82_protein_ids.txt")
    ap.add_argument("--pred-dir", default="data/comparisons/easifa")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.ids_out), exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)
    ids = None
    for name, f in VARIANTS.items():
        rows = first_rows(os.path.join(args.results_dir, f))
        preds = to_predictions(rows)
        if ids is not None and sorted(preds["protein_ids"]) != ids:
            raise SystemExit(f"{f}: different proteins than {VARIANTS['esm_bin']}")
        ids = sorted(preds["protein_ids"])
        out = os.path.join(args.pred_dir, f"easifa82_{name}_preds.pt")
        torch.save(preds, out)
        print(f"[+] {out}: {len(preds['protein_ids'])} proteins, {int(sum(y.sum() for y in preds['labels']))} sites")
    with open(args.ids_out, "w") as fh:
        fh.write("\n".join(ids) + "\n")
    print(f"[+] {args.ids_out}: {len(ids)} proteins")


if __name__ == "__main__":
    main()
