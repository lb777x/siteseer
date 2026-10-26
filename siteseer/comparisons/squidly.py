"""Squidly's ensemble output (<stem>_squidly_ensemble.pkl from squidly/squidly.py) in our prediction format.

Columns all_AS_probs_1..5 hold the per-residue probabilities of the five ensemble models (residue j = sequence
position j + 1). Score = their mean (Squidly's ensemble mean), stored as a logit; labels from our CSV (element-wise),
and every sequence length is checked.

  python -m siteseer.comparisons.squidly --pkl test_squidly_ensemble.pkl \\
      --labels-csv dataset/catalodb_squidly/catalodb_squidly.csv --out data/comparisons/squidly/catalodb_preds.pt
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from siteseer.data.labels import parse_site_labels

ID = "alphafolddb-id"
MCSA = [f"dataset/mcsa_fine_tune/normal_mcsa/{s}_dataset/mcsa_{s}.csv" for s in ("train", "valid", "test")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels-csv", nargs="+", default=MCSA, help="CSVs with id, aa_sequence, site_labels")
    args = ap.parse_args()

    rows = {}
    for p in args.labels_csv:
        d = pd.read_csv(p)
        for a, s, sl in zip(d[ID].astype(str).str.strip(), d["aa_sequence"], d["site_labels"]):
            rows.setdefault(a, (s, sl))
    sq = pd.read_pickle(args.pkl)
    cols = sorted(c for c in sq.columns if str(c).startswith("all_AS_probs_"))
    if len(cols) != 5:
        raise SystemExit(f"expected 5 ensemble columns, found {cols}")
    out, bad = {"protein_ids": [], "logits": [], "labels": []}, []
    for pid, r in sq.iterrows():
        pid = str(pid).strip()
        if pid not in rows:
            bad.append((pid, "no labels"))
            continue
        seq, sl = rows[pid]
        probs = [np.asarray(r[c], dtype=np.float64).reshape(-1) for c in cols]
        if any(len(p) != len(seq) for p in probs):
            bad.append((pid, f"lengths {[len(p) for p in probs]} vs sequence {len(seq)}"))
            continue
        mean = np.clip(np.mean(probs, axis=0), 1e-6, 1 - 1e-6)
        y = np.zeros(len(seq), dtype=np.float32)
        for i in parse_site_labels(sl, "catalodb"):
            if 1 <= i <= len(seq):
                y[i - 1] = 1.0
        out["protein_ids"].append(pid)
        out["logits"].append(torch.tensor(np.log(mean / (1 - mean)), dtype=torch.float32))
        out["labels"].append(torch.from_numpy(y))
    if bad:
        print(f"[!] {len(bad)} proteins left out, e.g. {bad[:5]}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(out, args.out)
    print(f"[+] {len(out['protein_ids'])} proteins -> {args.out}")


if __name__ == "__main__":
    main()
