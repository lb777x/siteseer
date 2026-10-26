"""Linear screen of two per-residue embeddings: A alone, B alone, and both concatenated.

Logistic regression (C = 0.1, standardised features) fitted on residues of --n-train train proteins (all positives
plus --neg-per-pos negatives each) and scored on every residue of --n-test held-out proteins. Reported: macro and
pooled AUROC and macro AUPRC per feature set. Absolute values sit below the trained probe; the screen compares
feature sets, not models. Both cache directories are keyed by _original_row_index.

  python -m siteseer.eval.linear_screen --csv CSV --dir-a data/esm2_layers/layer_33 --label-a "layer 33" \\
      --dir-b data/esm2_layers/layer_30 --label-b "layer 30" --n-train 2000 --n-test 400 --out-json out.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from siteseer.data.labels import parse_site_labels, row_label_source

ID = "alphafolddb-id"


def load_pair(row_index, dir_a, dir_b):
    """Both embeddings of one row, cut to their common length; None if a file is missing or the lengths differ by
    more than 10%."""
    pa, pb = (os.path.join(d, f"row_{row_index}.pt") for d in (dir_a, dir_b))
    if not (os.path.exists(pa) and os.path.exists(pb)):
        return None
    try:
        a = torch.load(pa, map_location="cpu", weights_only=False)["embedding"].float().numpy()
        b = torch.load(pb, map_location="cpu", weights_only=False)["embedding"].float().numpy()
    except Exception:
        return None
    n = min(len(a), len(b))
    if n == 0 or abs(len(a) - len(b)) > 0.1 * max(len(a), len(b)):
        return None
    return a[:n], b[:n]


def collect(df, splits, n_proteins, dir_a, dir_b, neg_per_pos, rng, keep_all):
    rows = df[df["dataset_flag"].isin(splits)].drop_duplicates(ID)
    rows = rows.iloc[rng.permutation(len(rows))]
    Xa, Xb, y, groups, used = [], [], [], [], 0
    for _, row in rows.iterrows():
        if used >= n_proteins:
            break
        pair = load_pair(int(row["_original_row_index"]), dir_a, dir_b)
        if pair is None:
            continue
        a, b = pair
        labels = parse_site_labels(row["site_labels"], row_label_source(row))
        if not labels:
            continue
        yy = np.zeros(len(a), dtype=np.int8)
        for idx in labels:
            if 1 <= idx <= len(a):
                yy[idx - 1] = 1
        if yy.sum() == 0:
            continue
        if keep_all:
            sel = np.arange(len(a))
        else:
            pos, neg = np.flatnonzero(yy), np.flatnonzero(yy == 0)
            sel = np.concatenate([pos, rng.choice(neg, min(len(neg), neg_per_pos * len(pos)), replace=False)])
        Xa.append(a[sel]); Xb.append(b[sel]); y.append(yy[sel])
        groups.append(np.full(len(sel), used))
        used += 1
    if not Xa:
        raise RuntimeError(f"no usable proteins in {splits}")
    print(f"[*] {'+'.join(splits)}: {used} proteins, {sum(len(v) for v in y)} residues, "
          f"{int(sum(v.sum() for v in y))} positive")
    return np.concatenate(Xa), np.concatenate(Xb), np.concatenate(y), np.concatenate(groups)


def score(name, Xtr, ytr, Xte, yte, gte, out, chunk=20000):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=300, C=0.1).fit(sc.transform(Xtr), ytr)
    p = np.empty(len(yte), dtype=np.float64)
    for i in range(0, len(yte), chunk):   # chunked: the concatenated test matrix is large
        p[i:i + chunk] = clf.decision_function(sc.transform(Xte[i:i + chunk]))
    groups = np.unique(gte)
    res = {"dim": int(Xtr.shape[1]),
           "macro_auroc": float(np.mean([roc_auc_score(yte[gte == g], p[gte == g]) for g in groups
                                         if 0 < yte[gte == g].sum() < (gte == g).sum()])),
           "pooled_auroc": float(roc_auc_score(yte, p)),
           "macro_auprc": float(np.mean([average_precision_score(yte[gte == g], p[gte == g]) for g in groups
                                         if yte[gte == g].sum()]))}
    out[name] = res
    print(f"{name:26s} dim {res['dim']:5d}   macro AUROC {res['macro_auroc']:.4f}   "
          f"pooled AUROC {res['pooled_auroc']:.4f}   macro AUPRC {res['macro_auprc']:.4f}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--dir-a", required=True)
    ap.add_argument("--dir-b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--test-splits", nargs="+", default=["test"])
    ap.add_argument("--neg-per-pos", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    rng = np.random.default_rng(args.seed)
    Atr, Btr, ytr, _ = collect(df, ["train"], args.n_train, args.dir_a, args.dir_b, args.neg_per_pos, rng, False)
    Ate, Bte, yte, gte = collect(df, args.test_splits, args.n_test, args.dir_a, args.dir_b, args.neg_per_pos, rng, True)
    out = {"n_train_residues": int(len(ytr)), "n_test_residues": int(len(yte)),
           "n_test_proteins": int(len(np.unique(gte)))}
    ra = score(args.label_a, Atr, ytr, Ate, yte, gte, out, args.chunk)
    score(args.label_b, Btr, ytr, Bte, yte, gte, out, args.chunk)
    Ctr = np.empty((len(ytr), Atr.shape[1] + Btr.shape[1]), dtype=np.float32)
    Ctr[:, :Atr.shape[1]], Ctr[:, Atr.shape[1]:] = Atr, Btr
    Cte = np.empty((len(yte), Ate.shape[1] + Bte.shape[1]), dtype=np.float32)
    Cte[:, :Ate.shape[1]], Cte[:, Ate.shape[1]:] = Ate, Bte
    del Atr, Btr, Ate, Bte
    rb = score("both concatenated", Ctr, ytr, Cte, yte, gte, out, args.chunk)
    out["delta_macro_auroc_both_minus_a"] = rb["macro_auroc"] - ra["macro_auroc"]
    print(f"\nadding {args.label_b} to {args.label_a}: {out['delta_macro_auroc_both_minus_a']:+.4f} macro AUROC")
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[+] {args.out_json}")


if __name__ == "__main__":
    main()
