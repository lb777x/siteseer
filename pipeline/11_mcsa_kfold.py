"""Step 11: homology-aware 5-fold cross-validation over the whole M-CSA pool (train + valid + test, 962 proteins).

The pool is clustered at 90% identity (`mmseqs easy-cluster --min-seq-id 0.9 -c 0.8 --cov-mode 0`) and whole
clusters are assigned to five folds by the same seeded greedy rule as step 05 (largest cluster first, to the fold
furthest below one fifth). Fold i: test = fold i, valid = fold i+1, train = the other three. Every protein is
tested exactly once, out of fold.

  python pipeline/11_mcsa_kfold.py         # -> dataset/mcsa_fine_tune/kfold/fold_{0..4}_{train,valid,test}.csv
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import MCSA_CSV, run, tool

N_FOLDS = 5


def assign_folds(cluster_tsv: str, seed: int = 42):
    clusters = pd.read_csv(cluster_tsv, sep="\t", header=None, names=["rep", "member"])
    membership = clusters.groupby("rep")["member"].apply(list).to_dict()
    folds = [f"fold_{i}" for i in range(N_FOLDS)]
    target = clusters["member"].nunique() / N_FOLDS
    rng = np.random.default_rng(seed)
    items = list(membership.items())
    rng.shuffle(items)
    items.sort(key=lambda kv: len(kv[1]), reverse=True)
    counts = {f: 0 for f in folds}
    fold_of, cluster_of, size_of = {}, {}, {}
    for rep, members in items:
        fold = max(folds, key=lambda f: target - counts[f])
        counts[fold] += len(members)
        for m in members:
            fold_of[m], cluster_of[m], size_of[m] = fold, rep, len(members)
    print(f"[*] {len(membership)} clusters -> fold sizes {counts}")
    return fold_of, cluster_of, size_of


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mcsa-csv", nargs=3, default=[MCSA_CSV[s] for s in ("train", "valid", "test")])
    ap.add_argument("--out-dir", default="dataset/mcsa_fine_tune/kfold")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(p) for p in args.mcsa_csv], ignore_index=True)
    df["id"] = df["alphafolddb-id"].astype(str).str.strip()
    df = df.drop_duplicates(subset="id")
    os.makedirs(args.out_dir, exist_ok=True)
    fasta = os.path.join(args.out_dir, "mcsa_unique_proteins.fasta")
    with open(fasta, "w") as f:
        for _, row in df.iterrows():
            seq = str(row["aa_sequence"]).strip()
            if seq and seq != "nan":
                f.write(f">{row['id']}\n{seq}\n")
    prefix = os.path.join(args.out_dir, "mmseqs90", "mcsa_cluster90")
    if not os.path.exists(prefix + "_cluster.tsv"):
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        run([tool("mmseqs"), "easy-cluster", fasta, prefix, os.path.join(args.out_dir, "mmseqs_tmp"),
             "--min-seq-id", 0.9, "-c", 0.8, "--cov-mode", 0, "--threads", args.threads])
    fold_of, cluster_of, size_of = assign_folds(prefix + "_cluster.tsv", args.seed)
    if (~df["id"].isin(fold_of)).any():
        raise RuntimeError("every M-CSA protein must land in a fold")
    pd.DataFrame({"alphafolddb-id": list(fold_of), "cluster_rep_id": [cluster_of[i] for i in fold_of],
                  "cluster_size": [size_of[i] for i in fold_of], "fold": list(fold_of.values())}).to_csv(
        os.path.join(args.out_dir, "mcsa_cluster_assignment_90.csv"), index=False)
    df["fold"] = df["id"].map(fold_of)
    cols = [c for c in df.columns if c not in ("id", "fold")]
    for i in range(N_FOLDS):
        test, valid = f"fold_{i}", f"fold_{(i + 1) % N_FOLDS}"
        parts = {"test": df[df["fold"] == test], "valid": df[df["fold"] == valid],
                 "train": df[~df["fold"].isin([test, valid])]}
        for split, part in parts.items():
            part[cols].to_csv(os.path.join(args.out_dir, f"fold_{i}_{split}.csv"), index=False)
        print(f"[+] fold {i}: {len(parts['train'])} train / {len(parts['valid'])} valid / {len(parts['test'])} test")


if __name__ == "__main__":
    main()
