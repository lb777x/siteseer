"""Step 05: new train/valid/test split of the (extended) Goliath pool by 90%-identity clustering.

EasIFA's inherited split (80% CD-HIT) leaks between valid and test, and 102 ids carried two split flags at once.
Here every unique protein is clustered with `mmseqs easy-cluster --min-seq-id 0.9 -c 0.8 --cov-mode 0`, and whole
clusters are assigned to train/valid/test (target 94.3 / 4.1 / 1.6 %) by a seeded greedy rule: clusters sorted by
size (ties broken by a seeded shuffle), each to the split furthest below its target. Every row of a protein gets
the same flag; row order and count are unchanged, so row-keyed embedding caches stay valid.

--exclude-ids-file removes proteins from the clustering input entirely (used for the split without EasIFA's 853
test proteins, step 07); their rows keep their old flag here and are flagged explicitly by step 06.

  python pipeline/05_resplit.py                       # production split -> data/pipeline/resplit/
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import EASIFA_DIR, PIPELINE, run, tool

TARGET_PROPORTIONS = {"train": 0.943, "valid": 0.041, "test": 0.016}


def build_fasta(df: pd.DataFrame, path: str, exclude_ids: set = None):
    dedup = df.drop_duplicates(subset="id")
    if exclude_ids:
        n = len(dedup)
        dedup = dedup[~dedup["id"].isin(exclude_ids)]
        print(f"[*] excluded {n - len(dedup)} of {len(exclude_ids)} listed ids from the clustering input")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for _, row in dedup.iterrows():
            seq = str(row["aa_sequence"]).strip()
            if seq and seq != "nan":
                f.write(f">{row['id']}\n{seq}\n")
    print(f"[*] {len(dedup)} unique proteins -> {path}")


def assign_clusters(cluster_tsv: str, seed: int = 42):
    clusters = pd.read_csv(cluster_tsv, sep="\t", header=None, names=["rep", "member"])
    membership = clusters.groupby("rep")["member"].apply(list).to_dict()
    n = clusters["member"].nunique()
    targets = {"train": round(TARGET_PROPORTIONS["train"] * n), "valid": round(TARGET_PROPORTIONS["valid"] * n)}
    targets["test"] = n - targets["train"] - targets["valid"]
    rng = np.random.default_rng(seed)
    items = list(membership.items())
    rng.shuffle(items)
    items.sort(key=lambda kv: len(kv[1]), reverse=True)
    counts = {"train": 0, "valid": 0, "test": 0}
    split_of, cluster_of, size_of = {}, {}, {}
    for rep, members in items:
        deficits = {s: targets[s] - counts[s] for s in counts}
        split = max(deficits, key=deficits.get) if max(deficits.values()) > 0 else "train"
        counts[split] += len(members)
        for m in members:
            split_of[m], cluster_of[m], size_of[m] = split, rep, len(members)
    print(f"[*] {len(membership)} clusters, {n} proteins -> {counts} (targets {targets})")
    return split_of, cluster_of, size_of


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-csv", default=f"{EASIFA_DIR}/cached_dataframe_expanded.csv")
    ap.add_argument("--resplit-dir", default=f"{PIPELINE}/resplit")
    ap.add_argument("--out-csv", default=f"{EASIFA_DIR}/cached_dataframe_resplit90.csv")
    ap.add_argument("--exclude-ids-file", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    df = pd.read_csv(args.source_csv)
    df["id"] = df["alphafolddb-id"].astype(str).str.strip()
    old_flag = df.groupby("id")["dataset_flag"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else None)
    exclude = None
    if args.exclude_ids_file:
        exclude = {line.strip() for line in open(args.exclude_ids_file) if line.strip()}
    fasta = os.path.join(args.resplit_dir, "goliath_unique_proteins.fasta")
    prefix = os.path.join(args.resplit_dir, "mmseqs90", "goliath_cluster90")
    cluster_tsv = prefix + "_cluster.tsv"
    build_fasta(df, fasta, exclude)
    if not os.path.exists(cluster_tsv):
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        run([tool("mmseqs"), "easy-cluster", fasta, prefix, os.path.join(args.resplit_dir, "mmseqs_tmp"),
             "--min-seq-id", 0.9, "-c", 0.8, "--cov-mode", 0, "--threads", args.threads])
    split_of, cluster_of, size_of = assign_clusters(cluster_tsv, args.seed)
    mapping = pd.DataFrame({"alphafolddb-id": list(split_of), "cluster_rep_id": [cluster_of[i] for i in split_of],
                            "cluster_size": [size_of[i] for i in split_of],
                            "old_dataset_flag": [old_flag.get(i) for i in split_of],
                            "new_dataset_flag": list(split_of.values())})
    mapping.to_csv(os.path.join(args.resplit_dir, "goliath_cluster_assignment_90.csv"), index=False)
    df["dataset_flag"] = df["id"].map(split_of)
    unmapped = df["dataset_flag"].isna()
    if unmapped.any():
        print(f"[!] {unmapped.sum()} rows not clustered (excluded or no sequence): old flag kept")
        df.loc[unmapped, "dataset_flag"] = df.loc[unmapped, "id"].map(old_flag)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.drop(columns=["id"]).to_csv(args.out_csv, index=False)
    print(f"[+] {args.resplit_dir}/goliath_cluster_assignment_90.csv, {args.out_csv}")


if __name__ == "__main__":
    main()
