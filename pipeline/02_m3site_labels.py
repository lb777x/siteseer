"""Step 02: M3Site residue labels for Goliath training proteins that step 01 did not upgrade.

M3Site (Gift-OYS/M3Site, data/label.json, commit 42b730a) maps a UniProt accession to one class per residue
(0 = no site, 1-6 = active-site subtypes). A Goliath training row with label_source 'goliath_original' whose protein
is in M3Site and whose sequence length equals the length of M3Site's array (proxy for the same sequence version)
gets one singleton group per labelled residue, label_source 'm3site'. M-CSA-derived rows are never overwritten.

  python pipeline/02_m3site_labels.py
"""
import argparse
import json
import os

import pandas as pd

from common import EXTERNAL, LABELS, PIPELINE


def dense_to_groups(class_array: list) -> tuple:
    """[0, 0, 2, 0, 1] -> ([[3], [5]], [2, 1]) with 1-based residue positions."""
    labels, types = [], []
    for i, cls in enumerate(class_array, start=1):
        if cls != 0.0:
            labels.append([i])
            types.append(int(cls))
    return labels, types


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-csv", default=f"{LABELS}/cached_dataframe_mcsa_labels.csv")
    ap.add_argument("--m3site-labels", default=f"{EXTERNAL}/m3site/label.json")
    ap.add_argument("--out-csv", default=f"{LABELS}/cached_dataframe_mcsa_m3site_labels.csv")
    ap.add_argument("--out-ids", default=f"{PIPELINE}/m3site_step_upgraded_goliath_ids.txt")
    args = ap.parse_args()

    m3site = json.load(open(args.m3site_labels))
    big = pd.read_csv(args.in_csv)
    big["id"] = big["alphafolddb-id"].astype(str).str.strip()
    train = big["dataset_flag"] == "train"
    eligible = train & (big["label_source"] == "goliath_original") & big["id"].isin(m3site)
    n_up = n_len = n_zero = 0
    for idx in big.index[eligible]:
        seq, classes = str(big.at[idx, "aa_sequence"]).strip(), m3site[big.at[idx, "id"]]
        if len(classes) != len(seq):
            n_len += 1
            continue
        labels, types = dense_to_groups(classes)
        if not labels:
            n_zero += 1
            continue
        big.at[idx, "site_labels"] = str(labels)
        big.at[idx, "site_types"] = str(types)
        big.at[idx, "label_source"] = "m3site"
        big.at[idx, "label_confidence"] = pd.NA
        n_up += 1
    print(f"[*] {eligible.sum()} eligible rows: {n_up} upgraded, {n_len} length mismatch, {n_zero} without a site")
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    big.drop(columns=["id"]).to_csv(args.out_csv, index=False)
    upgraded = sorted(set(big.loc[big["label_source"] != "goliath_original", "id"]))
    os.makedirs(os.path.dirname(args.out_ids) or ".", exist_ok=True)
    with open(args.out_ids, "w") as f:
        f.write("\n".join(upgraded) + "\n")
    print(f"[+] {args.out_csv}; {len(upgraded)} upgraded proteins so far (M-CSA + M3Site)")


if __name__ == "__main__":
    main()
