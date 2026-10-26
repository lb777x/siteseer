"""Step 15: Squidly's CataloDB split in our dataset format, for the same-data comparison with Squidly.

As in Squidly's training script: sequences from fastas/train_and_test.fasta (24 test ids only in fastas/test.fasta),
labels from UniProt ACT_SITE in All_metadata.tsv, test = final_test_set_post_structural_filtering.txt (< 30% sequence
and structure identity to training); valid = random 10% of the remaining proteins (seed 42; Squidly also
early-stops on a random split). Proteins without an ACT_SITE in range or longer than 1022 residues are dropped.
Labels are singleton groups with label_source 'catalodb'; _original_row_index keys an embedding cache of its own.

  python pipeline/15_catalodb_split.py
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

from common import EXTERNAL

ACT_SITE = re.compile(r"ACT_SITE\s+(\d+)")


def parse_fasta(path: str) -> dict:
    """UniProt FASTA -> {accession: sequence} (accession = 2nd '|' field of the header)."""
    seqs, acc, chunks = {}, None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if acc is not None:
                    seqs[acc] = "".join(chunks)
                acc, chunks = line[1:].split("|")[1], []
            else:
                chunks.append(line.strip())
    if acc is not None:
        seqs[acc] = "".join(chunks)
    return seqs


def active_sites(cell) -> list:
    """Unique ACT_SITE positions of a UniProt 'Active site' cell, in order of appearance."""
    out = []
    for m in ACT_SITE.findall(cell if isinstance(cell, str) else ""):
        if int(m) not in out:
            out.append(int(m))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalodb-dir", default=f"{EXTERNAL}/catalodb")
    ap.add_argument("--out-dir", default="dataset/catalodb_squidly")
    ap.add_argument("--valid-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    d = args.catalodb_dir

    seqs = parse_fasta(f"{d}/fastas/train_and_test.fasta")
    test_fasta = parse_fasta(f"{d}/fastas/test.fasta")
    meta = pd.read_csv(f"{d}/All_metadata.tsv", sep="\t", usecols=["Entry", "Active site"], dtype=str)
    sites = dict(zip(meta["Entry"], meta["Active site"]))
    test_ids = {l.strip() for l in open(f"{d}/final_test_set_post_structural_filtering.txt") if l.strip()}
    mismatch = [a for a in test_ids & set(seqs) & set(test_fasta) if seqs[a] != test_fasta[a]]
    if mismatch:
        raise SystemExit(f"{len(mismatch)} test ids differ between the two fastas, e.g. {mismatch[:5]}")
    for a in sorted(test_ids & set(test_fasta)):   # sorted: a set's order changes between Python processes
        seqs.setdefault(a, test_fasta[a])

    rows, drop = [], {"no_metadata": 0, "no_act_site": 0, "site_out_of_range": 0, "longer_than_1022": 0}
    for acc, seq in seqs.items():
        if acc not in sites:
            drop["no_metadata"] += 1
            continue
        pos = active_sites(sites[acc])
        if not pos:
            drop["no_act_site"] += 1
        elif any(p < 1 or p > len(seq) for p in pos):
            drop["site_out_of_range"] += 1
        elif len(seq) > 1022:
            drop["longer_than_1022"] += 1
        else:
            rows.append({"alphafolddb-id": acc, "aa_sequence": seq, "site_labels": str([[p] for p in sorted(pos)]),
                         "site_types": str([0] * len(pos)), "label_source": "catalodb",
                         "dataset_flag": "test" if acc in test_ids else "train"})
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(args.seed)
    tr = df.index[df.dataset_flag == "train"].to_numpy()
    df.loc[rng.choice(tr, int(round(args.valid_frac * len(tr))), replace=False), "dataset_flag"] = "valid"
    df["_original_row_index"] = range(len(df))
    df.to_csv(f"{args.out_dir}/catalodb_squidly.csv", index=False)
    with open(f"{args.out_dir}/test.fasta", "w") as f:
        t = df[df.dataset_flag == "test"]
        for acc, seq in zip(t["alphafolddb-id"], t["aa_sequence"]):
            f.write(f">{acc}\n{seq}\n")
    rep = {"test_ids_missing": sorted(test_ids - set(seqs)), "dropped": drop,
           "n_by_split": df.dataset_flag.value_counts().to_dict()}
    with open(f"{args.out_dir}/split_report.json", "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
