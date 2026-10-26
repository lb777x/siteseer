"""Step 01: replace the Goliath pool's weak labels with M-CSA's curated catalytic residues where a Goliath training
protein is an M-CSA training protein or a close homolog of one.

  pass 1  same UniProt id and identical sequence: copy the M-CSA labels
  pass 2  BLAST (blastp, 5 best targets, e-value <= 1e-5) of every M-CSA training protein against EasIFA's Goliath
          BLAST database; for a Goliath training protein with a hit of >= 60% identity, the best hit's catalytic
          residues are mapped through the alignment (residues in gaps or outside the aligned region are dropped)

Only rows with dataset_flag == 'train' (EasIFA's split) are touched, so nothing is transferred into an evaluation
split. Every upgraded protein id goes to data/pipeline/mcsa_derived_goliath_ids.txt.
Note: fix2 (step 11) later removes the homology-transferred rows (label_source mcsa_blast_homology) again, because
their donors rotate into the M-CSA K-fold test folds.

  python pipeline/01_mcsa_label_transfer.py
"""
import argparse
import ast
import os

import pandas as pd

from common import (BLAST_COLUMNS, GOLIATH_BLAST_DB, GOLIATH_CSV, LABELS, MCSA_CSV, PIPELINE, blastp,
                    build_alignment_map, remap_labels, write_fasta)


def parse_labels(raw) -> list:
    """'[[68], [71], [128]]' -> [[68], [71], [128]]; group structure kept so site_types stay index-aligned."""
    try:
        parsed = ast.literal_eval(str(raw))
    except (ValueError, SyntaxError):
        return []
    return [[int(x) for x in (g if isinstance(g, (list, tuple)) else [g])] for g in parsed]


def parse_types(raw) -> list:
    try:
        return list(ast.literal_eval(str(raw)))
    except (ValueError, SyntaxError):
        return []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--goliath-csv", default=GOLIATH_CSV)
    ap.add_argument("--goliath-blast-db", default=GOLIATH_BLAST_DB)
    ap.add_argument("--mcsa-train-csv", default=MCSA_CSV["train"])
    ap.add_argument("--work-dir", default=f"{PIPELINE}/label_transfer_mcsa")
    ap.add_argument("--out-csv", default=f"{LABELS}/cached_dataframe_mcsa_labels.csv")
    ap.add_argument("--out-ids", default=f"{PIPELINE}/mcsa_derived_goliath_ids.txt")
    ap.add_argument("--identity-threshold", type=float, default=60.0)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    mcsa = pd.read_csv(args.mcsa_train_csv, usecols=["alphafolddb-id", "aa_sequence", "site_labels", "site_types"])
    mcsa["id"] = mcsa["alphafolddb-id"].astype(str).str.strip()
    mcsa = mcsa.drop_duplicates(subset="id").set_index("id")
    query = f"{args.work_dir}/mcsa_train_query.fasta"
    hits_path = f"{args.work_dir}/blast_mcsa_train_vs_goliath.txt"
    write_fasta(mcsa.index, mcsa["aa_sequence"], query)
    blastp(query, args.goliath_blast_db, hits_path, max_target_seqs=5, evalue=1e-5, threads=args.threads)

    big = pd.read_csv(args.goliath_csv)
    big["id"] = big["alphafolddb-id"].astype(str).str.strip()
    train_mask = big["dataset_flag"] == "train"
    train_ids = set(big.loc[train_mask, "id"])
    big["label_source"] = "goliath_original"
    big["label_confidence"] = pd.NA

    n_exact = 0
    for idx in big.index[train_mask]:
        pid = big.at[idx, "id"]
        if pid not in mcsa.index:
            continue
        row = mcsa.loc[pid]
        if str(big.at[idx, "aa_sequence"]).strip() != str(row["aa_sequence"]).strip():
            continue   # same id, different sequence version: positions are not transferable verbatim
        big.at[idx, "site_labels"] = str(parse_labels(row["site_labels"]))
        big.at[idx, "site_types"] = str(parse_types(row["site_types"]))
        big.at[idx, "label_source"] = "mcsa_exact"
        big.at[idx, "label_confidence"] = 100.0
        n_exact += 1
    print(f"[*] pass 1 (same id and sequence): {n_exact} rows")

    exact = set(big.loc[big["label_source"] == "mcsa_exact", "id"])
    hits = pd.read_csv(hits_path, sep="\t", names=BLAST_COLUMNS)
    hits = hits[hits["sseqid"].isin(train_ids) & ~hits["sseqid"].isin(exact) & (hits["pident"] >= args.identity_threshold)]
    hits = hits.sort_values("pident", ascending=False).drop_duplicates(subset="sseqid", keep="first")
    row_of = {pid: idx for idx, pid in zip(big.index[train_mask], big.loc[train_mask, "id"])}
    n_hom = n_lost = 0
    for _, hit in hits.iterrows():
        if hit["qseqid"] not in mcsa.index or hit["sseqid"] not in row_of:
            continue
        idx, row = row_of[hit["sseqid"]], mcsa.loc[hit["qseqid"]]
        pos_map = build_alignment_map(int(hit["qstart"]), int(hit["qend"]), int(hit["sstart"]), int(hit["send"]),
                                      hit["qseq"], hit["sseq"])
        labels, types = remap_labels(parse_labels(row["site_labels"]), parse_types(row["site_types"]), pos_map)
        if not labels:
            n_lost += 1
            continue
        big.at[idx, "site_labels"] = str(labels)
        big.at[idx, "site_types"] = str(types)
        big.at[idx, "label_source"] = "mcsa_blast_homology"
        big.at[idx, "label_confidence"] = float(hit["pident"])
        n_hom += 1
    print(f"[*] pass 2 (homology >= {args.identity_threshold}%): {n_hom} rows ({n_lost} hits without an aligned site)")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    big.drop(columns=["id"]).to_csv(args.out_csv, index=False)
    upgraded = sorted(set(big.loc[big["label_source"] != "goliath_original", "id"]))
    os.makedirs(os.path.dirname(args.out_ids) or ".", exist_ok=True)
    with open(args.out_ids, "w") as f:
        f.write("\n".join(upgraded) + "\n")
    print(f"[+] {args.out_csv}; {len(upgraded)} upgraded proteins -> {args.out_ids}")


if __name__ == "__main__":
    main()
