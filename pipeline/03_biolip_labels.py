"""Step 03: BioLiP2 catalytic-site labels for Goliath training proteins not upgraded by steps 01-02.

BioLiP (Yang, Roy & Zhang; BioLiP2, NAR 2024), non-redundant table BioLiP_nr.txt (zhanggroup.org/BioLiP; the table
changes weekly, the frozen copy used here is in the data release). Column 11 holds the catalytic residues, numbered
into the receptor sequence of that PDB chain (column 21), which is often a fragment with gaps. Per UniProt id the row
with the longest receptor sequence is used.

  pass 1  receptor sequence identical to the Goliath sequence: copy the groups (label_source biolip_exact)
  pass 2  blastp (5 best targets, e-value <= 1e-5) of the receptor sequences against the Goliath BLAST database;
          hits of >= 60% identity map the residues through the alignment (biolip_blast_homology)

Rows already upgraded by M-CSA or M3Site are never overwritten. Writes the training CSV and the list of every
protein whose labels now come from M-CSA, M3Site or BioLiP.

  python pipeline/03_biolip_labels.py
"""
import argparse
import csv
import os

import pandas as pd

from common import (BLAST_COLUMNS, EXTERNAL, GOLIATH_BLAST_DB, LABELS, PIPELINE, blastp, build_alignment_map,
                    remap_labels, write_fasta)


def parse_catalytic_groups(raw: str) -> list:
    """'C81 E119 R163 R165' -> [[81, 119, 163, 165]]; ';' separates groups."""
    groups = []
    for block in raw.split(";"):
        positions = [int(tok[1:]) for tok in block.split() if tok[1:].isdigit()]
        if positions:
            groups.append(positions)
    return groups


def load_biolip_representatives(path: str) -> dict:
    """uniprot id -> {'sequence', 'groups'}, the longest receptor sequence per id."""
    best = {}
    with open(path) as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 21:
                continue
            catalytic, uid, seq = row[10].strip(), row[17].strip(), row[20].strip()
            if not (catalytic and uid and seq):
                continue
            if uid not in best or len(seq) > len(best[uid]["sequence"]):
                best[uid] = {"sequence": seq, "groups": parse_catalytic_groups(catalytic)}
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-csv", default=f"{LABELS}/cached_dataframe_mcsa_m3site_labels.csv")
    ap.add_argument("--biolip", default=f"{EXTERNAL}/biolip/BioLiP_nr.txt")
    ap.add_argument("--goliath-blast-db", default=GOLIATH_BLAST_DB)
    ap.add_argument("--work-dir", default=f"{PIPELINE}/label_transfer_biolip")
    ap.add_argument("--out-csv", default=f"{LABELS}/cached_dataframe_all_sources_labels.csv")
    ap.add_argument("--out-ids", default=f"{PIPELINE}/external_derived_goliath_ids.txt")
    ap.add_argument("--identity-threshold", type=float, default=60.0)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    biolip = load_biolip_representatives(args.biolip)
    print(f"[*] {len(biolip)} UniProt proteins with a BioLiP catalytic-site row")
    query = f"{args.work_dir}/biolip_catalytic_query.fasta"
    hits_path = f"{args.work_dir}/blast_biolip_vs_goliath.txt"
    write_fasta(biolip, (e["sequence"] for e in biolip.values()), query)
    blastp(query, args.goliath_blast_db, hits_path, max_target_seqs=5, evalue=1e-5, threads=args.threads)

    big = pd.read_csv(args.in_csv)
    big["id"] = big["alphafolddb-id"].astype(str).str.strip()
    train = big["dataset_flag"] == "train"
    train_ids = set(big.loc[train, "id"])
    handled = set(big.loc[big["label_source"] != "goliath_original", "id"])
    row_of = {pid: idx for idx, pid in zip(big.index[train], big.loc[train, "id"])}

    n_exact = 0
    for pid, entry in biolip.items():
        if pid in handled or pid not in row_of:
            continue
        idx = row_of[pid]
        if str(big.at[idx, "aa_sequence"]).strip() != entry["sequence"]:
            continue
        big.at[idx, "site_labels"] = str(entry["groups"])
        big.at[idx, "site_types"] = str([0] * len(entry["groups"]))
        big.at[idx, "label_source"] = "biolip_exact"
        big.at[idx, "label_confidence"] = 100.0
        handled.add(pid)
        n_exact += 1
    print(f"[*] pass 1 (identical sequence): {n_exact} rows")

    hits = pd.read_csv(hits_path, sep="\t", names=BLAST_COLUMNS)
    hits = hits[hits["sseqid"].isin(train_ids) & ~hits["sseqid"].isin(handled) & (hits["pident"] >= args.identity_threshold)]
    hits = hits.sort_values("pident", ascending=False).drop_duplicates(subset="sseqid", keep="first")
    n_hom = n_lost = 0
    for _, hit in hits.iterrows():
        if hit["qseqid"] not in biolip or hit["sseqid"] not in row_of:
            continue
        idx, entry = row_of[hit["sseqid"]], biolip[hit["qseqid"]]
        pos_map = build_alignment_map(int(hit["qstart"]), int(hit["qend"]), int(hit["sstart"]), int(hit["send"]),
                                      hit["qseq"], hit["sseq"])
        labels, types = remap_labels(entry["groups"], [0] * len(entry["groups"]), pos_map)
        if not labels:
            n_lost += 1
            continue
        big.at[idx, "site_labels"] = str(labels)
        big.at[idx, "site_types"] = str(types)
        big.at[idx, "label_source"] = "biolip_blast_homology"
        big.at[idx, "label_confidence"] = float(hit["pident"])
        n_hom += 1
    print(f"[*] pass 2 (homology >= {args.identity_threshold}%): {n_hom} rows ({n_lost} hits without an aligned site)")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    big.drop(columns=["id"]).to_csv(args.out_csv, index=False)
    upgraded = sorted(set(big.loc[big["label_source"] != "goliath_original", "id"]))
    os.makedirs(os.path.dirname(args.out_ids) or ".", exist_ok=True)
    with open(args.out_ids, "w") as f:
        f.write("\n".join(upgraded) + "\n")
    print(f"[+] {args.out_csv}; {len(upgraded)} proteins with M-CSA, M3Site or BioLiP labels -> {args.out_ids}")


if __name__ == "__main__":
    main()
