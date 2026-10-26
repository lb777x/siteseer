"""Step 04: add M3Site and BioLiP proteins that are not in the Goliath pool or in M-CSA as new rows.

Starts from EasIFA's original cached_dataframe.csv (the clustering in step 05 only needs sequences; the upgraded
labels of the existing rows come from step 03 and are joined in step 06). New M3Site proteins take their sequence
from M3Site's split TSVs or from UniProt (data/external/m3site/m3site_sequences_fetched.json, fetch_m3site_sequences.py;
UniProt changes over time, the frozen copy is in the data release). A protein new in both sources keeps its BioLiP
row (labels tied to an experimental structure). New rows get an empty dataset_flag; step 05 assigns the split.

  python pipeline/04_expand_pool.py
"""
import argparse
import importlib.util
import json
import os

import pandas as pd

from common import EASIFA_DIR, EXTERNAL, GOLIATH_CSV, MCSA_CSV


def _step(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--goliath-csv", default=GOLIATH_CSV)
    ap.add_argument("--mcsa-csv", nargs=3, default=[MCSA_CSV[s] for s in ("train", "valid", "test")])
    ap.add_argument("--m3site-dir", default=f"{EXTERNAL}/m3site")
    ap.add_argument("--biolip", default=f"{EXTERNAL}/biolip/BioLiP_nr.txt")
    ap.add_argument("--out-csv", default=f"{EASIFA_DIR}/cached_dataframe_expanded.csv")
    args = ap.parse_args()
    dense_to_groups = _step("02_m3site_labels").dense_to_groups
    load_biolip = _step("03_biolip_labels").load_biolip_representatives

    big = pd.read_csv(args.goliath_csv)
    existing = set(big["alphafolddb-id"].astype(str).str.strip())
    for path in args.mcsa_csv:
        existing |= set(pd.read_csv(path, usecols=["alphafolddb-id"])["alphafolddb-id"].astype(str).str.strip())

    m3_labels = json.load(open(f"{args.m3site_dir}/label.json"))
    m3_seqs = {}
    for split in ("train_0.5.tsv", "test_0.5.tsv"):
        df = pd.read_csv(f"{args.m3site_dir}/{split}", sep="\t")
        m3_seqs.update(dict(zip(df["Entry"], df["Sequence"])))
    m3_seqs.update(json.load(open(f"{args.m3site_dir}/m3site_sequences_fetched.json")))
    biolip = load_biolip(args.biolip)
    new_m3 = sorted(set(m3_labels) - existing)
    new_biolip = sorted(set(biolip) - existing)
    both = set(new_m3) & set(new_biolip)
    print(f"[*] new proteins: {len(new_m3)} M3Site, {len(new_biolip)} BioLiP, {len(both)} in both (BioLiP row kept)")

    rows, n_no_seq, n_no_labels = [], 0, 0
    for pid in new_m3:
        if pid in both:
            continue
        seq = m3_seqs.get(pid)
        if not seq:
            n_no_seq += 1
            continue
        labels, types = dense_to_groups(m3_labels[pid])
        if not labels:
            n_no_labels += 1
            continue
        rows.append({"alphafolddb-id": pid, "aa_sequence": seq, "site_labels": str(labels), "site_types": str(types),
                     "dataset_flag": None, "label_source": "m3site_new"})
    for pid in new_biolip:
        entry = biolip[pid]
        if entry["sequence"]:
            rows.append({"alphafolddb-id": pid, "aa_sequence": entry["sequence"], "site_labels": str(entry["groups"]),
                         "site_types": str([0] * len(entry["groups"])), "dataset_flag": None,
                         "label_source": "biolip_new"})
    combined = pd.concat([big, pd.DataFrame(rows)], ignore_index=True)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    combined.to_csv(args.out_csv, index=False)
    print(f"[+] {args.out_csv}: {len(big)} + {len(rows)} = {len(combined)} rows "
          f"({n_no_seq} M3Site proteins without sequence, {n_no_labels} without labels)")


if __name__ == "__main__":
    main()
