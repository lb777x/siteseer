"""Step 14: EasIFA's own training data as a training set, for the same-data comparison on its Swiss-Prot benchmark.

train/valid: EasIFA's split and labels (Goliath pool, protein-reaction pairs, unmodified); test: the 825 proteins of
step 13. Rows are keyed to the embedding caches by (id, sequence) in cached_dataframe_final_resplit90.csv; label
source empty = EasIFA's range convention. The 107 test proteins that also occur in EasIFA's train/valid split stay
in training, as in EasIFA's own setup; the comparison excludes them (step 13, subset noleak).

  python pipeline/14_easifa_own_split.py
"""
import argparse
import json

import pandas as pd

from common import GOLIATH_CSV, LABELS, PIPELINE

ID = "alphafolddb-id"
COLS = [ID, "aa_sequence", "site_labels", "site_types", "reaction", "dataset_flag"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=f"{LABELS}/cached_dataframe_final_resplit90.csv")
    ap.add_argument("--eval-csv", default=f"{LABELS}/easifa_swissprot_eval_fix1.csv")
    ap.add_argument("--out", default=f"{LABELS}/easifa_own_split_fix1.csv")
    args = ap.parse_args()

    o = pd.read_csv(GOLIATH_CSV, low_memory=False)
    o[ID] = o[ID].astype(str).str.strip()
    pool = pd.read_csv(args.pool, usecols=[ID, "aa_sequence"], low_memory=False)
    pool[ID] = pool[ID].astype(str).str.strip()
    row_of = {}
    for k, (a, s) in enumerate(zip(pool[ID], pool["aa_sequence"])):
        row_of.setdefault((a, s), k)
    tr = o[o.dataset_flag.isin(["train", "valid"])][COLS].copy()
    tr["_original_row_index"] = [row_of.get((a, s)) for a, s in zip(tr[ID], tr["aa_sequence"])]
    missing = int(tr["_original_row_index"].isna().sum())
    tr = tr.dropna(subset=["_original_row_index"])

    ev = pd.read_csv(args.eval_csv)
    te = ev[ev.dataset_flag == "test"][COLS + ["_original_row_index"]].copy()
    df = pd.concat([tr, te], ignore_index=True)
    df["label_source"] = ""
    df["_original_row_index"] = df["_original_row_index"].astype(int)
    df.to_csv(args.out, index=False)
    rep = {"pairs_by_split": df.dataset_flag.value_counts().to_dict(),
           "train_valid_rows_without_cached_embedding": missing,
           "test_proteins_also_in_easifa_train_valid": len(set(te[ID]) & set(tr[ID]))}
    with open(f"{PIPELINE}/easifa_own_split_report.json", "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
