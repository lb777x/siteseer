"""Step 06: the labelled pool with the new split.

The pool rows are EasIFA's rows with the upgraded labels of step 03 (cached_dataframe_all_sources_labels.csv)
followed by the new M3Site/BioLiP rows of step 04 (the rows of cached_dataframe_expanded.csv beyond EasIFA's), in
this order; row positions are the keys of the embedding caches. The split mapping of step 05 then replaces
dataset_flag for every clustered protein; others keep their flag. With --exclude-ids-file, those proteins get
dataset_flag 'excluded_easifa_test' (step 07).

  python pipeline/06_apply_split.py         # -> dataset/label_diffusion/cached_dataframe_final_resplit90.csv
"""
import argparse
import os

import pandas as pd

from common import EASIFA_DIR, LABELS, PIPELINE

EXCLUDED_FLAG = "excluded_easifa_test"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-csv", default=f"{LABELS}/cached_dataframe_all_sources_labels.csv")
    ap.add_argument("--expanded-csv", default=f"{EASIFA_DIR}/cached_dataframe_expanded.csv")
    ap.add_argument("--mapping", default=f"{PIPELINE}/resplit/goliath_cluster_assignment_90.csv")
    ap.add_argument("--out-csv", default=f"{LABELS}/cached_dataframe_final_resplit90.csv")
    ap.add_argument("--exclude-ids-file", default=None)
    ap.add_argument("--source", choices=["labels", "expanded"], default="labels",
                    help="'labels': upgraded labels + new rows (training pool); 'expanded': the extended pool "
                         "with EasIFA's original labels (the split-only lineage of step 07)")
    args = ap.parse_args()

    expanded = pd.read_csv(args.expanded_csv)
    if args.source == "labels":
        labelled = pd.read_csv(args.labels_csv)
        if not labelled["alphafolddb-id"].equals(expanded["alphafolddb-id"].iloc[:len(labelled)]):
            raise SystemExit("labels CSV and expanded pool disagree on the row order of EasIFA's rows")
        df = pd.concat([labelled, expanded.iloc[len(labelled):]], ignore_index=True)
    else:
        df = expanded
    mapping = pd.read_csv(args.mapping, usecols=["alphafolddb-id", "new_dataset_flag"])
    split_of = dict(zip(mapping["alphafolddb-id"].astype(str).str.strip(), mapping["new_dataset_flag"]))
    ids = df["alphafolddb-id"].astype(str).str.strip()
    df["dataset_flag"] = ids.map(split_of).fillna(df["dataset_flag"])
    if args.exclude_ids_file:
        exclude = {line.strip() for line in open(args.exclude_ids_file) if line.strip()}
        df.loc[ids.isin(exclude), "dataset_flag"] = EXCLUDED_FLAG
        print(f"[*] {ids[ids.isin(exclude)].nunique()} of {len(exclude)} listed proteins flagged {EXCLUDED_FLAG}")
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[+] {args.out_csv}: {len(df)} rows, flags {df['dataset_flag'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
