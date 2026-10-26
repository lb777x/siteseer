"""Step 08: flag training rows in the same 90%-identity cluster (step 05 map) as protected proteins.

Applied twice: M-CSA valid+test (excluded_mcsa_cluster90), then EasIFA's test set (excluded_easifa_test_cluster90).

  python pipeline/08_cluster_exclusion.py --source-csv IN --out-csv OUT --flag NAME \\
      [--protect-csv CSV ...] [--protect-ids IDS.txt]
"""
import argparse
import os

import pandas as pd

from common import PIPELINE

ID = "alphafolddb-id"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-csv", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--flag", required=True)
    ap.add_argument("--protect-csv", nargs="*", default=[], help="CSVs whose protein ids are protected")
    ap.add_argument("--protect-ids", default=None, help="file with protected protein ids, one per line")
    ap.add_argument("--cluster-map", default=f"{PIPELINE}/resplit/goliath_cluster_assignment_90.csv")
    args = ap.parse_args()

    protected = set()
    for path in args.protect_csv:
        protected |= set(pd.read_csv(path)[ID].astype(str).str.strip())
    if args.protect_ids:
        protected |= {line.strip() for line in open(args.protect_ids) if line.strip()}
    cmap = pd.read_csv(args.cluster_map)
    cmap[ID] = cmap[ID].astype(str).str.strip()
    clusters = set(cmap.loc[cmap[ID].isin(protected), "cluster_rep_id"])
    members = set(cmap.loc[cmap["cluster_rep_id"].isin(clusters), ID])
    print(f"[*] {len(protected)} protected proteins, {cmap[ID].isin(protected).sum()} in the cluster map, "
          f"{len(clusters)} clusters, {len(members)} proteins in them")
    df = pd.read_csv(args.source_csv)
    df[ID] = df[ID].astype(str).str.strip()
    hit = df[ID].isin(members) & (df["dataset_flag"] == "train")
    df.loc[hit, "dataset_flag"] = args.flag
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[+] {args.out_csv}: {hit.sum()} training rows flagged {args.flag}")


if __name__ == "__main__":
    main()
