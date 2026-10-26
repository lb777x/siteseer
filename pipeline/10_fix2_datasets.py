"""Step 10: flag proteins with M-CSA labels transferred by homology (step 01) in train and valid, so no relative
of an M-CSA cross-validation test protein carries its labels upstream.

  python pipeline/10_fix2_datasets.py
"""
import argparse
import json
import os

import pandas as pd

from common import LABELS, PIPELINE

ID = "alphafolddb-id"
FLAG = "excluded_mcsa_label_transfer"
PAIRS = (("cached_dataframe_fully_excl_fix1.csv", "cached_dataframe_fully_excl_fix2.csv"),
         ("cached_dataframe_real_labels_only_fix1.csv", "cached_dataframe_real_labels_only_fix2.csv"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default=LABELS)
    ap.add_argument("--out-dir", default=LABELS)
    ap.add_argument("--report", default=f"{PIPELINE}/fix2_mcsa_label_transfer.json")
    args = ap.parse_args()
    frames = {src: pd.read_csv(os.path.join(args.in_dir, src), low_memory=False) for src, _ in PAIRS}
    ids = set()
    for df in frames.values():
        ids |= set(df.loc[df.label_source == "mcsa_blast_homology", ID].astype(str).str.strip())
    report = {"proteins_with_transferred_mcsa_labels": len(ids), "files": {}}
    os.makedirs(args.out_dir, exist_ok=True)
    for src, dst in PAIRS:
        df = frames[src]
        hit = df[ID].astype(str).str.strip().isin(ids) & df.dataset_flag.isin(["train", "valid"])
        before = df.dataset_flag.value_counts().to_dict()
        df.loc[hit, "dataset_flag"] = FLAG
        df.to_csv(os.path.join(args.out_dir, dst), index=False)
        uniq = lambda f: int(df.loc[df.dataset_flag == f, ID].nunique())
        report["files"][dst] = {"rows_reflagged": int(hit.sum()), "unique_proteins_reflagged": int(df.loc[hit, ID].nunique()),
                                "rows_before": before, "rows_after": df.dataset_flag.value_counts().to_dict(),
                                "unique_after": {f: uniq(f) for f in ("train", "valid", "test")}}
        print(f"[*] {dst}: {int(hit.sum())} rows of {df.loc[hit, ID].nunique()} proteins flagged; "
              f"train/valid/test {uniq('train')}/{uniq('valid')}/{uniq('test')}")
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    json.dump(report, open(args.report, "w"), indent=1)


if __name__ == "__main__":
    main()
