"""Step 10 (fix2): remove the M-CSA labels transferred to homologs (step 01, pass 2) from stages 1 and 1.5.

Step 01 copied M-CSA's catalytic residues onto 60-90% identical Goliath proteins (label_source
mcsa_blast_homology). In the M-CSA 5-fold cross-validation the donors rotate into the test folds, so a fold-test
protein could have a relative upstream carrying its own labels (257 of 962). Every protein that carries such a label
in either fix1 file is flagged 'excluded_mcsa_label_transfer' in train and valid (all its rows). Rows are re-flagged,
never dropped, so the embedding-cache keys stay valid; the stage-1/1.5 test split is unchanged.

  python pipeline/10_fix2_datasets.py      # -> cached_dataframe_{fully_excl,real_labels_only}_fix2.csv
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
