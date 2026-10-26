"""Step 07: ids of the 853 proteins of EasIFA's Swiss-Prot test set, kept out of all training splits.

  python pipeline/07_easifa_test_ids.py
"""
import argparse
import os

import pandas as pd

from common import EASIFA_DIR, PIPELINE


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--easifa-test-csv", default=f"{EASIFA_DIR}/test_dataset/uniprot_ecreact_merge.csv")
    ap.add_argument("--out", default=f"{PIPELINE}/easifa_test_protein_ids.txt")
    args = ap.parse_args()
    ids = pd.read_csv(args.easifa_test_csv)["alphafolddb-id"].astype(str).str.strip().str.rstrip(";").str.strip()
    ids = sorted(set(ids))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(ids) + "\n")
    print(f"[+] {args.out}: {len(ids)} proteins")


if __name__ == "__main__":
    main()
