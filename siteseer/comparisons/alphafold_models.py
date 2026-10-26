"""AlphaFold DB models for the P2Rank comparison: V and U pocket candidates, set G, Goliath and M-CSA test proteins.

The API serves the current model version; the reported run used the models downloaded on 2026-08-01 (AlphaFold
Monomer v2.0 predictions). P2Rank's outputs on them are in the data release.

  python -m siteseer.comparisons.alphafold_models --out-dir data/comparisons/p2rank/pdb
"""
import argparse
import os

import pandas as pd
import requests

ID = "alphafolddb-id"
SOURCES = [("dataset/negatives/pocket_candidates.csv", "accession", None),
           ("dataset/negatives/negatives_gold.csv", "accession", None),
           ("dataset/label_diffusion/cached_dataframe_fully_excl_fix1.csv", ID, "test"),
           ("dataset/mcsa_fine_tune/normal_mcsa/test_dataset/mcsa_test.csv", ID, None)]


def fetch(acc, folder):
    path = os.path.join(folder, f"AF-{acc}.pdb")
    if os.path.exists(path):
        return True
    try:
        r = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{acc}", timeout=60)
        r.raise_for_status()
        pdb = requests.get(r.json()[0]["pdbUrl"], timeout=120)
        pdb.raise_for_status()
    except Exception as e:
        print(f"  [!] {acc}: no AlphaFold model ({type(e).__name__})")
        return False
    with open(path + ".tmp", "w") as f:
        f.write(pdb.text)
    os.replace(path + ".tmp", path)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="data/comparisons/p2rank/pdb")
    ap.add_argument("--dataset-file", default="data/comparisons/p2rank/all.ds", help="P2Rank input list to write")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    accs = []
    for path, col, flag in SOURCES:
        d = pd.read_csv(path, low_memory=False)
        if flag:
            d = d[d["dataset_flag"] == flag]
        accs += d[col].astype(str).str.strip().tolist()
    accs = sorted(set(accs))
    ok = [a for a in accs if fetch(a, args.out_dir)]
    with open(args.dataset_file, "w") as f:
        f.write("\n".join(os.path.abspath(os.path.join(args.out_dir, f"AF-{a}.pdb")) for a in ok) + "\n")
    print(f"[+] {len(ok)} of {len(accs)} models; P2Rank list {args.dataset_file}")


if __name__ == "__main__":
    main()
