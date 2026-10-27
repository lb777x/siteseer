"""Structures for the structural ablation: the AlphaFold DB model of every protein in the given CSVs (v6, else v5,
else v4) and, for ids that are 4-character PDB codes, the RCSB entry. Files are named as graph.resolve_pdb_path
expects; existing files are kept. The reported runs used the models available in 2026-08 (mostly v6).

  python -m siteseer.structural.download --csv CSV [CSV ...] --out-dir data/structures
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from siteseer.structural.graph import AF_MODEL_VERSIONS


def present(uid, root):
    return any(os.path.exists(os.path.join(root, f"AF-{uid}-F1-model_v{v}.pdb")) for v in AF_MODEL_VERSIONS) or \
        os.path.exists(os.path.join(root, f"{uid.lower()}.pdb"))


def fetch(uid, root):
    if present(uid, root):
        return "present"
    for v in (6, 5, 4):
        r = requests.get(f"https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v{v}.pdb", timeout=60)
        if r.status_code == 200:
            open(os.path.join(root, f"AF-{uid}-F1-model_v{v}.pdb"), "w").write(r.text)
            return f"v{v}"
    if len(uid) == 4:
        r = requests.get(f"https://files.rcsb.org/download/{uid.upper()}.pdb", timeout=60)
        if r.status_code == 200:
            open(os.path.join(root, f"{uid.lower()}.pdb"), "w").write(r.text)
            return "rcsb"
    return "missing"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--out-dir", default="data/structures")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    ids = sorted({str(a).strip() for p in args.csv for a in pd.read_csv(p, usecols=["alphafolddb-id"])["alphafolddb-id"]
                  if isinstance(a, str) and a.strip()})
    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(lambda u: fetch(u, args.out_dir), ids))
    counts = pd.Series(res).value_counts().to_dict()
    print(f"[+] {len(ids)} proteins: {counts}")


if __name__ == "__main__":
    main()
