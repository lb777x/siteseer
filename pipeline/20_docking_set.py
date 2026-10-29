"""Step 20: ligands and protein set for the docking application.

Ligand per protein: the longest reactant SMILES (characters) of its first reaction (rxn_smiles) in EasIFA's Goliath
pool. Set: test proteins of the stage-1 data (fix1; never trained on) with a ligand, exactly one AlphaFold model
AF-<uid>-F1-*.pdb whose sequence equals aa_sequence with numbering 1..L (predicted positions and PDB residue numbers
then coincide), and <= 1022 residues; shuffled (seed 42) so any prefix is a random sample.

  python pipeline/20_docking_set.py [--structure-root data/structures]
"""
import argparse
import glob
import json
import os

import pandas as pd

from common import GOLIATH_CSV, LABELS
from siteseer.structural.graph import load_structure_frames, resolve_pdb_path

COLS = ["alphafolddb-id", "aa_sequence", "reaction", "site_labels", "label_source", "pdb_files"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structure-root", default="data/structures")
    ap.add_argument("--ligands-out", default="dataset/docking/ligand_smiles.json")
    ap.add_argument("--out", default="dataset/docking/goliath_test_dock.csv")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    g = pd.read_csv(GOLIATH_CSV, usecols=["alphafolddb-id", "rxn_smiles"])
    g["id"] = g["alphafolddb-id"].astype(str).str.strip()
    ligands = {i: max(r.split(">>")[0].split("."), key=len)
               for i, r in g.drop_duplicates("id")[["id", "rxn_smiles"]].itertuples(index=False) if isinstance(r, str)}
    with open(args.ligands_out, "w") as f:
        json.dump(ligands, f)

    df = pd.read_csv(f"{LABELS}/cached_dataframe_fully_excl_fix1.csv", usecols=COLS + ["dataset_flag"])
    df = df[df["dataset_flag"] == "test"].drop_duplicates("alphafolddb-id")
    counts = {"test": len(df), "no_ligand": 0, "structure_missing_or_ambiguous": 0, "too_long": 0,
              "sequence_differs": 0}
    keep = []
    for _, row in df.iterrows():
        raw = str(row["alphafolddb-id"]).strip()
        uid = raw.split("-")[1] if raw.startswith("AF-") else raw
        if not ligands.get(uid):
            counts["no_ligand"] += 1
            continue
        hits = glob.glob(os.path.join(args.structure_root, f"AF-{uid}-F1-*.pdb"))
        path = resolve_pdb_path(row, args.structure_root)
        if len(hits) != 1 or path is None or os.path.abspath(path) != os.path.abspath(hits[0]):
            counts["structure_missing_or_ambiguous"] += 1
            continue
        seq = row["aa_sequence"]
        if not isinstance(seq, str) or len(seq) > 1022:
            counts["too_long"] += 1
            continue
        try:
            fr = load_structure_frames(path)
        except Exception:
            counts["structure_missing_or_ambiguous"] += 1
            continue
        if fr.sequence != seq or list(fr.residue_numbers) != list(range(1, len(seq) + 1)):
            counts["sequence_differs"] += 1
            continue
        keep.append(row)
    out = pd.DataFrame(keep)[COLS].sample(frac=1.0, random_state=42).reset_index(drop=True)
    out.to_csv(args.out, index=False)
    print(f"[+] {len(ligands)} ligands -> {args.ligands_out}; {len(out)} proteins -> {args.out}; {counts}")


if __name__ == "__main__":
    main()
