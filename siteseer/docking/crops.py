"""Predicted active-site residues and pocket crops for the docking benchmark.

Per protein: the probe's residue probabilities on the sequence (first 1022 residues); residues with p > 0.5, or
the single most probable residue if none passes. Writes <uid>.pdb crops (crop.py) and a (uid, residues) CSV;
--weighted-centroid centres the crop on the probability-weighted CA centroid. Existing crops are skipped.

  python -m siteseer.docking.crops --csv dataset/docking/goliath_test_dock.csv --structure-root data/structures \\
      --probe-checkpoint checkpoints/main_stage2_mcsa/best.pt --output-dir data/docking/predicted_cropped \\
      --residues-out-csv data/docking/predicted_residues.csv [--weighted-centroid] [--shard-index K --num-shards N]
"""
import argparse
import csv
import os

import pandas as pd
import torch

from siteseer.docking.crop import crop_protein
from siteseer.embeddings.embedders import MAX_SEQ_LEN, PRESETS, get_embedder
from siteseer.models.probe import load_probe
from siteseer.structural.graph import resolve_pdb_path


def predict_residues(embedder, probe, sequence, threshold=0.5):
    """-> (1-based residue numbers, {residue: probability}) for p > threshold, else the most probable residue."""
    sequence = sequence[:MAX_SEQ_LEN]
    with torch.no_grad():
        probs = torch.sigmoid(probe(embedder.embed(sequence).to(next(probe.parameters()).device)))
    idx = (probs > threshold).nonzero(as_tuple=True)[0].tolist() or [int(torch.argmax(probs))]
    return [i + 1 for i in idx], {i + 1: probs[i].item() for i in idx}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--structure-root", default="data/structures")
    ap.add_argument("--probe-checkpoint", required=True)
    ap.add_argument("--embedder", default="esm2_650m_l30", choices=sorted(PRESETS))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--residues-out-csv", default=None)
    ap.add_argument("--weighted-centroid", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--radius", type=float, default=15.0)
    ap.add_argument("--shard-index", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.output_dir, exist_ok=True)
    embedder = get_embedder(args.embedder).load(device)
    probe, _ = load_probe(args.probe_checkpoint, device)

    df = pd.read_csv(args.csv).drop_duplicates(subset="alphafolddb-id")
    if args.shard_index is not None:
        df = df.iloc[args.shard_index::args.num_shards]
    done = set()
    writer = None
    if args.residues_out_csv:
        if os.path.exists(args.residues_out_csv):
            done = set(pd.read_csv(args.residues_out_csv, dtype={"uid": str})["uid"])
        fh = open(args.residues_out_csv, "a", newline="")
        writer = csv.writer(fh)
        if not done:
            writer.writerow(["uid", "residues"])
    counts = {"ok": 0, "unresolved": 0, "crop_failed": 0}
    for _, row in df.iterrows():
        raw = str(row["alphafolddb-id"]).strip()
        uid = raw.split("-")[1] if raw.startswith("AF-") else raw
        out_pdb = os.path.join(args.output_dir, f"{uid}.pdb")
        if os.path.exists(out_pdb) and (writer is None or uid in done):
            counts["ok"] += 1
            continue
        pdb = resolve_pdb_path(row, args.structure_root)
        if pdb is None:
            counts["unresolved"] += 1
            continue
        residues, probs = predict_residues(embedder, probe, row["aa_sequence"], args.threshold)
        if writer and uid not in done:
            writer.writerow([uid, " ".join(map(str, residues))])
            fh.flush()
        try:
            crop_protein(pdb, out_pdb, residues, args.radius, probs if args.weighted_centroid else None)
            counts["ok"] += 1
        except Exception as e:
            counts["crop_failed"] += 1
            print(f"[-] {uid}: crop failed ({e})")
    print(f"[+] {counts} of {len(df)}")


if __name__ == "__main__":
    main()
