"""Dock one condition of the benchmark with DiffDock (gcorso/DiffDock @ 85c49b6, model v1.1): 20 inference steps
(18 actual, no final-step noise), 10 poses per complex. Runs inside the DiffDock environment and checkout
(--diffdock-home) and imports nothing else from this package.

global   the full AlphaFold model (AF-<uid>-F1-*.pdb in --raw-dir)
cropped  <uid>.pdb from --structure-dir (crops.py)
Per complex: <out-dir>/<uid><suffix>/ with DiffDock's ranked poses; wall-clock seconds and success are appended to
--tracker-csv. Complexes with a scored pose or a failure log are skipped (DOCK_RETRY_FAILED=1 retries failures).

  python -m siteseer.docking.run_diffdock --csv SET --condition global --raw-dir data/structures \\
      --complex-suffix _global --out-dir results/docking/runs --tracker-csv results/docking/tracker.csv \\
      --smiles-json dataset/docking/ligand_smiles.json --diffdock-home external/DiffDock
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time

import pandas as pd

SCORED = re.compile(r"rank1_confidence")


def has_valid_output(folder):
    return os.path.isdir(folder) and any(SCORED.search(f) for f in os.listdir(folder))


def run_diffdock(protein, smiles, name, out_dir, home):
    cmd = [sys.executable, "-u", "-m", "inference", "--protein_path", protein, "--ligand", smiles,
           "--complex_name", name, "--out_dir", out_dir, "--inference_steps", "20", "--samples_per_complex", "10",
           "--batch_size", "1", "--actual_steps", "18", "--no_final_step_noise",
           "--model_dir", "workdir/v1.1/score_model", "--confidence_model_dir", "workdir/v1.1/confidence_model"]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=home, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode == 0, time.perf_counter() - t0, p.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--condition", required=True, choices=["global", "cropped"])
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--structure-dir", default=None)
    ap.add_argument("--complex-suffix", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tracker-csv", required=True)
    ap.add_argument("--smiles-json", required=True)
    ap.add_argument("--diffdock-home", default=os.environ.get("DIFFDOCK_HOME", "external/DiffDock"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard-index", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    smiles = json.load(open(args.smiles_json))
    df = pd.read_csv(args.csv).drop_duplicates(subset="alphafolddb-id")
    if args.limit is not None:
        df = df.head(args.limit)
    if args.shard_index is not None:
        df = df.iloc[args.shard_index::args.num_shards]
    counts = {"ok": 0, "failed": 0, "skipped": 0}
    for _, row in df.iterrows():
        raw = str(row["alphafolddb-id"]).strip()
        uid = raw.split("-")[1] if raw.startswith("AF-") else raw
        name = f"{uid}{args.complex_suffix}"
        target = os.path.join(out_dir, name)
        failure_log = os.path.join(out_dir, "_failure_logs", f"{name}.log")
        if has_valid_output(target) or (os.path.exists(failure_log) and not os.environ.get("DOCK_RETRY_FAILED")):
            counts["skipped"] += 1
            continue
        if args.condition == "cropped":
            protein = os.path.join(args.structure_dir, f"{uid}.pdb")
            protein = protein if os.path.exists(protein) else None
        else:
            hits = glob.glob(os.path.join(args.raw_dir, f"AF-{uid}-F1-*.pdb"))
            protein = hits[0] if hits else None
        if protein is None or not smiles.get(uid):
            counts["skipped"] += 1
            continue
        ok, seconds, log = run_diffdock(os.path.abspath(protein), smiles[uid], name, out_dir, args.diffdock_home)
        ok = ok and has_valid_output(target)
        new = not os.path.exists(args.tracker_csv)
        with open(args.tracker_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["uid", "condition", "complex_name", "success", "seconds"])
            if new:
                w.writeheader()
            w.writerow({"uid": uid, "condition": name, "complex_name": name, "success": ok, "seconds": f"{seconds:.2f}"})
        counts["ok" if ok else "failed"] += 1
        if not ok:
            os.makedirs(os.path.dirname(failure_log), exist_ok=True)
            open(failure_log, "w").write(log)
    print(f"[+] {args.complex_suffix}: {counts} of {len(df)}")


if __name__ == "__main__":
    main()
