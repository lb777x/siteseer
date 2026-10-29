"""Does the docked ligand land at the annotated site? Per protein and condition (rank-1 pose):
hit = a heavy ligand atom within --contact A (4) of a heavy atom of an annotated site residue; d = distance of the
ligand centroid to the CA centroid of the site. From the 10 poses of the global run, without new docking:
global_rerank = the pose closest to the predicted site's CA centroid, global_oracle = any pose hits (upper bound).
Poses are in the frame of the full structure (crops keep coordinates). Figure 3.17 is drawn from this table
(figures/plot_docking_site_hits.py --csv).

  python -m siteseer.docking.site_accuracy --set dataset/docking/goliath_test_dock.csv --raw-dir data/structures \\
      --out-dir results/docking/runs --pred-residues "data/docking/predicted_residues*.csv" \\
      --out-csv results/docking/site_accuracy.csv
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

from siteseer.data.labels import parse_site_labels, row_label_source

CONDS = ["_global", "_predicted_cropped", "_predicted_cropped_weighted"]
RANK = re.compile(r"rank(\d+)_confidence([+-]?\d+\.\d+)\.sdf")


def read_sdf(path):
    """Heavy-atom coordinates of a V2000 SDF pose."""
    lines = open(path).read().splitlines()
    n = int(lines[3][:3])
    return np.array([[float(line[0:10]), float(line[10:20]), float(line[20:30])]
                     for line in lines[4:4 + n] if line[31:34].strip() != "H"])


def read_pdb(path):
    """{residue number: (heavy-atom coords, CA coord or None)} of the first chain."""
    res, chain0 = {}, None
    for line in open(path):
        if not line.startswith("ATOM"):
            continue
        chain0 = chain0 or line[21]
        if line[21] != chain0 or line[76:78].strip() == "H":
            continue
        r = int(line[22:26])
        xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        atoms, ca = res.get(r, ([], None))
        atoms.append(xyz)
        res[r] = (atoms, xyz if line[12:16].strip() == "CA" else ca)
    return {r: (np.array(a), None if c is None else np.array(c)) for r, (a, c) in res.items()}


def poses(folder):
    if not os.path.isdir(folder):
        return []
    out = [(int(m.group(1)), float(m.group(2)), read_sdf(os.path.join(folder, f)))
           for f in os.listdir(folder) for m in [RANK.fullmatch(f)] if m]
    return sorted(out, key=lambda t: t[0])


def score(lig, site_atoms, site_centroid, contact):
    d = np.sqrt(((lig[:, None, :] - site_atoms[None, :, :]) ** 2).sum(-1)).min()
    return bool(d <= contact), float(np.linalg.norm(lig.mean(0) - site_centroid))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pred-residues", required=True, help="glob of (uid, residues) CSVs from crops.py")
    ap.add_argument("--contact", type=float, default=4.0)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()
    pred = {}
    for f in glob.glob(args.pred_residues):
        t = pd.read_csv(f, dtype={"uid": str})
        pred.update({u: [int(x) for x in str(r).split()] for u, r in zip(t["uid"], t["residues"])
                     if isinstance(r, str) and r.strip()})
    rows, skipped = [], {"no_structure": 0, "no_labels": 0, "labels_not_in_structure": 0}
    for _, row in pd.read_csv(args.set).iterrows():
        raw = str(row["alphafolddb-id"]).strip()
        uid = raw.split("-")[1] if raw.startswith("AF-") else raw
        hits = glob.glob(os.path.join(args.raw_dir, f"AF-{uid}-F1-*.pdb"))
        if len(hits) != 1:
            skipped["no_structure"] += 1
            continue
        gt = parse_site_labels(row["site_labels"], row_label_source(row))
        if not gt:
            skipped["no_labels"] += 1
            continue
        res = read_pdb(hits[0])
        gt = [r for r in gt if r in res]
        if not gt:
            skipped["labels_not_in_structure"] += 1
            continue
        site_atoms = np.concatenate([res[r][0] for r in gt])
        site_c = np.mean([res[r][1] if res[r][1] is not None else res[r][0].mean(0) for r in gt], axis=0)
        rec, pc = {"uid": uid, "n_site_residues": len(gt)}, None
        pr = [r for r in pred.get(uid, []) if r in res and res[r][1] is not None]
        if pr:
            pc = np.mean([res[r][1] for r in pr], axis=0)
            rec["d_compass_site"] = float(np.linalg.norm(pc - site_c))
        for cond in CONDS:
            ps = poses(os.path.join(args.out_dir, f"{uid}{cond}"))
            if not ps:
                continue
            rec[f"{cond}_hit"], rec[f"{cond}_d"] = score(ps[0][2], site_atoms, site_c, args.contact)
            rec[f"{cond}_conf"] = ps[0][1]
            if cond == "_global":
                rec["_global_oracle_hit"] = any(score(p[2], site_atoms, site_c, args.contact)[0] for p in ps)
                rec["_global_n_poses"] = len(ps)
                if pc is not None:
                    best = min(ps, key=lambda p: np.linalg.norm(p[2].mean(0) - pc))
                    rec["_global_rerank_hit"], rec["_global_rerank_d"] = score(best[2], site_atoms, site_c, args.contact)
                    rec["_global_rerank_rank"] = best[0]
        rows.append(rec)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"[+] {len(rows)} proteins -> {args.out_csv}; skipped {skipped}")


if __name__ == "__main__":
    main()
