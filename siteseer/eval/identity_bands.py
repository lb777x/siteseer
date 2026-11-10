"""How does the main model do on M-CSA proteins far from its training data? Out-of-fold 5-fold predictions, per protein
the highest sequence identity (MMseqs2 -s 7.5, best hit with query coverage >= 0.5, 0 without a hit) to
  pool  the stage-1 training proteins (fix2; stage 1.5 is a subset), and
  fold  the protein's own M-CSA training fold.
Per identity band the macro AUPRC with a 95% percentile bootstrap CI over proteins (2,000 draws, seed 42), and the
Spearman correlation of per-protein AUPRC with identity. Proteins of the overlap list (pipeline step 21) are left out.

  python -m siteseer.eval.identity_bands        (MMseqs2 from PATH or MMSEQS_BIN)
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from siteseer.eval.metrics import binary_metrics

ID = "alphafolddb-id"
KFOLD = "dataset/mcsa_fine_tune/kfold"
BANDS = {"pool": [(0.0, 0.3, "< 30%"), (0.3, 0.5, "30–50%"), (0.5, 0.7, "50–70%"), (0.7, 1.01, "≥ 70%")],
         "fold": [(0.0, 0.3, "< 30%"), (0.3, 1.01, "≥ 30%")]}


def write_fasta(path, ids, seqs):
    with open(path, "w") as f:
        f.writelines(f">{i}\n{s}\n" for i, s in zip(ids, seqs))


def max_identity(q: pd.DataFrame, targets: list, mmseqs: str, threads: int) -> pd.Series:
    """Highest identity of every query to the target sequences (best hit with query coverage >= 0.5)."""
    work = tempfile.mkdtemp(prefix="identity_bands_")
    try:
        write_fasta(f"{work}/q.fasta", q["id"], q["seq"])
        write_fasta(f"{work}/t.fasta", range(len(targets)), targets)
        subprocess.run([mmseqs, "easy-search", f"{work}/q.fasta", f"{work}/t.fasta", f"{work}/hits.tsv", f"{work}/tmp",
                        "-s", "7.5", "--max-seqs", "1000", "--format-output", "query,target,fident,qcov",
                        "--threads", str(threads), "-v", "1"], check=True)
        h = pd.read_csv(f"{work}/hits.tsv", sep="\t", header=None, names=["id", "target", "fident", "qcov"])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    best = h[h["qcov"] >= 0.5].groupby("id")["fident"].max()
    return q["id"].map(best).fillna(0.0)


def bands(x: np.ndarray, ident: np.ndarray, edges: list, n_bootstrap: int, seed: int) -> list:
    out = []
    for lo, hi, label in edges:
        v = x[(ident >= lo) & (ident < hi)]
        rng = np.random.default_rng(seed)
        boot = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_bootstrap)] if len(v) else [np.nan]
        out.append({"band": label, "n": int(len(v)), "macro_auprc": float(v.mean()) if len(v) else None,
                    "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))] if len(v) else None})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", nargs=5, default=[f"results/predictions/main_kfold{i}_test.pt" for i in range(5)])
    ap.add_argument("--stage1-csv", default="dataset/label_diffusion/cached_dataframe_fully_excl_fix2.csv")
    ap.add_argument("--exclude-ids", default=f"{KFOLD}/excluded_overlap_ids.txt")
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out-json", default="results/reports/identity_bands.json")
    ap.add_argument("--out-csv", default="results/reports/identity_bands_per_protein.csv")
    args = ap.parse_args()
    mmseqs = os.environ.get("MMSEQS_BIN") or shutil.which("mmseqs") or "mmseqs"
    exclude = {l.strip() for l in open(args.exclude_ids) if l.strip()}

    auprc = {}
    for path in args.predictions:
        d = torch.load(path, weights_only=False)
        for i, lo, la in zip(d["protein_ids"], d["logits"], d["labels"]):
            auprc[str(i).strip()] = binary_metrics(lo, la)["auprc"]
    folds = []
    for k in range(5):
        t = pd.read_csv(f"{KFOLD}/fold_{k}_test.csv")
        folds.append(pd.DataFrame({"id": t[ID].astype(str).str.strip(), "seq": t["aa_sequence"], "fold": k}))
    q = pd.concat(folds, ignore_index=True)
    q = q[~q["id"].isin(exclude)].reset_index(drop=True)
    print(f"[*] {len(q)} proteins ({len(exclude)} excluded by {args.exclude_ids})", flush=True)

    s1 = pd.read_csv(args.stage1_csv, usecols=[ID, "aa_sequence", "dataset_flag"], low_memory=False)
    pool = s1.loc[s1["dataset_flag"] == "train", "aa_sequence"].astype(str).drop_duplicates().tolist()
    q["identity_pool"] = max_identity(q, pool, mmseqs, args.threads).values
    q["identity_fold"] = 0.0
    for k in range(5):
        tr = pd.read_csv(f"{KFOLD}/fold_{k}_train.csv")["aa_sequence"].astype(str).tolist()
        m = q["fold"] == k
        q.loc[m, "identity_fold"] = max_identity(q[m], tr, mmseqs, args.threads).values
    q["auprc"] = q["id"].map(auprc)
    if q["auprc"].isna().any():
        raise SystemExit(f"{int(q['auprc'].isna().sum())} proteins without predictions")

    x = q["auprc"].to_numpy()
    out = {"n_proteins": len(q), "n_excluded": len(exclude), "macro_auprc_all": float(x.mean()),
           "n_bootstrap": args.n_bootstrap, "seed": args.seed}
    for name in ("pool", "fold"):
        ident = q[f"identity_{name}"].to_numpy()
        out[name] = {"bands": bands(x, ident, BANDS[name], args.n_bootstrap, args.seed),
                     "spearman": float(spearmanr(ident, x).statistic), "n_no_hit": int((ident == 0).sum())}
        print(f"--- identity to {name} (Spearman {out[name]['spearman']:.3f}, no hit {out[name]['n_no_hit']}) ---")
        for b in out[name]["bands"]:
            print(f"  {b['band']:7s} n = {b['n']:4d}  {b['macro_auprc']:.3f} [{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}]")
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    q.drop(columns="seq").to_csv(args.out_csv, index=False)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"[+] {args.out_json}, {args.out_csv}")


if __name__ == "__main__":
    main()
