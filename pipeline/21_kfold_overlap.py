"""Step 21: M-CSA 5-fold proteins that also occur in stage-1 training data, excluded from 5-fold scoring.

Step 09 removes M-CSA proteins from stage 1 by exact id and >= 90% identity over >= 80% of both sequences. That misses
BioLiP entries that name several chains (e.g. 'P00720,P07550', T4 lysozyme fused to a receptor) and single domains of
longer entries. A 5-fold protein is listed here if a stage-1 train or valid entry (fix1, which contains fix2) carries
its id, or matches it at >= 90% identity over >= 50% of either sequence. The proteins stay in the folds for training;
only their test predictions are left out of every 5-fold result.

  python pipeline/21_kfold_overlap.py
"""
import argparse
import os

import pandas as pd

from common import LABELS, PIPELINE, run, tool, write_fasta

ID = "alphafolddb-id"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kfold-dir", default="dataset/mcsa_fine_tune/kfold")
    ap.add_argument("--stage1-csv", default=f"{LABELS}/cached_dataframe_fully_excl_fix1.csv")
    ap.add_argument("--work-dir", default=f"{PIPELINE}/kfold_overlap")
    ap.add_argument("--min-identity", type=float, default=0.9)
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="dataset/mcsa_fine_tune/kfold/excluded_overlap_ids.txt")
    args = ap.parse_args()

    q = pd.concat([pd.read_csv(os.path.join(args.kfold_dir, f"fold_{i}_test.csv")) for i in range(5)])
    q[ID] = q[ID].astype(str).str.strip()
    s1 = pd.read_csv(args.stage1_csv, usecols=[ID, "aa_sequence", "dataset_flag"], low_memory=False)
    s1 = s1[s1.dataset_flag.isin(["train", "valid"])].reset_index(drop=True)
    print(f"[*] {len(q)} 5-fold proteins vs {len(s1)} stage-1 train/valid entries", flush=True)

    ids = set(q[ID])
    by_id = {}
    for entry in s1[ID].astype(str):
        for token in entry.split(","):
            if token.strip() in ids:
                by_id.setdefault(token.strip(), entry)

    w = args.work_dir
    write_fasta(q[ID], q["aa_sequence"], os.path.join(w, "q.fasta"))
    write_fasta(range(len(s1)), s1["aa_sequence"], os.path.join(w, "t.fasta"))
    out = os.path.join(w, "hits.m8")
    run([tool("mmseqs"), "easy-search", os.path.join(w, "q.fasta"), os.path.join(w, "t.fasta"), out,
         os.path.join(w, "tmp"), "-s", "7.5", "--format-output", "query,target,fident,qcov,tcov",
         "--threads", args.threads, "-v", "1"])
    h = pd.read_csv(out, sep="\t", names=["q", "t", "fident", "qcov", "tcov"])
    h = h[(h.fident >= args.min_identity) & ((h.qcov >= args.min_coverage) | (h.tcov >= args.min_coverage))]
    h = h.sort_values("fident", ascending=False).drop_duplicates("q")

    rows = [{"id": i, "reason": "shared id", "stage1_entry": e} for i, e in by_id.items()]
    rows += [{"id": r.q, "reason": f"identity {r.fident:.3f}, qcov {r.qcov:.2f}, tcov {r.tcov:.2f}",
              "stage1_entry": s1[ID].iloc[r.t]} for r in h.itertuples() if r.q not in by_id]
    table = pd.DataFrame(rows).sort_values("id")
    table.to_csv(os.path.join(w, "excluded.tsv"), sep="\t", index=False)
    with open(args.out, "w") as f:
        f.write("\n".join(table["id"]) + "\n")
    print(f"[+] {args.out}: {len(table)} proteins ({len(by_id)} by shared id); reasons in {w}/excluded.tsv")


if __name__ == "__main__":
    main()
