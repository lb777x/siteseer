"""Step 16: M-CSA proteins unseen by Squidly, for the comparison with Squidly's released models on M-CSA.

Squidly's models were trained on its CataloDB training list (5,355 proteins), which shares accessions with M-CSA.
Highest identity of each of the 962 M-CSA proteins to that list (mmseqs easy-search -s 7.5, best hit with query
coverage >= 0.5); kept: < 30% (primary) or < 50% identity, and not the same accession. Also writes all M-CSA
sequences as FASTA for the Squidly run.

  python pipeline/16_squidly_mcsa.py
"""
import argparse
import os
import tempfile

import pandas as pd

from common import EXTERNAL, MCSA_CSV, PIPELINE, run, tool

ID = "alphafolddb-id"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--squidly-train", default=f"{EXTERNAL}/squidly/Low30_mmseq_ID_exp_subset_train.csv")
    ap.add_argument("--out-dir", default="dataset/squidly")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(PIPELINE, exist_ok=True)

    seqs = {}
    for s in ("train", "valid", "test"):
        d = pd.read_csv(MCSA_CSV[s])
        for a, q in zip(d[ID].astype(str).str.strip(), d["aa_sequence"]):
            seqs.setdefault(a, q)
    if any(len(q) > 1022 for q in seqs.values()):
        raise SystemExit("M-CSA sequences > 1022 residues would need the same truncation on both sides")
    fasta = f"{args.out_dir}/mcsa_all.fasta"
    with open(fasta, "w") as f:
        for a in sorted(seqs):
            f.write(f">{a}\n{seqs[a]}\n")
    sq = pd.read_csv(args.squidly_train)
    with tempfile.TemporaryDirectory() as w:
        with open(f"{w}/t.fasta", "w") as f:
            for a, q in zip(sq["Entry"], sq["Sequence"]):
                f.write(f">{a}\n{q}\n")
        run([tool("mmseqs"), "easy-search", fasta, f"{w}/t.fasta", f"{w}/h.m8", f"{w}/tmp", "-s", "7.5",
             "--format-output", "query,target,fident,qcov,tcov,evalue", "--threads", args.threads])
        h = pd.read_csv(f"{w}/h.m8", sep="\t", names=["query", "target", "fident", "qcov", "tcov", "evalue"])
    h = h[h.qcov >= 0.5].sort_values("fident", ascending=False).drop_duplicates("query")
    out = pd.DataFrame({ID: sorted(seqs)}).merge(
        h.rename(columns={"query": ID, "target": "best_squidly_train_hit"})[[ID, "best_squidly_train_hit", "fident", "qcov"]],
        on=ID, how="left")
    out["max_identity"] = out.fident.fillna(0.0)
    out["same_accession_in_squidly_train"] = out[ID].isin(set(sq["Entry"]))
    out.drop(columns="fident").to_csv(f"{PIPELINE}/mcsa_identity_to_squidly_train.csv", index=False)
    for cut, name in ((0.3, "lt30"), (0.5, "lt50")):
        ids = out[(out.max_identity < cut) & ~out.same_accession_in_squidly_train][ID]
        with open(f"{args.out_dir}/mcsa_ids_{name}.txt", "w") as f:
            f.write("\n".join(ids) + "\n")
        print(f"[*] < {int(cut * 100)}% identity to Squidly's training set: {len(ids)} of {len(out)} proteins")


if __name__ == "__main__":
    main()
