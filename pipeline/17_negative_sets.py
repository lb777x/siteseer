"""Step 17: non-enzyme sets A, B and C for the negative control.

UniProt (data/external/uniprot/uniprot_{A,B}.tsv.gz, query of 2026-10-05; --fetch queries again): Swiss-Prot,
no fragment, 50-1022 residues, AlphaFold model, and no sign of catalysis (no EC number, catalytic activity, ACT_SITE
or SITE feature, not GO:0003824 or its children). A: also no binding-site feature; B: with binding sites (the
training labels count binding sites as positive, so an alarm is expected there).
One protein per 40% MMseqs2 cluster; proteins of unknown function and proteins in our training or test data are not
drawn. A: 1,000 and B: 300, stratified by organism group x keyword category (seed 42). C: every remaining
representative with >= 50% identity to a training enzyme. 30% of each set (seed 42; C seed 43) validate the
protein-level threshold.

  python pipeline/17_negative_sets.py [--fetch]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from common import EXTERNAL, PIPELINE, tool
from siteseer.negatives import uniprot as u

RAW = f"{EXTERNAL}/uniprot"
WORK = f"{PIPELINE}/negatives"   # clusters and representatives, reused by step 19
N_SAMPLE = {"A": 1000, "B": 300}


def target_coverage(df, mmseqs, threads):
    """Coverage of the best training hit (tcov) in the pool with the highest identity: full-length homologue
    (possible pseudoenzyme) or similar to part of a larger enzyme only."""
    rows = []
    for name in u.POOLS:
        sub = df[df["max_identity_pool"] == name]
        if len(sub):
            h = u.best_hits(u.search_pool(name, sub, mmseqs, threads, columns="query,target,fident,qcov,tcov"))
            rows.append(h[["accession", "tcov"]].rename(columns={"tcov": "target_cov"}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["accession", "target_cov"])


def add_binding(df):
    bp = df["ft_binding"].map(u.binding_info)
    df["binding_positions"] = [";".join(map(str, p)) for p, _ in bp]
    df["binding_ligands"] = [";".join(lig) for _, lig in bp]
    df["n_binding_residues"] = [len(p) for p, _ in bp]
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true", help="query UniProt again instead of using the frozen copies")
    ap.add_argument("--out", default="dataset/negatives/negatives_abc.csv")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    mmseqs = tool("mmseqs")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    if args.fetch:
        os.makedirs(RAW, exist_ok=True)
        for s, term in u.SETS.items():
            u.stream(" AND ".join(u.COMMON + [term]), u.FIELDS, f"{RAW}/uniprot_{s}.tsv.gz")

    parts = []
    for s in u.SETS:
        d = u.load_tsv(f"{RAW}/uniprot_{s}.tsv.gz")
        d["set"] = s
        cl = u.cluster40(d, mmseqs, args.threads)
        cl.to_csv(f"{WORK}/clusters_{s}.tsv", sep="\t", index=False)
        size = cl.groupby("representative").size().rename("cluster_size")
        parts.append(d[d["accession"].isin(size.index)].merge(size, left_on="accession", right_index=True))
        print(f"[*] set {s}: {len(d)} proteins, {len(size)} clusters (40%)")
    d = pd.concat(parts, ignore_index=True)
    ident = u.identity_to_pools(d[["accession", "sequence"]], mmseqs, args.threads)
    d = d.merge(ident, on="accession", how="left")
    d["org_group"] = d["lineage"].map(u.org_group)
    d["kw_category"] = d["keywords"].map(u.kw_category)
    d["uncharacterized"] = d["protein_name"].astype(str).str.contains(u.UNCHARACTERIZED)
    d["in_training_data"] = d["accession"].isin(u.known_ids())
    d.drop(columns=["sequence"]).to_csv(f"{WORK}/representatives.csv", index=False)

    rng = np.random.default_rng(u.SEED)
    picked = [u.stratified(d[(d["set"] == s) & ~d["uncharacterized"] & ~d["in_training_data"]], n, rng)
              for s, n in N_SAMPLE.items()]
    out = add_binding(pd.concat(picked, ignore_index=True))
    out["split_role"] = "report"
    for s in u.SETS:
        idx = out.index[out["set"] == s]
        out.loc[rng.choice(idx, size=int(round(u.VALID_FRACTION * len(idx))), replace=False), "split_role"] = "valid"
    out["origin_set"] = out["set"]

    rng_c = np.random.default_rng(u.SEED + 1)
    c = d[~d["uncharacterized"] & ~d["in_training_data"] & (d["max_identity"] >= 50)
          & ~d["accession"].isin(out["accession"])].copy()
    c["origin_set"], c["set"], c["split_role"] = c["set"], "C", "report"
    c = add_binding(c)
    if len(c):
        c.loc[rng_c.choice(c.index, size=int(round(u.VALID_FRACTION * len(c))), replace=False), "split_role"] = "valid"
    out = pd.concat([out, c], ignore_index=True)

    out = out.merge(target_coverage(out[out["max_identity"] >= 50], mmseqs, args.threads), on="accession", how="left")
    out["homology_type"] = ""
    hi = out["max_identity"] >= 50
    full = (out[[f"qcov_{n}" for n in u.POOLS]].max(axis=1) >= 0.8) & (out["target_cov"] >= 0.8)
    out.loc[hi & full, "homology_type"] = "full length"
    out.loc[hi & ~full & (out["target_cov"] < 0.5), "homology_type"] = "part of a larger enzyme"
    out.loc[hi & ~full & (out["target_cov"] >= 0.5), "homology_type"] = "partial"
    cols = ["accession", "set", "origin_set", "split_role", "entry_name", "protein_name", "organism", "organism_id",
            "org_group", "kw_category", "keywords", "length", "alphafold_id", "cluster_size", "max_identity",
            "max_identity_pool", "identity_band", "target_cov", "homology_type"] + \
           [c for n in u.POOLS for c in (f"identity_{n}", f"best_hit_{n}", f"qcov_{n}")] + \
           ["n_binding_residues", "binding_positions", "binding_ligands", "protein_existence", "annotation_score",
            "sequence"]
    out = out.sort_values(["set", "accession"])[cols]
    out.to_csv(args.out, index=False)
    rep = {"by_set_and_role": {f"{k[0]}/{k[1]}": int(v) for k, v in out.groupby(["set", "split_role"]).size().items()},
           "by_set_and_band": {f"{k[0]}/{k[1]}": int(v) for k, v in out.groupby(["set", "identity_band"]).size().items()}}
    with open(f"{WORK}/negatives_abc_report.json", "w") as f:
        json.dump(rep, f, indent=1)
    print(f"[+] {len(out)} proteins -> {args.out}")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
