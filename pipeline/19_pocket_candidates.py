"""Step 19: pocket candidates among set-A non-enzymes (no enzyme and no binding-site annotation in UniProt, not in
our training or test data), the truth for the comparison with P2Rank.

U  40% cluster representatives with >= 30% identity to a training enzyme (from step 17)
V  proteins with a small-molecule or metal contact in a crystal structure (BioLiP, non-redundant), mapped onto the
   UniProt sequence (siteseer/negatives/biolip.py)
One row per protein (set U, V or UV), columns as the other set CSVs so the scoring reads them.

  python pipeline/19_pocket_candidates.py
"""
import argparse

import numpy as np
import pandas as pd

from common import EXTERNAL, PIPELINE, tool
from siteseer.negatives import biolip
from siteseer.negatives import uniprot as u

WORK = f"{PIPELINE}/negatives"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="dataset/negatives/pocket_candidates.csv")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    a = u.load_tsv(f"{EXTERNAL}/uniprot/uniprot_A.tsv.gz")
    a["set"] = "A"
    a = a[~a["accession"].isin(u.known_ids())]
    cl = pd.read_csv(f"{WORK}/clusters_A.tsv", sep="\t")
    rep_of = dict(zip(cl["member"], cl["representative"]))
    reps = pd.read_csv(f"{WORK}/representatives.csv", low_memory=False)
    reps = reps.merge(a[["accession", "sequence"]], on="accession", how="left")
    un = reps[(reps["set"] == "A") & ~reps["in_training_data"] & (reps["max_identity"] >= 30)].copy()

    bl = biolip.read()
    bl = bl[bl["uniprot"].isin(set(a["accession"])) & ~bl["lig"].isin(biolip.NOT_A_SITE | biolip.ADDITIVES | biolip.GLYCANS)]
    seq_of = a.set_index("accession")["sequence"]
    rows = []
    for acc, g in bl.groupby("uniprot"):
        pos = set()
        for r in g.itertuples():
            pos |= biolip.mapped(str(r.bres).split(), biolip.position_map(r.seq, seq_of[acc]), seq_of[acc])
        if pos:
            rows.append({"accession": acc, "biolip_residues": ";".join(map(str, sorted(pos))),
                         "biolip_ligands": ";".join(sorted(set(g["lig"]))), "biolip_pdb": ";".join(sorted(set(g["pdb"])))})
    v = a[a["accession"].isin({r["accession"] for r in rows})].merge(pd.DataFrame(rows), on="accession")
    v["family"] = v["accession"].map(rep_of)
    v["cluster_size"] = v["family"].map(cl.groupby("representative").size())
    v = v.merge(u.identity_to_pools(v[["accession", "sequence"]], tool("mmseqs"), args.threads), on="accession",
                how="left")
    v["org_group"] = v["lineage"].map(u.org_group)
    v["kw_category"] = v["keywords"].map(u.kw_category)
    v["uncharacterized"] = v["protein_name"].astype(str).str.contains(u.UNCHARACTERIZED)

    both = pd.concat([v, un[~un["accession"].isin(v["accession"])]], ignore_index=True)
    both["in_U"] = both["accession"].isin(un["accession"])
    both["in_V"] = both["accession"].isin(v["accession"])
    both["set"] = np.where(both["in_U"] & both["in_V"], "UV", np.where(both["in_V"], "V", "U"))
    both["origin_set"] = "A"
    both["split_role"] = "report"
    both["homology_type"] = ""
    both["family"] = both["accession"].map(rep_of)
    for c in ("biolip_residues", "biolip_ligands", "biolip_pdb"):
        both[c] = both[c].fillna("")
    cols = ["accession", "set", "in_U", "in_V", "origin_set", "split_role", "protein_name", "organism", "org_group",
            "kw_category", "keywords", "length", "family", "cluster_size", "uncharacterized", "max_identity",
            "max_identity_pool", "identity_band"] + \
           [c for n in u.POOLS for c in (f"identity_{n}", f"best_hit_{n}", f"qcov_{n}")] + \
           ["homology_type", "biolip_residues", "biolip_ligands", "biolip_pdb", "sequence"]
    both = both[cols].sort_values(["set", "accession"])
    both.to_csv(args.out, index=False)
    print(f"[+] {args.out}: U {int(both['in_U'].sum())}, V {int(both['in_V'].sum())}, both "
          f"{int((both['in_U'] & both['in_V']).sum())}")


if __name__ == "__main__":
    main()
