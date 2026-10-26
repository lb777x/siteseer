"""Step 18: set G, confident non-enzymes without a ligand pocket, for the negative control.

UniProt (data/external/uniprot/uniprot_gold.tsv.gz, query of 2026-10-05; --fetch queries again): as set A plus
protein-level evidence and not GO:0000166 / GO:0046872 / GO:0048037 (nucleotide, metal ion, cofactor binding).
Local filters, stored as columns (True = excluded), gold_pass = none applies: ligand keywords, enzyme-class
keywords, uncertain names (inactive, -like, putative, ...), a Pfam family in which >= 30% of Swiss-Prot members (>= 5)
carry an EC number (swissprot_pfam_ec.tsv.gz), in our training or test data, already in sets A/B/C.
40% MMseqs2 clusters; 30% of the clusters validate the threshold (seed 42). The scored subset: 3,000 gold_pass
cluster representatives, stratified by organism group x keyword category (seed 44).

  python pipeline/18_gold_set.py [--fetch]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from common import EXTERNAL, PIPELINE, tool
from siteseer.negatives import uniprot as u

RAW = f"{EXTERNAL}/uniprot"
N_SUBSET = 3000
PFAM_ENZ_SHARE = 0.3
PFAM_MIN_MEMBERS = 5
GOLD_TERMS = ["existence:1", "NOT go:0000166", "NOT go:0046872", "NOT go:0048037"]
LIGAND_KW = {"ATP-binding", "GTP-binding", "Nucleotide-binding", "Metal-binding", "Zinc", "Calcium", "Magnesium",
             "Iron", "Copper", "Manganese", "Nickel", "Cobalt", "Molybdenum", "Tungsten", "Heme", "FAD", "FMN",
             "NAD", "NADP", "Pyridoxal phosphate", "Flavoprotein", "Iron-sulfur", "2Fe-2S", "3Fe-4S", "4Fe-4S",
             "Chlorophyll", "Bacteriochlorophyll", "Thiamine pyrophosphate", "Biotin", "Lipoyl", "Selenium",
             "Cobalamin", "Chromophore", "Retinal protein", "Heme-binding", "S-adenosyl-L-methionine"}
ENZYME_KW = (r"Hydrolase|transferase|Kinase|kinase|Oxidoreductase|Lyase|Isomerase|Ligase|[Pp]rotease|Glycosidase|"
             r"nuclease|Helicase|Translocase|peptidase|Peroxidase|oxygenase|Decarboxylase|Esterase|Lipase|"
             r"Phosphatase|Allosteric enzyme|Multifunctional enzyme")
UNSURE_NAME = (r"(?i)[Ii]nactive|-like|like protein|[Hh]omolog|[Pp]seudo|[Pp]utative|[Pp]robable|[Pp]redicted|"
               r"uncharacteri[sz]ed|hypothetical|\bUPF\d|\bDUF\d")
FLAG_COLS = ["f_ligand_kw", "f_enzyme_kw", "f_unsure_name", "f_enzyme_pfam", "f_in_training", "f_in_abc"]


def enzyme_pfam_families(path):
    d = pd.read_csv(path, sep="\t", compression="gzip", dtype=str).fillna("")
    d["has_ec"] = d["EC number"].str.len() > 0
    e = d.assign(pfam=d["Pfam"].str.strip(";").str.split(";"))[["has_ec", "pfam"]].explode("pfam")
    t = e[e["pfam"].str.len() > 0].groupby("pfam")["has_ec"].agg(n_members="size", n_ec="sum")
    return set(t.index[(t["n_members"] >= PFAM_MIN_MEMBERS) & (t["n_ec"] / t["n_members"] >= PFAM_ENZ_SHARE)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true", help="query UniProt again instead of using the frozen copies")
    ap.add_argument("--abc", default="dataset/negatives/negatives_abc.csv")
    ap.add_argument("--out-all", default="dataset/negatives/negatives_gold_all.csv")
    ap.add_argument("--out", default="dataset/negatives/negatives_gold.csv")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    mmseqs = tool("mmseqs")
    if args.fetch:
        os.makedirs(RAW, exist_ok=True)
        u.stream(" AND ".join(u.COMMON + [u.SETS["A"]] + GOLD_TERMS), u.FIELDS + ",xref_pfam",
                 f"{RAW}/uniprot_gold.tsv.gz")
        u.stream("reviewed:true", "accession,ec,xref_pfam", f"{RAW}/swissprot_pfam_ec.tsv.gz")

    d = u.load_tsv(f"{RAW}/uniprot_gold.tsv.gz")
    enz_fam = enzyme_pfam_families(f"{RAW}/swissprot_pfam_ec.tsv.gz")
    kws = d["keywords"].fillna("").map(lambda x: {k.strip() for k in x.split(";")})
    pf = d["pfam"].fillna("").map(lambda x: {y for y in x.strip(";").split(";") if y})
    d["f_ligand_kw"] = kws.map(lambda k: bool(k & LIGAND_KW))
    d["f_enzyme_kw"] = d["keywords"].fillna("").str.contains(ENZYME_KW)
    d["f_unsure_name"] = d["protein_name"].astype(str).str.contains(UNSURE_NAME)
    d["f_enzyme_pfam"] = pf.map(lambda x: bool(x & enz_fam))
    d["f_in_training"] = d["accession"].isin(u.known_ids())
    d["f_in_abc"] = d["accession"].isin(set(pd.read_csv(args.abc)["accession"]))
    d["gold_pass"] = ~d[FLAG_COLS].any(axis=1)

    cl = u.cluster40(d, mmseqs, args.threads).rename(columns={"representative": "cluster_rep", "member": "accession"})
    d = d.merge(cl, on="accession", how="left")
    d["cluster_size"] = d.groupby("cluster_rep")["accession"].transform("size")
    d["is_cluster_rep"] = d["accession"] == d["cluster_rep"]
    d = d.merge(u.identity_to_pools(d[["accession", "sequence"]], mmseqs, args.threads), on="accession", how="left")
    d["org_group"] = d["lineage"].map(u.org_group)
    d["kw_category"] = d["keywords"].map(u.kw_category)

    rng = np.random.default_rng(u.SEED)
    clusters = np.array(sorted(d["cluster_rep"].unique()))   # validation by cluster: near-duplicates stay together
    val = set(rng.choice(clusters, size=int(round(u.VALID_FRACTION * len(clusters))), replace=False))
    d["split_role"] = np.where(d["cluster_rep"].isin(val), "valid", "report")
    pool = d[d["gold_pass"] & d["is_cluster_rep"]]
    d["in_gold_sample"] = d["accession"].isin(u.stratified(pool, min(1000, len(pool)), rng)["accession"])

    d["set"] = d["origin_set"] = "G"
    d["target_cov"] = np.nan
    d["homology_type"] = ""
    d["n_binding_residues"] = 0
    d["binding_positions"] = d["binding_ligands"] = ""
    base = [c for c in pd.read_csv(args.abc, nrows=0).columns if c != "sequence"]
    cols = base + ["pfam", "cluster_rep", "is_cluster_rep", "gold_pass", "in_gold_sample"] + FLAG_COLS + ["sequence"]
    d = d.sort_values("accession")[cols]
    d.to_csv(args.out_all, index=False)

    pool = d[d["gold_pass"] & d["is_cluster_rep"]]
    g = u.stratified(pool, min(N_SUBSET, len(pool)), np.random.default_rng(u.SEED + 2)).sort_values("accession")
    g.to_csv(args.out, index=False)
    rep = {"candidates": len(d), "gold_pass": int(d["gold_pass"].sum()), "clusters": int(d["cluster_rep"].nunique()),
           "gold_pass_representatives": len(pool), "scored_subset": len(g),
           "subset_by_role": g["split_role"].value_counts().to_dict(),
           "flags": {c: int(d[c].sum()) for c in FLAG_COLS}}
    os.makedirs(f"{PIPELINE}/negatives", exist_ok=True)
    with open(f"{PIPELINE}/negatives/negatives_gold_report.json", "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
