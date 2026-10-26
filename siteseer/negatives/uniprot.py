"""Shared pieces of the negative-control sets: UniProt queries and fields, the training pools, MMseqs2 identity to
them, and the stratification by organism group and keyword category.

The UniProt query results change over time; the reported sets were built from the copies of 2026-10-05 in the data
release (data/external/uniprot/). Identity = MMseqs2 fident of the best hit covering >= 50% of the query.
"""
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
import requests

API = "https://rest.uniprot.org/uniprotkb"
SEED = 42
VALID_FRACTION = 0.3
# reviewed, no fragment, no sign of catalysis, 50-1022 residues, AlphaFold model
COMMON = ["reviewed:true", "fragment:false", "NOT ec:*", "NOT cc_catalytic_activity:*", "NOT ft_act_site:*",
          "NOT ft_site:*", "NOT go:0003824", "length:[50 TO 1022]", "database:alphafolddb"]
SETS = {"A": "NOT ft_binding:*", "B": "ft_binding:*"}
FIELDS = ("accession,id,protein_name,organism_name,organism_id,lineage,keyword,length,sequence,"
          "xref_alphafolddb,ft_binding,protein_existence,annotation_score")
RENAME = {"Entry": "accession", "Entry Name": "entry_name", "Protein names": "protein_name", "Organism": "organism",
          "Organism (ID)": "organism_id", "Taxonomic lineage": "lineage", "Keywords": "keywords", "Length": "length",
          "Sequence": "sequence", "AlphaFoldDB": "alphafold_id", "Binding site": "ft_binding",
          "Protein existence": "protein_existence", "Annotation": "annotation_score", "Pfam": "pfam"}

# training data of the model the sets were built for (fix1 lineage), train + valid
POOLS = {
    "goliath": ["dataset/label_diffusion/cached_dataframe_fully_excl_fix1.csv"],
    "real_labels": ["dataset/label_diffusion/cached_dataframe_real_labels_only_fix1.csv"],
    "mcsa": ["dataset/mcsa_fine_tune/normal_mcsa/train_dataset/mcsa_train.csv",
             "dataset/mcsa_fine_tune/normal_mcsa/valid_dataset/mcsa_valid.csv"],
}
# proteins listed here (any split) were seen as enzymes and are never drawn as negatives
ID_EXCLUDE = POOLS["goliath"] + POOLS["real_labels"] + POOLS["mcsa"] + \
    ["dataset/mcsa_fine_tune/normal_mcsa/test_dataset/mcsa_test.csv"]
BANDS = [(0, 30, "<30"), (30, 50, "30-50"), (50, 90, "50-90"), (90, 101, ">=90")]
UNCHARACTERIZED = re.compile(r"uncharacteri[sz]ed|hypothetical|\bUPF\d|\bDUF\d", re.I)
# coarse function from UniProt keywords; the first matching category wins
KW_CATEGORIES = [
    ("Ribosome / RNA-binding", {"Ribosomal protein", "Ribonucleoprotein", "RNA-binding", "rRNA-binding",
                                "tRNA-binding"}),
    ("DNA-binding / transcription", {"DNA-binding", "Transcription", "Transcription regulation", "Repressor",
                                     "Activator", "Chromatin regulator"}),
    ("Receptor / signalling", {"Receptor", "G-protein coupled receptor", "Transducer", "Hormone", "Cytokine",
                               "Growth factor"}),
    ("Transport / channel", {"Transport", "Ion transport", "Ion channel", "Protein transport", "Symport",
                             "Antiport", "Sugar transport", "Amino-acid transport"}),
    ("Structural", {"Structural protein", "Cytoskeleton", "Collagen", "Keratin", "Cell shape", "Flagellum",
                    "Capsid protein", "Extracellular matrix"}),
    ("Toxin / defence", {"Toxin", "Antimicrobial", "Antibiotic", "Neurotoxin"}),
    ("Chaperone", {"Chaperone"}),
]
ORG_GROUPS = [("Viruses", "Viruses"), ("Archaea", "Archaea"), ("Bacteria", "Bacteria"),
              ("Vertebrata", "Vertebrates"), ("Metazoa", "Other animals"), ("Fungi", "Fungi"),
              ("Viridiplantae", "Plants"), ("Eukaryota", "Other eukaryotes")]


def stream(query, fields, path):
    """Download a UniProt query as gzipped TSV (only with --fetch; the reported sets used the frozen copies)."""
    r = requests.get(f"{API}/stream", params={"query": query, "format": "tsv", "fields": fields,
                                               "compressed": "true"}, stream=True, timeout=3600)
    r.raise_for_status()
    with open(path + ".tmp", "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    os.replace(path + ".tmp", path)


def load_tsv(path):
    return pd.read_csv(path, sep="\t", compression="gzip", low_memory=False).rename(columns=RENAME)


def band(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "<30"
    for lo, hi, name in BANDS:
        if lo <= x < hi:
            return name
    return ">=90"


def org_group(lineage):
    for key, name in ORG_GROUPS:
        if re.search(rf"\b{key}\b", str(lineage)):
            return name
    return "Other"


def kw_category(keywords):
    kws = {k.strip() for k in str(keywords).split(";")}
    for name, keys in KW_CATEGORIES:
        if kws & keys:
            return name
    return "Other"


def binding_info(ft):
    """1-based positions and ligand names of a UniProt ft_binding field."""
    if not isinstance(ft, str) or not ft:
        return [], []
    pos = set()
    for a, b in re.findall(r"BINDING (\d+)(?:\.\.(\d+))?", ft):
        pos.update(range(int(a), int(b) + 1) if b else [int(a)])
    return sorted(pos), sorted(set(re.findall(r'/ligand="([^"]+)"', ft)))


def write_fasta(path, ids, seqs):
    with open(path, "w") as f:
        for i, s in zip(ids, seqs):
            f.write(f">{i}\n{s}\n")


def known_ids():
    known = set()
    for p in ID_EXCLUDE:
        known |= set(pd.read_csv(p, low_memory=False)["alphafolddb-id"].astype(str).str.strip())
    return known


def cluster40(d, mmseqs, threads):
    """MMseqs2 easy-cluster at 40% identity and 80% coverage: (representative, member)."""
    work = tempfile.mkdtemp(prefix="neg_cluster_")
    try:
        write_fasta(f"{work}/in.fasta", d["accession"], d["sequence"])
        subprocess.run([mmseqs, "easy-cluster", f"{work}/in.fasta", f"{work}/res", f"{work}/tmp", "--min-seq-id",
                        "0.4", "-c", "0.8", "--cov-mode", "0", "--threads", str(threads), "-v", "1"], check=True)
        return pd.read_csv(f"{work}/res_cluster.tsv", sep="\t", header=None, names=["representative", "member"])
    finally:
        shutil.rmtree(work, ignore_errors=True)


def pool_sequences(name):
    """Deduplicated train + valid sequences of a training pool: {sequence: accession}."""
    seqs = {}
    for p in POOLS[name]:
        d = pd.read_csv(p, low_memory=False)
        if "dataset_flag" in d:
            d = d[d["dataset_flag"].isin(["train", "valid"])]
        for acc, seq in zip(d["alphafolddb-id"].astype(str).str.strip(), d["aa_sequence"].astype(str).str.strip()):
            if seq and seq.isalpha():
                seqs.setdefault(seq, acc)
    return seqs


def search_pool(name, q, mmseqs, threads, columns="query,target,fident,alnlen,qcov,tcov,evalue"):
    """MMseqs2 easy-search of the query proteins against one training pool (-s 7.5, 1,000 targets, e-value 1e-3)."""
    seqs = pool_sequences(name)
    work = tempfile.mkdtemp(prefix=f"neg_identity_{name}_")
    try:
        write_fasta(f"{work}/q.fasta", q["accession"], q["sequence"])
        write_fasta(f"{work}/t.fasta", [f"{a}__{k}" for k, a in enumerate(seqs.values())], list(seqs.keys()))
        subprocess.run([mmseqs, "easy-search", f"{work}/q.fasta", f"{work}/t.fasta", f"{work}/hits.tsv", f"{work}/tmp",
                        "-s", "7.5", "--max-seqs", "1000", "-e", "1e-3", "--format-output", columns,
                        "--threads", str(threads), "-v", "1"], check=True)
        return pd.read_csv(f"{work}/hits.tsv", sep="\t", header=None, names=["accession"] + columns.split(",")[1:])
    finally:
        shutil.rmtree(work, ignore_errors=True)


def best_hits(h):
    """Best hit per query (coverage >= 0.5); ties in identity broken by coverage, then target name, so the result
    does not depend on MMseqs2's thread-dependent output order."""
    h = h[h["qcov"] >= 0.5]
    return h.sort_values(["fident", "qcov", "target"], ascending=[False, False, True], kind="mergesort") \
        .drop_duplicates("accession")


def identity_to_pools(q, mmseqs, threads):
    """Best hit per query and pool (query coverage >= 0.5), the maximum over pools and its band."""
    res = pd.DataFrame({"accession": q["accession"]})
    for name in POOLS:
        h = best_hits(search_pool(name, q, mmseqs, threads))
        res = res.merge(pd.DataFrame({"accession": h["accession"], f"identity_{name}": 100 * h["fident"],
                                      f"best_hit_{name}": h["target"].str.split("__").str[0],
                                      f"qcov_{name}": h["qcov"]}), on="accession", how="left")
    idc = [f"identity_{n}" for n in POOLS]
    res["max_identity"] = res[idc].max(axis=1)
    res["max_identity_pool"] = res[idc].fillna(-1).idxmax(axis=1).str.replace("identity_", "", regex=False)
    res.loc[res["max_identity"].isna(), "max_identity_pool"] = ""
    res["identity_band"] = [band(x) for x in res["max_identity"]]
    return res


def allocate(sizes, n):
    """Proportional allocation of n over strata (largest remainder)."""
    raw = sizes / sizes.sum() * n
    base = np.floor(raw).astype(int)
    order = (raw - base).sort_values(ascending=False).index[:n - base.sum()]
    base[order] += 1
    return base.clip(upper=sizes)


# Order in which strata are visited (and random draws made): that of the reported run, whose labels sorted differently.
ORG_ORDER = ["Archaea", "Bacteria", "Plants", "Fungi", "Viruses", "Vertebrates", "Other", "Other eukaryotes",
             "Other animals"]
KW_ORDER = ["Chaperone", "DNA-binding / transcription", "Receptor / signalling", "Ribosome / RNA-binding", "Other",
            "Structural", "Toxin / defence", "Transport / channel"]


def stratified(pool, n, rng):
    """n proteins stratified by organism group x keyword category."""
    picked = []
    keys = pd.DataFrame({"org_group": pd.Categorical(pool["org_group"], ORG_ORDER, ordered=True),
                         "kw_category": pd.Categorical(pool["kw_category"], KW_ORDER, ordered=True)}, index=pool.index)
    sizes = keys.groupby(["org_group", "kw_category"], observed=True).size()
    for key, k in allocate(sizes, n).items():
        if k:
            grp = pool[(pool["org_group"] == key[0]) & (pool["kw_category"] == key[1])]
            picked.append(grp.sample(n=int(k), random_state=int(rng.integers(1 << 31))))
    return pd.concat(picked, ignore_index=True)
