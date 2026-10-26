"""Shared helpers for the data pipeline (pipeline/NN_*.py): default paths of the release data layout and tool calls.

Layout (relative to the working directory, the data root):
  dataset/ec_site_dataset/.../  dataset/mcsa_fine_tune/   EasIFA's dataset (00_download_easifa.py)
  data/external/                                         other downloaded sources (M3Site, BioLiP2)
  data/pipeline/                                         intermediate results (BLAST/MMseqs2 outputs, id lists)
  dataset/label_diffusion/                               the training CSVs the configs read
"""
import os
import shutil
import subprocess

EASIFA_DIR = "dataset/ec_site_dataset/uniprot_ecreact_cluster_split_merge_dataset_limit_100"
GOLIATH_CSV = f"{EASIFA_DIR}/cached_dataframe.csv"
GOLIATH_BLAST_DB = f"{EASIFA_DIR}/blast_database"
MCSA_DIR = "dataset/mcsa_fine_tune/normal_mcsa"
MCSA_CSV = {s: f"{MCSA_DIR}/{s}_dataset/mcsa_{s}.csv" for s in ("train", "valid", "test")}
EXTERNAL = "data/external"
PIPELINE = "data/pipeline"
LABELS = "dataset/label_diffusion"

BLAST_COLUMNS = ["qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
                 "qstart", "qend", "sstart", "send", "evalue", "bitscore", "qseq", "sseq"]


def tool(name: str) -> str:
    """Path of an external tool (blastp, makeblastdb, mmseqs): $<NAME>_BIN, else PATH (envs/compass-tools.yml)."""
    path = os.environ.get(f"{name.upper()}_BIN") or shutil.which(name)
    if not path:
        raise SystemExit(f"{name} not found; activate the compass-tools environment or set {name.upper()}_BIN")
    return path


def run(cmd: list):
    print("[$] " + " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


def blastp(query: str, db: str, out: str, max_target_seqs: int, evalue: float, threads: int = 8):
    """Tabular blastp (BLAST_COLUMNS); skipped if `out` exists."""
    if os.path.exists(out):
        print(f"[*] {out} exists, BLAST skipped")
        return
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    run([tool("blastp"), "-query", query, "-db", db, "-outfmt", "6 " + " ".join(BLAST_COLUMNS),
         "-max_target_seqs", max_target_seqs, "-evalue", evalue, "-num_threads", threads, "-out", out])


def write_fasta(ids, seqs, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("".join(f">{i}\n{s}\n" for i, s in zip(ids, seqs)))


def build_alignment_map(qstart: int, qend: int, sstart: int, send: int, qseq: str, sseq: str) -> dict:
    """{query position: subject position} (1-based) over aligned, non-gap columns."""
    mapping, q_pos, s_pos = {}, qstart, sstart
    for qc, sc in zip(qseq, sseq):
        if qc != "-" and sc != "-":
            mapping[q_pos] = s_pos
        q_pos += qc != "-"
        s_pos += sc != "-"
    assert q_pos == qend + 1 and s_pos == send + 1, "alignment walk did not end at qend/send"
    return mapping


def remap_labels(labels: list, types: list, pos_map: dict) -> tuple:
    """Map each residue group through pos_map; a group losing all residues is dropped with its type."""
    new_labels, new_types = [], []
    for i, group in enumerate(labels):
        mapped = [pos_map[p] for p in group if p in pos_map]
        if mapped:
            new_labels.append(mapped)
            new_types.append(types[i] if i < len(types) else 0)
    return new_labels, new_types
