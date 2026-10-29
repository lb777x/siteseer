"""Predict active-site residues for protein sequences (FASTA) with the trained main model.

Writes one row per residue (protein, position, amino acid, probability, predicted) and prints the predicted sites per
protein. A residue is predicted when its probability exceeds --threshold (0.5, as for the reported results).
Sequences are truncated to 1022 residues (ESM-2's limit).

  python -m siteseer.predict proteins.fasta --out predictions.tsv
"""
import argparse

import torch

from siteseer.embeddings.embedders import MAIN_NORM_STATS, MAX_SEQ_LEN, get_embedder
from siteseer.models.probe import load_probe


def read_fasta(path):
    seqs, name = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            name = line[1:].split()[0]
            seqs[name] = ""
        elif name is not None:
            seqs[name] += line.upper()
    return seqs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fasta")
    ap.add_argument("--out", default="predictions.tsv")
    ap.add_argument("--checkpoint", default="checkpoints/main_stage2_mcsa/best.pt")
    ap.add_argument("--norm-stats", default=MAIN_NORM_STATS)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    embedder = get_embedder("esm2_650m_l30", norm_stats=args.norm_stats).load(device)
    probe, _ = load_probe(args.checkpoint, device)
    with open(args.out, "w") as out:
        out.write("protein\tposition\tresidue\tprobability\tpredicted\n")
        for name, seq in read_fasta(args.fasta).items():
            if len(seq) > MAX_SEQ_LEN:
                print(f"[!] {name}: {len(seq)} residues, only the first {MAX_SEQ_LEN} are scored")
                seq = seq[:MAX_SEQ_LEN]
            with torch.no_grad():
                p = torch.sigmoid(probe(embedder.embed(seq).to(device))).cpu().tolist()
            for i, (aa, pi) in enumerate(zip(seq, p), 1):
                out.write(f"{name}\t{i}\t{aa}\t{pi:.4f}\t{int(pi > args.threshold)}\n")
            sites = [f"{seq[i]}{i + 1}" for i, pi in enumerate(p) if pi > args.threshold]
            print(f"{name}\t{len(seq)} residues\t{' '.join(sites) or '-'}")
    print(f"[+] {args.out}")


if __name__ == "__main__":
    main()
