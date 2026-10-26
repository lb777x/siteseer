"""BioLiP ligand contacts mapped onto UniProt sequences (non-enzyme ligand-site truth for the P2Rank comparison).

Contacts with peptides, nucleic acids, crystallisation additives, detergents and glycans are not sites. Residues are
mapped from the BioLiP chain onto the UniProt sequence by local alignment (BLOSUM62, gap open -10, extend -0.5) and
kept where the amino acid matches.
"""
import re

import pandas as pd
from Bio import Align
from Bio.Align import substitution_matrices

BIOLIP = "data/external/biolip/BioLiP_nr.txt"
COLUMNS = ["pdb", "chain", "res", "site", "lig", "lchain", "lser", "bres_pdb", "bres", "cat_pdb", "cat", "ec", "go",
           "aff1", "aff2", "aff3", "aff4", "uniprot", "pubmed", "lres", "seq"]
NOT_A_SITE = {"peptide", "dna", "rna"}
ADDITIVES = {"SO4", "PO4", "GOL", "EDO", "PEG", "PG4", "PGE", "1PE", "P6G", "MPD", "ACT", "FMT", "TRS", "EPE", "MES",
             "IMD", "CL", "BR", "IOD", "NO3", "SCN", "DMS", "BME", "C8E", "LMT", "BOG", "DMU", "LDA", "UNL", "UNX", "OLC",
             "MLI", "CIT", "TAR", "BCR", "HTG", "SOG", "NHE", "CPS"}
GLYCANS = {"NAG", "NDG", "MAN", "BMA", "FUC", "GAL", "GLA", "SIA", "FUL", "XYP", "A2G", "NGA"}
BUFFER_IONS = {"NA", "K"}
METALS = {"ZN", "MG", "CA", "MN", "FE", "FE2", "CU", "CU1", "CO", "NI", "CD", "HG", "MO", "W", "V", "SR", "BA", "LI",
          "CS", "RB", "PB", "PT", "AU", "AG", "GA", "IN", "TL", "YB", "LA", "EU", "GD", "SM", "TB", "CR", "AL", "BE",
          "SB", "AS"}
ALIGNER = Align.PairwiseAligner(mode="local", open_gap_score=-10, extend_gap_score=-0.5)
ALIGNER.substitution_matrix = substitution_matrices.load("BLOSUM62")


def read(path=BIOLIP):
    return pd.read_csv(path, sep="\t", header=None, names=COLUMNS, low_memory=False, keep_default_na=False,
                       dtype={"lig": str, "pdb": str, "chain": str, "bres": str, "seq": str, "uniprot": str})


def position_map(chain_seq, uni_seq, clean=False):
    """1-based chain position -> 1-based UniProt position (aligned columns). clean: letters outside BLOSUM62's
    alphabet become X first."""
    if clean:
        chain_seq, uni_seq = (re.sub(r"[^ARNDCQEGHILKMFPSTWYVBZX]", "X", s.upper()) for s in (chain_seq, uni_seq))
    aln = ALIGNER.align(chain_seq, uni_seq)[0]
    m = {}
    for (s1, e1), (s2, _) in zip(*aln.aligned):
        for k in range(e1 - s1):
            m[s1 + k + 1] = s2 + k + 1
    return m


def mapped(tokens, m, uni_seq):
    """BioLiP residues ('H57') -> UniProt positions, where the amino acid matches."""
    return {m[int(t[1:])] for t in tokens if t[1:].isdigit() and int(t[1:]) in m and uni_seq[m[int(t[1:])] - 1] == t[0]}
