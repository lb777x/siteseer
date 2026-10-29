"""Pocket crops for docking: residues within 15 A (any atom) of the predicted site's CA centroid (optionally
weighted by the predicted probabilities), gaps of <= 10 residues between them filled, islands of <= 3 consecutive
residues dropped. Coordinates are kept, so docked poses share the frame of the full structure.
"""
import numpy as np
from Bio.PDB import PDBIO, PDBParser, Select


def centroid(structure, residues, weights=None):
    """CA centroid of the given residue numbers; weights: {residue number: probability}."""
    xyz, w = [], []
    for model in structure:
        for chain in model:
            for res in chain:
                n = res.id[1]
                if "CA" in res and (n in weights if weights is not None else n in residues):
                    xyz.append(res["CA"].get_coord())
                    w.append(weights[n] if weights is not None else 1.0)
    if not xyz:
        raise ValueError("no CA atom for the given residues")
    w = np.asarray(w, dtype=float)
    if weights is not None and w.sum() <= 0:
        raise ValueError("weights sum to <= 0")
    return np.average(np.asarray(xyz), axis=0, weights=w) if weights is not None else np.mean(xyz, axis=0)


def pocket_residues(structure, center, radius=15.0, max_gap=10, min_island=3):
    core = set()
    for model in structure:
        for chain in model:
            for res in chain:
                if any(np.linalg.norm(atom.get_coord() - center) <= radius for atom in res):
                    core.add(res.id[1])
    s = sorted(core)
    filled = {r for a, b in zip(s, s[1:]) if 0 < b - a - 1 <= max_gap for r in range(a + 1, b)}
    keep = sorted(core | filled)
    orphans, island = set(), []
    for i, r in enumerate(keep):
        island.append(r)
        if i == len(keep) - 1 or keep[i + 1] != r + 1:
            if len(island) <= min_island:
                orphans.update(island)
            island = []
    return (core | filled) - orphans


class _Keep(Select):
    def __init__(self, keep):
        self.keep = keep

    def accept_residue(self, residue):
        return 1 if residue.id[1] in self.keep else 0


def crop_protein(input_pdb, output_pdb, residues, radius=15.0, weights=None):
    structure = PDBParser(QUIET=True).get_structure("protein", input_pdb)
    keep = pocket_residues(structure, centroid(structure, residues, weights), radius)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb, _Keep(keep))
    return len(keep)
