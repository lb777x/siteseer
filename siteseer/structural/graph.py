"""Protein structure -> backbone frames -> SE(3)-invariant k-nearest-neighbour residue graph.

Nodes: sin/cos of the backbone dihedrals phi, psi, omega (6 scalars); unit(CA-N), unit(C-CA) and their cross
product (3 vectors). Edges: the k nearest CA atoms (k = 24); 16 Gaussian radial basis functions of the CA-CA
distance (0-20 A) and the unit CA direction. An optional per-residue language-model embedding rides along as
data.side_channel_embedding, outside the graph features.
"""
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from Bio.Data.PDBData import protein_letters_3to1
from Bio.PDB import PDBParser
from torch_geometric.data import Data

AF_MODEL_VERSIONS = (6, 5, 4, 3, 2, 1)


@dataclass
class StructureFrames:
    coords_n: torch.Tensor      # [R, 3]
    coords_ca: torch.Tensor
    coords_c: torch.Tensor
    coords_o: torch.Tensor
    sequence: str
    residue_numbers: List[int]  # numbering as in the PDB file
    plddt: Optional[torch.Tensor]


def slice_frames(frames: StructureFrames, keep: torch.Tensor) -> StructureFrames:
    idx = keep.nonzero(as_tuple=True)[0]
    il = idx.tolist()
    return StructureFrames(frames.coords_n[idx], frames.coords_ca[idx], frames.coords_c[idx], frames.coords_o[idx],
                           "".join(frames.sequence[i] for i in il), [frames.residue_numbers[i] for i in il],
                           frames.plddt[idx] if frames.plddt is not None else None)


def resolve_pdb_path(row, structure_root: str, id_column: str = "alphafolddb-id",
                     pdb_files_column: Optional[str] = "pdb_files") -> Optional[str]:
    """AF-<id>-F1-model_v<6..1>.pdb, <id>.pdb, <id lowercase>.pdb or the basename of the row's pdb_files entry."""
    raw = row[id_column]
    if pd.isna(raw):
        return None
    pid = str(raw).strip()
    candidates = [f"AF-{pid}-F1-model_v{v}.pdb" for v in AF_MODEL_VERSIONS] + [f"{pid}.pdb", f"{pid.lower()}.pdb"]
    if pdb_files_column and pdb_files_column in row.index and isinstance(row[pdb_files_column], str) \
            and row[pdb_files_column].strip():
        candidates.append(os.path.basename(row[pdb_files_column].strip()))
    for c in candidates:
        if os.path.exists(os.path.join(structure_root, c)):
            return os.path.join(structure_root, c)
    return None


def load_structure_frames(pdb_path: str, chain_id: Optional[str] = None) -> StructureFrames:
    """Backbone atoms of the first (or given) chain; residues without a full backbone or of unknown type are
    skipped. pLDDT from the CA B-factor."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PDBParser(QUIET=True).get_structure("prot", pdb_path)[0]
    chain = model[chain_id] if chain_id is not None else next(model.get_chains())
    n, ca, c, o, seq, nums, plddt = [], [], [], [], [], [], []
    for res in chain:
        if res.id[0] != " " or not all(a in res for a in ("N", "CA", "C", "O")):
            continue
        aa = protein_letters_3to1.get(res.get_resname())
        if aa is None:
            continue
        n.append(res["N"].get_coord()); ca.append(res["CA"].get_coord())
        c.append(res["C"].get_coord()); o.append(res["O"].get_coord())
        seq.append(aa); nums.append(res.id[1]); plddt.append(res["CA"].get_bfactor())
    if not nums:
        raise ValueError(f"no parseable backbone residues in {pdb_path}")
    t = lambda x: torch.tensor(np.array(x), dtype=torch.float32)
    return StructureFrames(t(n), t(ca), t(c), t(o), "".join(seq), nums, torch.tensor(plddt, dtype=torch.float32))


def _dihedral(p0, p1, p2, p3, eps=1e-8):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / (b1.norm(dim=-1, keepdim=True) + eps)
    v = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
    return torch.atan2((torch.cross(b1n, v, dim=-1) * w).sum(-1), (v * w).sum(-1))


def dihedral_features(frames: StructureFrames) -> torch.Tensor:
    """[R, 6] sin/cos of phi, psi, omega; undefined angles at the chain ends are 0 for both."""
    n, ca, c = frames.coords_n, frames.coords_ca, frames.coords_c
    r = n.shape[0]
    phi, psi, omega = torch.zeros(r), torch.zeros(r), torch.zeros(r)
    if r > 1:
        phi[1:] = _dihedral(c[:-1], n[1:], ca[1:], c[1:])
        psi[:-1] = _dihedral(n[:-1], ca[:-1], c[:-1], n[1:])
        omega[1:] = _dihedral(ca[:-1], c[:-1], n[1:], ca[1:])
    return torch.stack([torch.sin(phi), torch.cos(phi), torch.sin(psi), torch.cos(psi), torch.sin(omega),
                        torch.cos(omega)], dim=-1)


def rbf_encode(d, d_min=0.0, d_max=20.0, num_rbf=16):
    centers = torch.linspace(d_min, d_max, num_rbf, device=d.device)
    return torch.exp(-((d.unsqueeze(-1) - centers) / ((d_max - d_min) / num_rbf)) ** 2)


def build_knn_graph(frames: StructureFrames, k: int = 24,
                    side_channel_embedding: Optional[torch.Tensor] = None) -> Data:
    r = frames.coords_ca.shape[0]
    k = min(k, r - 1) if r > 1 else 0
    eps = 1e-8
    v1 = frames.coords_ca - frames.coords_n
    v1 = v1 / (v1.norm(dim=-1, keepdim=True) + eps)
    v2 = frames.coords_c - frames.coords_ca
    v2 = v2 / (v2.norm(dim=-1, keepdim=True) + eps)
    v3 = torch.cross(v1, v2, dim=-1)
    v3 = v3 / (v3.norm(dim=-1, keepdim=True) + eps)
    ca = frames.coords_ca
    dist = torch.cdist(ca, ca)
    dist.fill_diagonal_(float("inf"))
    knn_dist, knn_idx = dist.topk(k, largest=False, dim=-1)
    dst = torch.arange(r).repeat_interleave(k)
    src = knn_idx.reshape(-1)
    edge_dir = ca[src] - ca[dst]
    edge_dir = edge_dir / (edge_dir.norm(dim=-1, keepdim=True) + eps)
    data = Data(edge_index=torch.stack([src, dst], dim=0), num_nodes=r)
    data.s = dihedral_features(frames)
    data.v = torch.stack([v1, v2, v3], dim=1)
    data.edge_s = rbf_encode(knn_dist.reshape(-1))
    data.edge_v = edge_dir.unsqueeze(1)
    data.pos = ca
    data.residue_numbers = frames.residue_numbers
    data.sequence = frames.sequence
    if side_channel_embedding is not None:
        if side_channel_embedding.shape[0] != r:
            raise ValueError(f"embedding has {side_channel_embedding.shape[0]} residues, structure {r}")
        data.side_channel_embedding = side_channel_embedding.float()
    return data
