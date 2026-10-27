"""Datasets for the structural probe: per protein the backbone frames of its AlphaFold model, a per-residue ESMC
embedding in the same residue order, and labels mapped by residue number.

Proteins need a resolvable structure with >= 10 residues and mean pLDDT >= 30. Cached (Goliath scale): the cached
embedding was computed from aa_sequence, so the structure's sequence must be a prefix of it; structure, embedding
and labels are cut to the first 1022 residues. Live (M-CSA scale): ESMC runs on the structure's own sequence.
"""
import os
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from siteseer.data.labels import parse_site_labels, row_label_source
from siteseer.embeddings.embedders import MAX_SEQ_LEN
from siteseer.structural.graph import load_structure_frames, resolve_pdb_path, slice_frames

MIN_CA, MIN_PLDDT = 10, 30.0


def labels_by_residue_number(residue_numbers, label_indices) -> Optional[torch.Tensor]:
    pos = {n: i for i, n in enumerate(residue_numbers)}
    y = torch.zeros(len(residue_numbers))
    hits = [pos[i] for i in label_indices if i in pos]
    y[hits] = 1.0
    return y if hits else None


def usable_frames(row, structure_root, id_column):
    path = resolve_pdb_path(row, structure_root, id_column)
    if path is None:
        return None, None, "unresolved"
    try:
        frames = load_structure_frames(path)
    except Exception:
        return None, None, "unresolved"
    if len(frames.residue_numbers) < MIN_CA or (frames.plddt.numel() and frames.plddt.mean().item() < MIN_PLDDT):
        return None, None, "failed_qc"
    return path, frames, None


class StructuralLiveDataset(Dataset):
    """ESMC computed on the structure's sequence at construction (M-CSA scale)."""

    def __init__(self, csv_path, structure_root, embedder, id_column="alphafolddb-id", filter_ids=None, label=""):
        df = pd.read_csv(csv_path).drop_duplicates(subset=id_column)
        if filter_ids is not None:
            df = df[df[id_column].astype(str).str.strip().isin({str(i).strip() for i in filter_ids})]
        self.items, skipped = [], {}
        for _, row in df.iterrows():
            _, frames, why = usable_frames(row, structure_root, id_column)
            labels = None
            if frames is not None:
                idx = parse_site_labels(row["site_labels"], row_label_source(row))
                why = "bad_labels" if idx is None else None
                labels = labels_by_residue_number(frames.residue_numbers, idx) if idx is not None else None
                why = why or (None if labels is not None else "no_positive")
            if why:
                skipped[why] = skipped.get(why, 0) + 1
                continue
            self.items.append({"frames": frames, "esm_embedding": embedder.embed(frames.sequence).float(),
                               "labels": labels, "protein_id": str(row[id_column]).strip()})
        print(f"[{label}] {len(self.items)} usable proteins, skipped {skipped}")

    @property
    def protein_ids(self):
        return [it["protein_id"] for it in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class StructuralCachedDataset(Dataset):
    """Cached ESMC embeddings (Goliath scale); structures and embeddings are read per item."""

    def __init__(self, csv_path, structure_root, embedding_dir, split_name, id_column="alphafolddb-id", label="",
                 resample_on_error=False):
        self.resample_on_error = resample_on_error
        d = pd.read_csv(csv_path)
        d = d[d["dataset_flag"] == split_name]
        d = d.rename(columns={"_original_row_index": "index"}) if "_original_row_index" in d else d.reset_index()
        d = d.drop_duplicates(subset=id_column)
        self.entries, skipped = [], {}
        for _, row in d.iterrows():
            path, frames, why = usable_frames(row, structure_root, id_column)
            emb_path = os.path.join(embedding_dir, f"row_{row['index']}.pt")
            if why is None and not os.path.exists(emb_path):
                why = "missing_embedding"
            if why is None and not (isinstance(row.get("aa_sequence"), str) and row["aa_sequence"].startswith(frames.sequence)):
                why = "sequence_mismatch"
            y = None
            if why is None:
                idx = parse_site_labels(row["site_labels"], row_label_source(row))
                y = labels_by_residue_number(frames.residue_numbers, idx) if idx is not None else None
                why = "bad_labels" if idx is None else (None if y is not None else "no_positive")
            n = len(frames.residue_numbers) if frames is not None else 0
            if why is None and n > MAX_SEQ_LEN:
                n, y = MAX_SEQ_LEN, y[:MAX_SEQ_LEN]
                why = None if y.sum() > 0 else "no_positive"
            if why:
                skipped[why] = skipped.get(why, 0) + 1
                continue
            self.entries.append({"pdb_path": path, "emb_path": emb_path, "n_residues": n, "labels": y,
                                 "protein_id": str(row[id_column]).strip()})
        print(f"[{label}] {len(self.entries)} usable proteins, skipped {skipped}")

    @property
    def protein_ids(self):
        return [e["protein_id"] for e in self.entries]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        try:
            frames = load_structure_frames(e["pdb_path"])
            if len(frames.residue_numbers) > e["n_residues"]:
                keep = torch.zeros(len(frames.residue_numbers), dtype=torch.bool)
                keep[:e["n_residues"]] = True
                frames = slice_frames(frames, keep)
            emb = torch.load(e["emb_path"], map_location="cpu", weights_only=False)["embedding"].float()[:e["n_residues"]]
        except Exception as err:
            if not self.resample_on_error:
                raise RuntimeError(f"failed to load {e['protein_id']}") from err
            return self[(i + 1) % len(self)]
        return {"frames": frames, "esm_embedding": emb, "labels": e["labels"], "protein_id": e["protein_id"]}
