"""Labelled per-residue datasets on frozen language-model embeddings.

Two kinds, same item schema ({"embedding": [L, d], "labels": [L], "protein_id": str}, plus "type_targets" /
"type_mask" [L, 2] with site types):

  LiveEmbeddingDataset    computes embeddings with an Embedder at construction (M-CSA scale, < 1,000 proteins)
                          and keeps them in memory
  CachedEmbeddingDataset  reads precomputed embeddings `<embedding_dir>/row_<key>.pt` lazily (Goliath scale);
                          the key is the row's `_original_row_index` if present, else its row position, i.e. the
                          row in cached_dataframe_final_resplit90.csv

`site_labels` index the sequence 1-based; sequences are truncated to 1,022 residues.
"""
import os
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from siteseer.data.labels import parse_site_labels, row_label_source
from siteseer.data.site_types import site_type_sets, type_targets
from siteseer.embeddings.embedders import MAX_SEQ_LEN, Embedder, apply_norm, load_norm_stats


def _filter_ids(df: pd.DataFrame, id_column: str, filter_ids: Optional[set]) -> pd.DataFrame:
    if filter_ids is None:
        return df
    keep = {str(i).strip() for i in filter_ids}
    return df[df[id_column].astype(str).str.strip().isin(keep)]


class LiveEmbeddingDataset(Dataset):
    def __init__(self, csv_path: str, embedder: Embedder, id_column: str = "alphafolddb-id",
                 sequence_column: str = "aa_sequence", filter_ids: Optional[set] = None, label: str = "",
                 with_site_types: bool = False):
        self.with_site_types = with_site_types
        df = _filter_ids(pd.read_csv(csv_path), id_column, filter_ids).drop_duplicates(subset=id_column)
        self.embeddings, self.labels, self.types, self.protein_ids = [], [], [], []
        n_missing = n_bad = n_no_pos = n_trunc = 0
        for _, row in df.iterrows():
            seq = row.get(sequence_column)
            if not isinstance(seq, str) or not seq:
                n_missing += 1
                continue
            positives = parse_site_labels(row["site_labels"], row_label_source(row))
            if positives is None:
                n_bad += 1
                continue
            if len(seq) > MAX_SEQ_LEN:
                seq = seq[:MAX_SEQ_LEN]
                n_trunc += 1
            y = torch.zeros(len(seq))
            matched = [p - 1 for p in positives if 1 <= p <= len(seq)]
            if not matched:
                n_no_pos += 1
                continue
            y[matched] = 1.0
            self.embeddings.append(embedder.embed(seq))
            self.labels.append(y)
            self.protein_ids.append(str(row[id_column]).strip())
            if with_site_types:
                # M-CSA CSVs have no label_source column: catalytic supervised, binding masked
                sets = site_type_sets(row["site_labels"], None, row_label_source(row) or "mcsa")
                self.types.append(type_targets(len(seq), sets))
        print(f"[{label}] {len(self.embeddings)} usable proteins ({n_missing} missing sequence, {n_bad} bad labels, "
              f"{n_no_pos} without a positive label, {n_trunc} truncated to {MAX_SEQ_LEN}, out of {len(df)} ids)")

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> dict:
        item = {"embedding": self.embeddings[idx], "labels": self.labels[idx], "protein_id": self.protein_ids[idx]}
        if self.with_site_types:
            item["type_targets"], item["type_mask"] = self.types[idx]
        return item


class CachedEmbeddingDataset(Dataset):
    def __init__(self, csv_path: str, embedding_dir: str, split_name: str = "train",
                 id_column: str = "alphafolddb-id", filter_ids: Optional[set] = None, label: str = "",
                 resample_on_error: bool = False, norm_stats: Optional[str] = None,
                 with_site_types: bool = False):
        # resample_on_error only for the training split: in evaluation a substituted neighbour would be scored
        # under the wrong protein id and labels, so there a corrupted cache file raises
        self.resample_on_error = resample_on_error
        self.with_site_types = with_site_types
        self.stats = load_norm_stats(norm_stats)
        full = pd.read_csv(csv_path)
        split = _filter_ids(full[full["dataset_flag"] == split_name], id_column, filter_ids)
        if "_original_row_index" in split.columns:
            df = split.rename(columns={"_original_row_index": "index"})
        else:
            df = split.reset_index()
        df = df.drop_duplicates(subset=id_column)

        self.paths, self.positives, self.type_sets, self.protein_ids = [], [], [], []
        n_missing = n_bad = 0
        for _, row in df.iterrows():
            path = os.path.join(embedding_dir, f"row_{row['index']}.pt")
            if not os.path.exists(path):
                n_missing += 1
                continue
            positives = parse_site_labels(row["site_labels"], row_label_source(row))
            if positives is None:
                n_bad += 1
                continue
            if with_site_types:
                sets = site_type_sets(row["site_labels"], row.get("site_types"), row_label_source(row))
                if sets is None:
                    n_bad += 1
                    continue
                self.type_sets.append(sets)
            self.paths.append(path)
            self.positives.append(positives)
            self.protein_ids.append(str(row[id_column]).strip())
        print(f"[{label}] {len(self.paths)} usable proteins ({n_missing} missing from the embedding cache, "
              f"{n_bad} bad labels, out of {len(df)} ids)")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        try:
            emb = torch.load(self.paths[idx], map_location="cpu", weights_only=False)["embedding"].float()
            emb = apply_norm(emb, self.stats)
        except Exception as e:
            if not self.resample_on_error:
                raise RuntimeError(f"corrupted embedding cache file {self.paths[idx]} "
                                   f"(protein {self.protein_ids[idx]})") from e
            return self.__getitem__((idx + 1) % len(self))
        y = torch.zeros(emb.shape[0])
        for p in self.positives[idx]:
            if 1 <= p <= emb.shape[0]:
                y[p - 1] = 1.0
        item = {"embedding": emb, "labels": y, "protein_id": self.protein_ids[idx]}
        if self.with_site_types:
            item["type_targets"], item["type_mask"] = type_targets(emb.shape[0], self.type_sets[idx])
        return item


def identity_collate(batch: list) -> list:
    return batch


def build_dataset(cfg: dict, split: str, embedder: Optional[Embedder] = None, filter_ids=None):
    """The dataset a config describes. data.dataset_class 'mcsa' (default): live embeddings from
    data.{split}_csv (needs embedder). 'goliath': cached embeddings from data.train_csv filtered by
    dataset_flag == split, data.embedding_dir and data.norm_stats."""
    data_cfg = cfg["data"]
    with_types = cfg.get("probe", {}).get("n_site_types", 0) > 0
    dataset_class = data_cfg.get("dataset_class", "mcsa")
    if dataset_class == "mcsa":
        if embedder is None:
            raise ValueError("dataset_class 'mcsa' computes embeddings live and needs an embedder")
        return LiveEmbeddingDataset(data_cfg[f"{split}_csv"], embedder, id_column=data_cfg["id_column"],
                                    sequence_column=data_cfg.get("sequence_column", "aa_sequence"),
                                    filter_ids=filter_ids, label=f"live:{split}", with_site_types=with_types)
    if dataset_class == "goliath":
        return CachedEmbeddingDataset(data_cfg["train_csv"], data_cfg["embedding_dir"], split_name=split,
                                      id_column=data_cfg["id_column"], filter_ids=filter_ids,
                                      label=f"cached:{split}", resample_on_error=(split == "train"),
                                      norm_stats=data_cfg.get("norm_stats"), with_site_types=with_types)
    raise ValueError(f"unknown data.dataset_class {dataset_class!r} (expected 'mcsa' or 'goliath')")
