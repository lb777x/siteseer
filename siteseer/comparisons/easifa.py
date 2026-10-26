"""EasIFA's per-residue predictions (EasIFA_results.zip) in our prediction format: {"protein_ids", "logits",
"labels"}, one entry per protein (its first protein-reaction pair).

predict_active_prob holds [p_inactive, p_active] per residue; p_active is clipped to [1e-6, 1 - 1e-6] before the
logit. Labels are EasIFA's own site_labels, 1-based: element-wise for the M-CSA transfer-learning set (singleton
groups), EasIFA's range convention ([b, e] = b..e) for the Swiss-Prot set.
"""
import ast

import numpy as np
import pandas as pd
import torch

from siteseer.data.labels import parse_site_labels

ID = "alphafolddb-id"
EPS = 1e-6


def first_rows(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d[ID] = d[ID].astype(str).str.strip()
    return d.drop_duplicates(ID).set_index(ID, drop=False)


def active_logits(cell: str) -> torch.Tensor:
    p = np.clip(np.asarray(ast.literal_eval(cell), dtype=np.float64)[:, 1], EPS, 1 - EPS)
    return torch.tensor(np.log(p / (1 - p)), dtype=torch.float32)


def elementwise_labels(site_labels: str, n: int) -> torch.Tensor:
    y = np.zeros(n, dtype=np.float32)
    for group in ast.literal_eval(site_labels):
        for pos in ([group] if isinstance(group, int) else group):
            y[int(pos) - 1] = 1.0
    return torch.tensor(y)


def range_labels(site_labels: str, n: int) -> torch.Tensor:
    y = np.zeros(n, dtype=np.float32)
    for p in parse_site_labels(site_labels, "goliath_original"):
        if 1 <= p <= n:
            y[p - 1] = 1.0
    return torch.from_numpy(y)


def to_predictions(rows: pd.DataFrame, labels_from: pd.DataFrame = None, label_fn=elementwise_labels) -> dict:
    """rows: EasIFA results (first_rows); labels_from: rows whose site_labels are used (default: rows itself)."""
    labels_from = rows if labels_from is None else labels_from
    out = {"protein_ids": [], "logits": [], "labels": []}
    for a, r in rows.iterrows():
        n = len(r["aa_sequence"])
        logits = active_logits(r["predict_active_prob"])
        if len(logits) != n:
            raise ValueError(f"{a}: {len(logits)} probabilities for {n} residues")
        out["protein_ids"].append(a)
        out["logits"].append(logits)
        out["labels"].append(label_fn(labels_from.loc[a, "site_labels"], n))
    return out
