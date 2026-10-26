"""Per-residue metrics. Pooled: one confusion matrix over all residues; macro: per protein, then averaged.
AUPRC is threshold-free, the other metrics use a probability threshold (0.5 unless calibrated). Bootstrap CIs
resample proteins.
"""
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef, precision_score, recall_score

KEYS = ("precision", "recall", "f1", "mcc", "auprc")


def binary_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict:
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    preds = (probs > threshold).astype(int)
    y = labels.detach().cpu().numpy().astype(int)
    both = len(np.unique(y)) > 1
    return {
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "mcc": matthews_corrcoef(y, preds) if both else 0.0,
        "auprc": average_precision_score(y, probs) if both else 0.0,
        "n_residues": len(y),
        "n_positive": int(y.sum()),
    }


def find_best_threshold(per_protein_logits: list, per_protein_labels: list,
                        thresholds: Optional[np.ndarray] = None) -> float:
    """The threshold maximising pooled F1, to be fitted on the validation split and applied to the test split."""
    thresholds = np.linspace(0.01, 0.99, 99) if thresholds is None else thresholds
    logits, labels = torch.cat(per_protein_logits), torch.cat(per_protein_labels)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = binary_metrics(logits, labels, float(t))["f1"]
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def bootstrap_ci(per_protein_logits: list, per_protein_labels: list, n_bootstrap: int = 1000, ci: float = 0.95,
                 threshold: float = 0.5, seed: int = 42) -> dict:
    """Pooled and macro point estimates with percentile bootstrap CIs over proteins."""
    n = len(per_protein_logits)
    point = binary_metrics(torch.cat(per_protein_logits), torch.cat(per_protein_labels), threshold)
    per_protein = [binary_metrics(per_protein_logits[i], per_protein_labels[i], threshold) for i in range(n)]
    macro_point = {k: float(np.mean([m[k] for m in per_protein])) for k in KEYS}
    macro_point["n_residues"], macro_point["n_positive"] = point["n_residues"], point["n_positive"]

    rng = np.random.default_rng(seed)
    boot = {k: [] for k in KEYS}
    macro_boot = {k: [] for k in KEYS}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        m = binary_metrics(torch.cat([per_protein_logits[i] for i in idx]),
                           torch.cat([per_protein_labels[i] for i in idx]), threshold)
        for k in KEYS:
            boot[k].append(m[k])
            macro_boot[k].append(float(np.mean([per_protein[i][k] for i in idx])))
    alpha = (1 - ci) / 2
    result = {"point": point, "macro_point": macro_point, "n_proteins": n, "n_bootstrap": n_bootstrap}
    for k in KEYS:
        result[f"{k}_ci"] = tuple(float(x) for x in np.quantile(boot[k], [alpha, 1 - alpha]))
        result[f"macro_{k}_ci"] = tuple(float(x) for x in np.quantile(macro_boot[k], [alpha, 1 - alpha]))
    return result


def format_bootstrap_result(result: dict) -> str:
    p, mp = result["point"], result["macro_point"]
    lines = [f"n_proteins={result['n_proteins']} n_residues={p['n_residues']} n_positive={p['n_positive']}",
             "  pooled (all residues in one confusion matrix):"]
    lines += [f"    {k}: {p[k]:.3f} [{result[f'{k}_ci'][0]:.3f}, {result[f'{k}_ci'][1]:.3f}]" for k in KEYS]
    lines.append("  macro (per-protein metric, averaged over proteins):")
    lines += [f"    {k}: {mp[k]:.3f} [{result[f'macro_{k}_ci'][0]:.3f}, {result[f'macro_{k}_ci'][1]:.3f}]"
              for k in KEYS]
    return "\n".join(lines)
