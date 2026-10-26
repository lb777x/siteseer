"""Per-residue site-type targets (catalytic, binding) with masks, for the site-type outputs of the probe.

The main output stays "any site". The two type outputs are auxiliary, and each label source only supervises the
types it knows:

  source                                 catalytic                     binding
  goliath_original (UniProt via EasIFA)  ACT_SITE groups               BINDING groups   (SITE residues masked)
  m3site, m3site_new                     all groups                    masked
  biolip_* (BioLiP catalytic column)     all groups                    masked
  mcsa_blast_homology, M-CSA             all groups                    masked

Positive-anchored: a type is supervised for a protein only if the protein has at least one residue of that type.
Within a supervised protein, residues outside the type's groups are negatives, including residues of the other type.
A SITE residue that is also in a BINDING or ACT_SITE group keeps that type's positive label.
"""
import ast
from typing import Optional

import torch

from compass.data.labels import parse_site_labels

TYPE_NAMES = ("catalytic", "binding")
CATALYTIC_SOURCES = frozenset({"m3site", "m3site_new", "mcsa_blast_homology", "mcsa",
                               "biolip_exact", "biolip_new", "biolip_blast_homology"})
BINDING_SOURCES = frozenset()   # no source besides goliath_original carries binding labels
GOLIATH_TYPE = {0: "binding", 1: "catalytic", 2: "site"}


def site_type_sets(raw_labels: str, raw_types, label_source: Optional[str]) -> Optional[dict]:
    """{'catalytic': set|None, 'binding': set|None, 'ignore': set} with 1-based residue numbers; None marks a type
    this protein does not supervise. label_source 'mcsa' is used for the M-CSA CSVs (no label_source column).
    Returns None if the labels cannot be parsed."""
    out = {"catalytic": None, "binding": None, "ignore": set()}
    if label_source in CATALYTIC_SOURCES:
        out["catalytic"] = parse_site_labels(raw_labels, label_source)
        return out if out["catalytic"] is not None else None
    if label_source in BINDING_SOURCES:
        out["binding"] = parse_site_labels(raw_labels, label_source)
        return out if out["binding"] is not None else None
    if label_source != "goliath_original":
        raise ValueError(f"no site-type rule for label_source {label_source!r}")
    try:
        groups, types = ast.literal_eval(str(raw_labels)), ast.literal_eval(str(raw_types))
    except (ValueError, SyntaxError):
        return None
    if len(groups) != len(types):
        return None
    sets = {"catalytic": set(), "binding": set(), "site": set()}
    for group, t in zip(groups, types):
        residues = parse_site_labels(str([group]), "goliath_original")
        if residues is None or int(t) not in GOLIATH_TYPE:
            return None
        sets[GOLIATH_TYPE[int(t)]] |= residues
    for name in TYPE_NAMES:
        out[name] = sets[name] or None
    out["ignore"] = sets["site"] - sets["catalytic"] - sets["binding"]
    return out


def type_targets(length: int, sets: dict):
    """-> targets [L, 2] and mask [L, 2] (float), columns in TYPE_NAMES order."""
    targets = torch.zeros(length, len(TYPE_NAMES))
    mask = torch.zeros(length, len(TYPE_NAMES))
    for k, name in enumerate(TYPE_NAMES):
        positives = sets.get(name)
        if not positives:
            continue
        mask[:, k] = 1.0
        for r in positives:
            if 1 <= r <= length:
                targets[r - 1, k] = 1.0
        for r in sets.get("ignore", ()):
            if 1 <= r <= length:
                mask[r - 1, k] = 0.0
    return targets, mask
