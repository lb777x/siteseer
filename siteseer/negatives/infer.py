"""Score proteins with a trained probe for the negative control: raw residue logits and per-protein summaries.

No fallback: a protein without a residue above the threshold stays empty. Groups: mcsa_valid and mcsa_test (always;
M-CSA valid only calibrates tau = best pooled F1), --positives (further enzyme sets, NAME=CSV[@dataset_flag]) and
--negatives (a set CSV with accession, sequence, set, split_role, ...). Sequences longer than the embedder's limit
are truncated and flagged.

Writes to --out-dir: raw_<group>.pt {id: {"logits", "labels" (None for negatives), "truncated"[, "type_logits"]}},
per_protein.csv (max probability, top-3 mean, residues above 0.5 and above tau) and run_info.json. Resumable.

  python -m siteseer.negatives.infer --checkpoint checkpoints/main_stage2_mcsa/best.pt --embedder esm2_650m_l30 \\
      --negatives dataset/negatives/negatives_abc.csv --out-dir results/negatives/main_abc
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from siteseer.data.labels import parse_site_labels, row_label_source
from siteseer.embeddings.embedders import PRESETS, get_embedder
from siteseer.eval.metrics import find_best_threshold
from siteseer.models.probe import load_probe

MCSA = "dataset/mcsa_fine_tune/normal_mcsa"
SAVE_EVERY = 50
NEG_COLUMNS = ["set", "origin_set", "split_role", "identity_band", "homology_type"]


def negative_items(path):
    d = pd.read_csv(path)
    ok = d["sequence"].apply(lambda s: isinstance(s, str) and len(s) > 0)
    if (~ok).any():
        print(f"[!] {path}: {int((~ok).sum())} rows without a sequence skipped", flush=True)
    return [(r.accession, r.sequence, None) for r in d[ok].itertuples()], d


def enzyme_items(path, max_len, flag=None):
    """(id, sequence, labels) per protein (first row per id); proteins without a labelled residue in range skipped."""
    d = pd.read_csv(path, low_memory=False)
    if flag is not None:
        d = d[d["dataset_flag"] == flag]
    items, skipped = [], {"no_sequence": 0, "invalid_labels": 0, "no_positive_in_range": 0}
    for _, row in d.drop_duplicates(subset="alphafolddb-id").iterrows():
        seq = row.get("aa_sequence")
        if not isinstance(seq, str) or not seq:
            skipped["no_sequence"] += 1
            continue
        idx = parse_site_labels(row["site_labels"], row_label_source(row))
        if idx is None:
            skipped["invalid_labels"] += 1
            continue
        y = np.zeros(min(len(seq), max_len), dtype=np.float32)
        for i in idx:
            if 0 <= i - 1 < len(y):
                y[i - 1] = 1.0
        if not y.any():
            skipped["no_positive_in_range"] += 1
            continue
        items.append((str(row["alphafolddb-id"]).strip(), seq, y))
    return items, skipped


def run_group(name, items, embedder, probe, out_dir, device):
    path = os.path.join(out_dir, f"raw_{name}.pt")
    res = torch.load(path, weights_only=False) if os.path.exists(path) else {}
    todo = [it for it in items if it[0] not in res]
    print(f"[{name}] {len(items)} proteins, {len(res)} done, {len(todo)} to score", flush=True)
    every = max(SAVE_EVERY, len(todo) // 50)
    for k, (pid, seq, y) in enumerate(todo, 1):
        truncated = len(seq) > embedder.max_len
        emb = embedder.embed(seq[:embedder.max_len]).to(device)
        with torch.no_grad():
            if probe.n_site_types:
                logits, types = probe(emb, return_types=True)
                logits, types = logits.float().cpu(), types.float().cpu()
            else:
                logits, types = probe(emb).float().cpu(), None
        res[pid] = {"logits": logits, "labels": None if y is None else torch.from_numpy(y), "truncated": truncated}
        if types is not None:
            res[pid]["type_logits"] = types
        if k % every == 0 or k == len(todo):
            torch.save(res, path + ".tmp")
            os.replace(path + ".tmp", path)
            print(f"  {k}/{len(todo)}", flush=True)
    return res


def summarize(pid, group, logits, taus):
    p = torch.sigmoid(logits).numpy()
    top = np.sort(p)[::-1]
    row = {"id": pid, "group": group, "length": len(p), "max_prob": float(top[0]),
           "top3_mean": float(top[:3].mean()), "argmax_residue": int(p.argmax()) + 1}
    for tag, t in taus.items():
        hit = np.nonzero(p > t)[0]
        row[f"n_over_{tag}"] = int(len(hit))
        row[f"frac_over_{tag}"] = float(len(hit) / len(p))
        row[f"residues_over_{tag}"] = ";".join(str(i + 1) for i in hit)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--embedder", default="esm2_650m_l30", choices=sorted(PRESETS))
    ap.add_argument("--negatives", default="", help="a set CSV; empty for positives only")
    ap.add_argument("--positives", action="append", default=[], metavar="NAME=CSV[@FLAG]")
    ap.add_argument("--mcsa-test", default=f"{MCSA}/test_dataset/mcsa_test.csv")
    ap.add_argument("--mcsa-valid", default=f"{MCSA}/valid_dataset/mcsa_valid.csv")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    embedder = get_embedder(args.embedder)
    probe, ckpt = load_probe(args.checkpoint, device)
    embedder.load(device)

    groups, skipped = {}, {}
    groups["mcsa_valid"], skipped["mcsa_valid"] = enzyme_items(args.mcsa_valid, embedder.max_len)
    groups["mcsa_test"], skipped["mcsa_test"] = enzyme_items(args.mcsa_test, embedder.max_len)
    for spec in args.positives:
        name, path = spec.split("=", 1)
        path, flag = (path.split("@", 1) + [None])[:2]
        groups[name], skipped[name] = enzyme_items(path, embedder.max_len, flag)
    neg_df = None
    if args.negatives:
        groups["negatives"], neg_df = negative_items(args.negatives)
    raw = {g: run_group(g, items, embedder, probe, args.out_dir, device) for g, items in groups.items()}

    v = raw["mcsa_valid"]
    tau = round(find_best_threshold([v[i]["logits"] for i in v], [v[i]["labels"] for i in v]), 2)
    taus = {"05": 0.5, "tauvalid": tau}
    meta = None if neg_df is None else neg_df.set_index("accession")[NEG_COLUMNS]
    rows = []
    for group, res in raw.items():
        for pid, r in res.items():
            row = summarize(pid, group, r["logits"], taus)
            row["truncated"] = r["truncated"]
            if r["labels"] is not None:
                row["n_true_sites"] = int(r["labels"].sum())
            if group == "negatives":
                row.update(meta.loc[pid].to_dict())
            rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "per_protein.csv"), index=False)
    info = {"checkpoint": args.checkpoint, "run_name": ckpt["config"].get("run_name"), "embedder": args.embedder,
            "seed": args.seed, "tau_valid": tau, "negatives_csv": args.negatives, "positives": args.positives,
            "counts": {g: len(r) for g, r in raw.items()}, "skipped": skipped,
            "runtime_min": round((time.time() - t0) / 60, 1)}
    with open(os.path.join(args.out_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[+] {args.out_dir}: {info['counts']}, tau {tau}", flush=True)


if __name__ == "__main__":
    main()
