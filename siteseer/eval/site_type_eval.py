"""Evaluate the site-type outputs (catalytic, binding) of a probe trained with probe.n_site_types>0.

The main "any site" output is evaluated as always by linear_probe_eval.py (and compared to Flat
with paired_bootstrap.py); this script covers what that one cannot see:

  per type      pooled AUPRC/AUROC over the residues the type supervises (masks from
                siteseer.data.site_types), and macro AUPRC over proteins with >= 1 positive and
                >= 1 negative of that type
  distinction   among proteins that supervise BOTH types, residues that are catalytic-only vs
                binding-only: AUROC of telling them apart (catalytic = positive class), scored by
                  * cat - bind     the type outputs' logit difference
                  * cat            the catalytic output alone
                  * any            the main output (reference: trained to call both "site")
                  * baseline any   the main output of --baseline-checkpoint (e.g. the Flat model),
                                   which may already rank catalytic residues above binding ones;
                                   the number the type outputs have to beat
Point estimates plus a protein-level bootstrap CI for the distinction AUROCs.

  python -m siteseer.eval.site_type_eval --config configs/sitetypes/stage1.yaml --encoder esm_raw \\
      --probe-checkpoint checkpoints/<run>/best.pt --split test \\
      [--baseline-checkpoint checkpoints/<flat run>/best.pt] \\
      [--save-predictions PATH.pt] [--out-json PATH.json]
"""
import argparse
import json

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from siteseer.data.datasets import build_dataset
from siteseer.data.site_types import TYPE_NAMES
from siteseer.embeddings.embedders import embedder_from_config
from siteseer.models.probe import load_probe
from siteseer.train.train_probe import ENCODERS, load_config


def load(path, device):
    probe, ckpt = load_probe(path, device)
    print(f"[*] {path}: run_name={ckpt['config'].get('run_name')} epoch={ckpt.get('epoch')} "
          f"val_auprc={ckpt.get('val_auprc')}")
    return probe


def auc_pair(y, s):
    if len(y) == 0 or y.min() == y.max():
        return None
    return float(average_precision_score(y, s)), float(roc_auc_score(y, s))


def distinction(rows, key, n_boot, rng):
    """AUROC catalytic-only (1) vs binding-only (0) residues, pooled; bootstrap over proteins."""
    def score(idx):
        y = np.concatenate([rows[i]["dy"] for i in idx])
        s = np.concatenate([rows[i][key] for i in idx])
        return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan
    idx = [i for i, r in enumerate(rows) if len(r["dy"])]
    if not idx:
        return None
    point = score(idx)
    boots = [score(rng.choice(idx, len(idx))) for _ in range(n_boot)]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"auroc": float(point), "ci95": [float(lo), float(hi)]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the run's YAML (data section; probe.n_site_types>0)")
    ap.add_argument("--probe-checkpoint", required=True)
    ap.add_argument("--baseline-checkpoint", default=None)
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-predictions", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--encoder", "--baseline-encoder", dest="encoder", required=True, choices=ENCODERS)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    if cfg["probe"].get("n_site_types", 0) != len(TYPE_NAMES):
        raise SystemExit(f"--config must set probe.n_site_types: {len(TYPE_NAMES)} (it decides the dataset's targets)")
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = load(args.probe_checkpoint, device)
    if not probe.n_site_types:
        raise SystemExit("--probe-checkpoint has no site-type outputs")
    base = load(args.baseline_checkpoint, device) if args.baseline_checkpoint else None

    embedder = None
    if cfg["data"].get("dataset_class", "mcsa") == "mcsa":
        embedder = embedder_from_config(cfg["data"], args.encoder).load(device)
    ds = build_dataset(cfg, args.split, embedder)

    rows, saved = [], {"protein_ids": [], "logits": [], "type_logits": [], "labels": [], "type_targets": [],
                       "type_mask": [], "baseline_logits": []}
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            emb = item["embedding"].to(device)
            any_logit, types = probe(emb, return_types=True)
            any_logit, types = any_logit.float().cpu(), types.float().cpu()
            b = base(emb).float().cpu() if base is not None else None
            t, m, y = item["type_targets"], item["type_mask"], item["labels"]
            both = bool(m[:, 0].any() and m[:, 1].any())
            r = {"id": item["protein_id"], "any": any_logit.numpy(), "y": y.numpy(),
                 "types": types.numpy(), "t": t.numpy(), "m": m.numpy(), "dy": np.array([])}
            if both:
                sel = (m[:, 0] > 0) & (m[:, 1] > 0) & (t[:, 0] != t[:, 1])   # catalytic-only or binding-only
                r["dy"] = t[sel, 0].numpy()
                r["cat-bind"] = (types[sel, 0] - types[sel, 1]).numpy()
                r["cat"] = types[sel, 0].numpy()
                r["any_d"] = any_logit[sel].numpy()
                if b is not None:
                    r["base_d"] = b[sel].numpy()
            rows.append(r)
            if args.save_predictions:
                for k, v in (("protein_ids", item["protein_id"]), ("logits", any_logit), ("type_logits", types),
                             ("labels", y), ("type_targets", t), ("type_mask", m), ("baseline_logits", b)):
                    saved[k].append(v)

    out = {"split": args.split, "config": args.config, "checkpoint": args.probe_checkpoint,
           "baseline": args.baseline_checkpoint, "n_proteins": len(rows)}
    ys, ss = np.concatenate([r["y"] for r in rows]), np.concatenate([r["any"] for r in rows])
    out["any"] = dict(zip(("pooled_auprc", "pooled_auroc"), auc_pair(ys, ss) or (None, None)))
    print(f"[{args.split}] {len(rows)} proteins | any site: pooled AUPRC {out['any']['pooled_auprc']}")
    for k, name in enumerate(TYPE_NAMES):
        sup = [r for r in rows if r["m"][:, k].any()]
        if not sup:
            print(f"  {name}: not supervised in this split")
            continue
        y = np.concatenate([r["t"][r["m"][:, k] > 0, k] for r in sup])
        s = np.concatenate([r["types"][r["m"][:, k] > 0, k] for r in sup])
        pooled = auc_pair(y, s)
        macro = [average_precision_score(r["t"][r["m"][:, k] > 0, k], r["types"][r["m"][:, k] > 0, k])
                 for r in sup if 0 < r["t"][r["m"][:, k] > 0, k].sum() < (r["m"][:, k] > 0).sum()]
        out[name] = {"n_proteins": len(sup), "n_residues": int(len(y)), "n_positive": int(y.sum()),
                     "pooled_auprc": pooled[0] if pooled else None, "pooled_auroc": pooled[1] if pooled else None,
                     "macro_auprc": float(np.mean(macro)) if macro else None}
        f3 = lambda v: "n/a" if v is None else f"{v:.3f}"
        print(f"  {name:10s} {len(sup)} proteins, {int(y.sum())} positive: pooled AUPRC {f3(out[name]['pooled_auprc'])} "
              f"AUROC {f3(out[name]['pooled_auroc'])} | macro AUPRC {f3(out[name]['macro_auprc'])}")

    rng = np.random.default_rng(args.seed)
    both = [r for r in rows if len(r["dy"])]
    if both:
        n_cat = int(sum(r["dy"].sum() for r in both))
        n_all = int(sum(len(r["dy"]) for r in both))
        print(f"  distinction catalytic-only ({n_cat}) vs binding-only ({n_all - n_cat}) residues, "
              f"{len(both)} proteins supervising both:")
        out["distinction"] = {"n_proteins": len(both), "n_catalytic": n_cat, "n_binding": n_all - n_cat}
        keys = [("cat-bind", "type outputs, cat - bind"), ("cat", "catalytic output"),
                ("any_d", "main output (reference)")]
        if base is not None:
            keys.append(("base_d", "baseline main output"))
        for key, label in keys:
            res = distinction(both, key, args.n_bootstrap, rng)
            out["distinction"][key] = res
            print(f"    {label:28s} AUROC {res['auroc']:.3f} [{res['ci95'][0]:.3f}, {res['ci95'][1]:.3f}]")
    else:
        print("  distinction: no protein supervises both types in this split (e.g. M-CSA)")

    if args.save_predictions:
        torch.save(saved, args.save_predictions)
        print(f"[*] saved {len(saved['protein_ids'])} proteins' predictions to {args.save_predictions}")
    if args.out_json:
        json.dump(out, open(args.out_json, "w"), indent=1)
        print(f"[+] wrote {args.out_json}")


if __name__ == "__main__":
    main()
