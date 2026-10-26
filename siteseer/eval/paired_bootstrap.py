"""Paired bootstrap of two models' saved predictions on the proteins both scored.

  python -m siteseer.eval.paired_bootstrap --a A_fold*.pt --b B_fold*.pt [--n-bootstrap 2000] [--out-json out.json]

Each draw resamples proteins once and scores both models on that sample. Reported per metric (pooled and macro
AUPRC, F1, MCC, precision, recall): both point estimates, B - A with its 95% CI, and the share of draws with B > A.
"""
import argparse
import json

import numpy as np
import torch

from sklearn.metrics import average_precision_score

from siteseer.eval.metrics import binary_metrics

KEYS = ("auprc", "f1", "mcc", "precision", "recall")


def load_predictions(paths):
    ids, logits, labels = [], [], []
    for path in paths:
        d = torch.load(path, weights_only=False)
        ids.extend(str(i).strip() for i in d["protein_ids"])
        logits.extend(d["logits"])
        labels.extend(d["labels"])
        print(f"[*] {path}: {len(d['protein_ids'])} proteins")
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate protein ids within one side -- the files are not a partition")
    return dict(zip(ids, zip(logits, labels)))


def per_protein(preds, ids, threshold):
    return [binary_metrics(preds[i][0], preds[i][1], threshold) for i in ids]


def pooled(preds, ids, threshold):
    return binary_metrics(torch.cat([preds[i][0] for i in ids]), torch.cat([preds[i][1] for i in ids]), threshold)


def confusion_counts(preds, ids, threshold):
    """Per protein (tp, fp, fn, tn) at the threshold; pooled metrics of a draw are sums of rows."""
    rows = np.empty((len(ids), 4))
    for k, i in enumerate(ids):
        y = preds[i][1].numpy().astype(bool)
        p = (torch.sigmoid(preds[i][0]).numpy() > threshold)
        rows[k] = [(p & y).sum(), (p & ~y).sum(), (~p & y).sum(), (~p & ~y).sum()]
    return rows


def metrics_from_counts(tp, fp, fn, tn):
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "mcc": mcc}


def pooled_auprc(preds, ids):
    return float(average_precision_score(np.concatenate([preds[i][1].numpy() for i in ids]),
                                         np.concatenate([preds[i][0].numpy() for i in ids])))


class WeightedAUPRC:
    """Pooled AUPRC of a draw from protein multiplicities, on one pre-sorted residue list (equal to
    average_precision_score on the duplicated residues; checked on the first draws).
    """

    def __init__(self, preds, ids):
        s = np.concatenate([preds[i][0].numpy().astype(np.float64) for i in ids])
        y = np.concatenate([preds[i][1].numpy().astype(np.float64) for i in ids])
        prot = np.concatenate([np.full(len(preds[i][1]), k) for k, i in enumerate(ids)])
        order = np.argsort(-s, kind="mergesort")
        self.y, self.prot = y[order], prot[order]
        s = s[order]
        self.last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]   # last index of every tie group

    def __call__(self, counts):
        w = counts[self.prot]
        tp = np.cumsum(w * self.y)[self.last]
        fp = np.cumsum(w * (1 - self.y))[self.last]
        if tp[-1] == 0:
            return 0.0
        precision = tp / np.maximum(tp + fp, 1e-12)
        recall = tp / tp[-1]
        return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--a", nargs="+", required=True, help="--save-predictions file(s) of model A")
    p.add_argument("--b", nargs="+", required=True, help="--save-predictions file(s) of model B")
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--threshold-a", type=float, default=0.5)
    p.add_argument("--threshold-b", type=float, default=0.5)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--restrict-ids", default=None,
                   help="restrict the comparison to these protein ids (one per line)")
    p.add_argument("--drop-label-mismatch", action="store_true",
                   help="leave out proteins whose label vectors differ between the sides (e.g. structure vs. UniProt "
                        "sequence in the structural ablation) instead of aborting")
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    A, B = load_predictions(args.a), load_predictions(args.b)
    ids = sorted(set(A) & set(B))
    if args.restrict_ids:
        keep = {l.strip() for l in open(args.restrict_ids) if l.strip()}
        ids = [i for i in ids if i in keep]
        print(f"[*] --restrict-ids: {len(ids)} of the shared proteins are in {args.restrict_ids}")
    print(f"[+] {args.label_a}: {len(A)} proteins, {args.label_b}: {len(B)} proteins, "
          f"intersection: {len(ids)} (only {args.label_a}: {len(set(A) - set(B))}, "
          f"only {args.label_b}: {len(set(B) - set(A))})")
    if not ids:
        raise RuntimeError("no shared protein ids")
    mismatch = [i for i in ids if A[i][1].shape != B[i][1].shape or not torch.equal(A[i][1], B[i][1])]
    if mismatch and not args.drop_label_mismatch:
        raise RuntimeError(f"labels differ for {mismatch[0]} between the two sides -- not the same eval "
                           f"data ({len(mismatch)} proteins; --drop-label-mismatch leaves them out)")
    if mismatch:
        ids = [i for i in ids if i not in set(mismatch)]
        print(f"[!] {len(mismatch)} proteins with different label vectors left out "
              f"(residue axes differ, e.g. structure vs. UniProt sequence): {', '.join(mismatch[:10])}"
              f"{' ...' if len(mismatch) > 10 else ''}; paired on {len(ids)}")

    ppA, ppB = per_protein(A, ids, args.threshold_a), per_protein(B, ids, args.threshold_b)
    n = len(ids)
    cA, cB = confusion_counts(A, ids, args.threshold_a), confusion_counts(B, ids, args.threshold_b)
    mA = np.array([[m[k] for k in KEYS] for m in ppA])  # [n, 5] per-protein metrics, for the macro draws
    mB = np.array([[m[k] for k in KEYS] for m in ppB])
    rng = np.random.default_rng(args.seed)
    draws = {f"{scope}_{k}": [] for scope in ("pooled", "macro") for k in KEYS}
    wA, wB = WeightedAUPRC(A, ids), WeightedAUPRC(B, ids)
    for d in range(args.n_bootstrap):
        idx = rng.integers(0, n, size=n)
        counts = np.bincount(idx, minlength=n).astype(np.float64)
        pa = metrics_from_counts(*cA[idx].sum(0))
        pb = metrics_from_counts(*cB[idx].sum(0))
        pa["auprc"], pb["auprc"] = wA(counts), wB(counts)
        if d < 3:   # the fast path must reproduce the explicit one exactly
            sub = [ids[j] for j in idx]
            for got, side in ((pa["auprc"], A), (pb["auprc"], B)):
                ref = pooled_auprc(side, sub)
                if abs(got - ref) > 1e-9:
                    raise RuntimeError(f"weighted AUPRC {got} != explicit {ref} -- fast path broken")
        macro_a, macro_b = mA[idx].mean(0), mB[idx].mean(0)
        for j, k in enumerate(KEYS):
            draws[f"pooled_{k}"].append(pb[k] - pa[k])
            draws[f"macro_{k}"].append(float(macro_b[j] - macro_a[j]))

    point = {"pooled": (pooled(A, ids, args.threshold_a), pooled(B, ids, args.threshold_b)),
             "macro": ({k: float(np.mean([m[k] for m in ppA])) for k in KEYS},
                       {k: float(np.mean([m[k] for m in ppB])) for k in KEYS})}
    out = {"label_a": args.label_a, "label_b": args.label_b, "n_shared_proteins": n,
           "threshold_a": args.threshold_a, "threshold_b": args.threshold_b, "n_bootstrap": args.n_bootstrap,
           "n_residues": int(point["pooled"][0]["n_residues"]), "n_positive": int(point["pooled"][0]["n_positive"]),
           "n_dropped_label_mismatch": len(mismatch), "dropped_ids": mismatch, "metrics": {}}
    print(f"\nn_proteins={n}  n_residues={out['n_residues']}  n_positive={out['n_positive']}  "
          f"thresholds {args.label_a}={args.threshold_a} {args.label_b}={args.threshold_b}")
    print(f"{'metric':18s} {args.label_a:>9s} {args.label_b:>9s} {'delta (B-A)':>12s} {'95% CI':>18s} {'P(delta>0)':>11s}")
    for scope in ("pooled", "macro"):
        for k in KEYS:
            d = np.array(draws[f"{scope}_{k}"])
            lo, hi = np.quantile(d, [0.025, 0.975])
            a, b = point[scope][0][k], point[scope][1][k]
            frac = float((d > 0).mean())
            out["metrics"][f"{scope}_{k}"] = {"a": a, "b": b, "delta": b - a, "ci95": [float(lo), float(hi)],
                                              "frac_delta_gt_0": frac}
            print(f"{scope + ' ' + k:18s} {a:9.3f} {b:9.3f} {b - a:+12.3f} [{lo:+.3f}, {hi:+.3f}] {frac:11.3f}")
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[+] wrote {args.out_json}")


if __name__ == "__main__":
    main()
