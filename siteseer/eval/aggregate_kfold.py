"""Pooled metrics over the out-of-fold test predictions of all folds, with a protein bootstrap CI.

With --valid-predictions, a threshold is also fitted on the pooled validation predictions (best pooled F1) and the
pooled test set is reported at it as well. --exclude-ids drops proteins from both (pipeline step 21).

  python -m siteseer.eval.aggregate_kfold --predictions fold{0..4}_test.pt [--valid-predictions fold{0..4}_valid.pt]
"""
import argparse
import json

import torch

from siteseer.eval.metrics import bootstrap_ci, find_best_threshold, format_bootstrap_result


def load_folds(paths: list, label: str, exclude: frozenset = frozenset()) -> tuple:
    ids, logits, labels = [], [], []
    for path in paths:
        fold = torch.load(path, weights_only=False)
        keep = [k for k, i in enumerate(fold["protein_ids"]) if str(i).strip() not in exclude]
        ids.extend(fold["protein_ids"][k] for k in keep)
        logits.extend(fold["logits"][k] for k in keep)
        labels.extend(fold["labels"][k] for k in keep)
        print(f"[*] {len(keep)} {label} proteins from {path} ({len(fold['protein_ids']) - len(keep)} excluded)")
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{len(ids) - len(set(ids))} {label} proteins occur in more than one fold")
    return ids, logits, labels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", nargs="+", required=True, help="each fold's test predictions")
    ap.add_argument("--valid-predictions", nargs="*", default=None, help="each fold's validation predictions")
    ap.add_argument("--exclude-ids", default=None, help="protein ids left out of scoring (one per line)")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    exclude = frozenset(l.strip() for l in open(args.exclude_ids) if l.strip()) if args.exclude_ids else frozenset()

    ids, logits, labels = load_folds(args.predictions, "test", exclude)
    print(f"[+] {len(ids)} out-of-fold test proteins from {len(args.predictions)} folds")
    out = {"n_proteins": len(ids), "threshold_0.5": bootstrap_ci(logits, labels, n_bootstrap=args.n_bootstrap)}
    print("--- threshold 0.5 ---")
    print(format_bootstrap_result(out["threshold_0.5"]))

    if args.valid_predictions:
        _, v_logits, v_labels = load_folds(args.valid_predictions, "valid", exclude)
        t = find_best_threshold(v_logits, v_labels)
        out["calibrated_threshold"] = t
        out["threshold_calibrated"] = bootstrap_ci(logits, labels, n_bootstrap=args.n_bootstrap, threshold=t)
        print(f"--- threshold {t:.3f} (best pooled F1 on the pooled validation folds) ---")
        print(format_bootstrap_result(out["threshold_calibrated"]))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(f"[+] {args.out_json}")


if __name__ == "__main__":
    main()
