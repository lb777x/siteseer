"""Average several models' predictions on the same proteins and score the result.

Inputs are --save-predictions files from compass.eval.evaluate for models that were
evaluated on the SAME split (e.g. one per seed). Proteins present in every file are
kept; probabilities (not logits) are averaged per residue, since the members are
separately trained and their logit scales need not agree. Reports each member's own
metrics, the spread across members (mean, std, min, max), and the ensemble's metrics
with bootstrap CIs -- so "the ensemble helps" can be read against the seed noise.

  python -m compass.eval.ensemble --predictions a.pt b.pt c.pt [--threshold 0.5]
"""
import argparse
import json

import numpy as np
import torch

from compass.eval.metrics import binary_metrics, bootstrap_ci, format_bootstrap_result

KEYS = ("precision", "recall", "f1", "mcc", "auprc")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--predictions", nargs="+", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--out-json", default=None)
    p.add_argument("--save-ensemble", default=None,
                   help="write the averaged predictions in --save-predictions format, so the ensemble "
                        "can be fed to compass.eval.paired_bootstrap like any other model")
    args = p.parse_args()

    members = []
    for path in args.predictions:
        d = torch.load(path, weights_only=False)
        members.append({str(i).strip(): (l, y) for i, l, y in zip(d["protein_ids"], d["logits"], d["labels"])})
        print(f"[*] {path}: {len(d['protein_ids'])} proteins")
    ids = sorted(set.intersection(*(set(m) for m in members)))
    print(f"[+] {len(members)} members, {len(ids)} proteins in all of them")
    labels = [members[0][i][1] for i in ids]
    for m in members[1:]:
        for k, i in enumerate(ids):
            if not torch.equal(m[i][1], labels[k]):
                raise RuntimeError(f"labels differ for {i} between members")

    out = {"n_members": len(members), "n_proteins": len(ids), "threshold": args.threshold, "members": {}}
    print(f"\n{'member':44s} {'pooled F1':>10s} {'macro F1':>10s} {'pooled AUPRC':>13s} {'macro AUPRC':>12s}")
    per_member = []
    for path, m in zip(args.predictions, members):
        lg = [m[i][0] for i in ids]
        pooled = binary_metrics(torch.cat(lg), torch.cat(labels), args.threshold)
        macro = {k: float(np.mean([binary_metrics(l, y, args.threshold)[k] for l, y in zip(lg, labels)])) for k in KEYS}
        per_member.append((pooled, macro))
        out["members"][path] = {"pooled": {k: pooled[k] for k in KEYS}, "macro": macro}
        print(f"{path.split('/')[-1][:44]:44s} {pooled['f1']:10.3f} {macro['f1']:10.3f} "
              f"{pooled['auprc']:13.3f} {macro['auprc']:12.3f}")

    print("\nspread across members (the noise floor for any later comparison):")
    out["spread"] = {}
    for scope, idx in (("pooled", 0), ("macro", 1)):
        for k in ("f1", "mcc", "auprc"):
            v = np.array([m[idx][k] for m in per_member])
            out["spread"][f"{scope}_{k}"] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                                             "min": float(v.min()), "max": float(v.max())}
            print(f"  {scope} {k:8s} mean {v.mean():.3f}  std {v.std(ddof=1) if len(v) > 1 else 0:.3f}  "
                  f"range [{v.min():.3f}, {v.max():.3f}]")

    # ensemble: average probabilities, convert back to a logit for the shared metric code
    ens = []
    for i in ids:
        pr = torch.stack([torch.sigmoid(m[i][0]) for m in members]).mean(0).clamp(1e-6, 1 - 1e-6)
        ens.append(torch.log(pr / (1 - pr)))
    res = bootstrap_ci(ens, labels, n_bootstrap=args.n_bootstrap, threshold=args.threshold)
    print(f"\n--- ensemble of {len(members)} members, threshold={args.threshold} ---")
    print(format_bootstrap_result(res))
    out["ensemble"] = {"pooled": {k: res["point"][k] for k in KEYS}, "macro": {k: res["macro_point"][k] for k in KEYS},
                       "pooled_f1_ci": res["f1_ci"], "macro_f1_ci": res["macro_f1_ci"],
                       "pooled_auprc_ci": res["auprc_ci"], "macro_auprc_ci": res["macro_auprc_ci"]}
    if args.save_ensemble:
        torch.save({"protein_ids": ids, "logits": [l for l in ens], "labels": labels}, args.save_ensemble)
        print(f"[+] wrote {args.save_ensemble} ({len(ids)} proteins)")
    if args.out_json:
        json.dump(out, open(args.out_json, "w"), indent=2)
        print(f"[+] wrote {args.out_json}")


if __name__ == "__main__":
    main()
