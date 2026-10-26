"""Negative control: does the model flag proteins without an active site? Numbers for figures 3.18-3.20.

Protein score = maximum residue probability. Threshold: Youden on the validation part of set G (split by 40%
family) against M-CSA valid; reported on the report parts of the non-enzyme sets and on the enzyme test sets
(never trained on). Shares with 95% Wilson intervals; ROC-AUC of G (report part) against each enzyme test set with a
bootstrap CI (1,000 draws, seed 42, resampling within classes).

  python -m siteseer.negatives.analyze --abc results/negatives/main_abc --gold results/negatives/main_gold \\
      --pos results/negatives/main_pos --gold-csv dataset/negatives/negatives_gold.csv --out-json negatives.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SEED = 42
ENZYMES = [("reallabels_test", "Real-labels test"), ("goliath_test", "Goliath test"), ("mcsa_test", "M-CSA test")]
NEGATIVES = [("G", "G: confident non-enzymes"), ("A", "A: non-enzymes by annotation"),
             ("C", "C: enzyme-like (>=50% identity)"), ("B", "B: with binding sites (alarm intended)")]


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def youden(pos_val, neg_val):
    """Threshold maximising sensitivity + specificity over all observed scores."""
    best_j, t_j = -1, 0.5
    for t in np.unique(np.r_[pos_val, neg_val, 0.0, 1.0]):
        j = (pos_val >= t).mean() + (neg_val < t).mean() - 1
        if j > best_j:
            best_j, t_j = j, float(t)
    return t_j


def auroc_ci(pos, neg, n=1000):
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos, neg]
    rng = np.random.default_rng(SEED)
    ip, ineg = np.nonzero(y == 1)[0], np.nonzero(y == 0)[0]
    boot = [roc_auc_score(y[i], s[i]) for i in
            (np.concatenate([rng.choice(ip, len(ip)), rng.choice(ineg, len(ineg))]) for _ in range(n))]
    return float(roc_auc_score(y, s)), [float(x) for x in np.percentile(boot, [2.5, 97.5])]


def share(scores, t):
    k, n = int((scores >= t).sum()), len(scores)
    lo, hi = wilson(k, n)
    return {"n": n, "k": k, "rate_pct": 100 * k / n, "ci_pct": [100 * lo, 100 * hi]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--abc", required=True, help="infer.py output for sets A/B/C")
    ap.add_argument("--gold", required=True, help="infer.py output for set G")
    ap.add_argument("--pos", required=True, help="infer.py output for the enzyme test sets")
    ap.add_argument("--gold-csv", default="dataset/negatives/negatives_gold.csv")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    read = lambda d: pd.read_csv(os.path.join(d, "per_protein.csv"))
    abc, gold, pos = read(args.abc), read(args.gold), read(args.pos)
    neg = pd.concat([d[d["group"] == "negatives"] for d in (abc, gold)], ignore_index=True)
    enz = {g: pos[pos["group"] == g]["max_prob"].to_numpy() for g, _ in ENZYMES}
    info = json.load(open(os.path.join(args.gold, "run_info.json")))

    g_val = neg[(neg["set"] == "G") & (neg["split_role"] == "valid")]["max_prob"].to_numpy()
    t = youden(pos[pos["group"] == "mcsa_valid"]["max_prob"].to_numpy(), g_val)
    g_all = neg[neg["set"] == "G"]["max_prob"]
    rep = neg[neg["split_role"] == "report"]
    g_rep = rep[rep["set"] == "G"]
    out = {"threshold_rule": "Youden on the G validation part (split by family) + M-CSA valid", "threshold": t,
           "score": "maximum probability over all residues", "model": info.get("run_name"),
           "embedder": info["embedder"],
           "score_dist": {"threshold": t, "G_n": len(g_all), "G_below_pct": 100 * (g_all < t).mean(),
                          "enzymes_n": int(sum(len(v) for v in enz.values())),
                          "enzymes_above_pct": {g: 100 * (v >= t).mean() for g, v in enz.items()}},
           "rejection": [{"label": lab, **share(enz[g], t)} for g, lab in ENZYMES]
                        + [{"label": lab, **share(rep[rep["set"] == s]["max_prob"].to_numpy(), t)}
                           for s, lab in NEGATIVES],
           "auroc_G_report_vs": {g: dict(zip(("auroc", "ci95"), auroc_ci(v, g_rep["max_prob"].to_numpy())))
                                 for g, v in enz.items()}}
    cat = g_rep.join(pd.read_csv(args.gold_csv).set_index("accession")[["kw_category"]], on="id")
    stats = [{"category": c, **share(d["max_prob"].to_numpy(), t)} for c, d in cat.groupby("kw_category")]
    out["categories"] = sorted(stats, key=lambda r: r["rate_pct"])
    out["G_report_rate_pct"] = 100 * (g_rep["max_prob"] >= t).mean()
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"threshold {t:.4f}; G report flagged {out['G_report_rate_pct']:.1f}%")
    for r in out["rejection"]:
        print(f"  {r['label']:42s} {r['k']:5d}/{r['n']:5d} = {r['rate_pct']:.1f}%")
    for g, r in out["auroc_G_report_vs"].items():
        print(f"  AUROC G vs {g}: {r['auroc']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]")


if __name__ == "__main__":
    main()
