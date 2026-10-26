"""Sequence-only site prediction against P2Rank (structure-based pocket prediction) on the same proteins.

Truths: V, non-enzymes with BioLiP small-molecule or metal contacts (all, metal only, organic only; residues of the
other class left out); Goliath test enzymes with UniProt labels (any site, catalytic, binding); M-CSA test.
Scores: P2Rank 2.5.1 (-c alphafold) residue probability on the AlphaFold model; ours_any = site-type model stage 1,
main output; ours_final = main model (M-CSA stage); fixed combinations mean and noisy-OR of ours_any and P2Rank;
a learned combination (logistic regression on both scores, their product and P2Rank pocket features), cross-fitted
on V in two halves split by 40% family (seed 42) and fitted on all of V for the enzyme sets.
Residue level: macro / pooled AUPRC with a paired bootstrap (2,000 draws). Pocket level: P2Rank's pockets ranked by
P2Rank, by our mean probability, or by the mean of both; is a site residue in the top pocket. Protein level: AUROC
of the maximum score, enzymes or V against the gold non-enzymes G.
AlphaFold numbering = UniProt numbering; where the model's sequence differs, a global alignment maps residues.

  python -m siteseer.comparisons.p2rank all --p2rank-dir data/comparisons/p2rank/out --out-dir results/reports/p2rank
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from Bio import Align
from Bio.Align import substitution_matrices
from Bio.Data.PDBData import protein_letters_3to1   # upper-case three-letter codes, as in PDB files

from siteseer.data.site_types import site_type_sets
from siteseer.negatives import biolip

EPS = 1e-6
SCORES = ["p2rank", "ours_any", "ours_final", "mean", "noisy_or"]
FEATURES = ["logit_ours", "logit_p2rank", "product", "pocket_score", "pocket_rank_inv", "pocket_mean_ours", "in_pocket"]
TASKS = [("V", "site"), ("V", "metal"), ("V", "organic"), ("goliath", "site"), ("goliath", "catalytic"),
         ("goliath", "binding"), ("mcsa", "site")]
OTHER = {"metal": "organic", "organic": "metal", "catalytic": "binding", "binding": "catalytic"}
GOLIATH = "dataset/label_diffusion/cached_dataframe_fully_excl_fix1.csv"
MCSA_TEST = "dataset/mcsa_fine_tune/normal_mcsa/test_dataset/mcsa_test.csv"
GLOBAL = Align.PairwiseAligner(mode="global", open_gap_score=-10, extend_gap_score=-0.5)
GLOBAL.substitution_matrix = substitution_matrices.load("BLOSUM62")


def p2rank_of(acc, p2_dir):
    rpath = os.path.join(p2_dir, f"AF-{acc}.pdb_residues.csv")
    if not os.path.exists(rpath):
        return None
    r = pd.read_csv(rpath, skipinitialspace=True)
    r.columns = [c.strip() for c in r.columns]
    pk = pd.read_csv(os.path.join(p2_dir, f"AF-{acc}.pdb_predictions.csv"), skipinitialspace=True)
    pk.columns = [c.strip() for c in pk.columns]
    return {"labels": r["residue_label"].astype(int).tolist(),
            "seq": "".join(protein_letters_3to1.get(str(n).strip().upper(), "X") for n in r["residue_name"]),
            "prob": r["probability"].astype(float).to_numpy(),
            "pockets": [(float(s), [int(x.split("_")[1]) for x in str(ids).split()])
                        for s, ids in zip(pk["score"], pk["residue_ids"])]}


def to_ours(p2, our_seq):
    """P2Rank residue number -> 0-based index in our sequence."""
    if p2["seq"] == our_seq[:len(p2["seq"])] and p2["labels"] == list(range(1, len(p2["seq"]) + 1)):
        return {n: n - 1 for n in p2["labels"]}
    aln = GLOBAL.align(p2["seq"], our_seq)[0]
    m = {}
    for (s1, e1), (s2, _) in zip(*aln.aligned):
        for k in range(e1 - s1):
            m[s1 + k + 1] = s2 + k + 1
    return {p2["labels"][k - 1]: v - 1 for k, v in m.items()}


def metal_organic(cand):
    """accession -> (metal, organic) contact residues, 1-based."""
    seqs = dict(zip(cand.accession, cand.sequence))
    bl = biolip.read()
    bl = bl[bl.uniprot.isin(seqs) & ~bl.lig.isin(biolip.NOT_A_SITE | biolip.ADDITIVES | biolip.GLYCANS |
                                                 biolip.BUFFER_IONS)]
    out, cache = {}, {}
    for r in bl.itertuples():
        if (r.seq, r.uniprot) not in cache:
            cache[(r.seq, r.uniprot)] = biolip.position_map(r.seq, seqs[r.uniprot], clean=True)
        res = biolip.mapped(str(r.bres).split(), cache[(r.seq, r.uniprot)], seqs[r.uniprot])
        mo = out.setdefault(r.uniprot, (set(), set()))
        (mo[0] if r.lig in biolip.METALS else mo[1]).update(res)
    return out


def assemble(args):
    cand = pd.read_csv(args.candidates)
    load = lambda d, f: torch.load(os.path.join(args.scores_dir, d, f), weights_only=False)
    records, skipped = [], {}

    def add(acc, group, seq, truth, any_logits, final_logits):
        p2 = p2rank_of(acc, args.p2rank_dir)
        if p2 is None or len(final_logits) != len(any_logits):
            key = "no_p2rank" if p2 is None else "length_mismatch"
            skipped[key] = skipped.get(key, 0) + 1
            return
        n = len(any_logits)
        idx = to_ours(p2, seq)
        q = np.zeros(n)
        for k, pr in zip(p2["labels"], p2["prob"]):
            if k in idx and idx[k] < n:
                q[idx[k]] = pr
        pockets = [(s, sorted({idx[k] for k in ids if k in idx and idx[k] < n})) for s, ids in p2["pockets"]]
        records.append({"accession": acc, "group": group, "length": n,
                        "truth": {k: sorted(i - 1 for i in v if 1 <= i <= n) for k, v in truth.items()},
                        "p2rank": q.astype(np.float32),
                        "ours_any": torch.sigmoid(any_logits).numpy().astype(np.float32),
                        "ours_final": torch.sigmoid(final_logits).numpy().astype(np.float32),
                        "pockets": [p for p in pockets if p[1]], "aligned": len(idx)})

    v = cand[cand.in_V].set_index("accession")
    mo = metal_organic(cand[cand.in_V])
    ra, rf = load(f"{args.any}_pocket", "raw_negatives.pt"), load(f"{args.final}_pocket", "raw_negatives.pt")
    for acc, r in v.iterrows():
        if acc in ra and acc in rf:
            site = {int(x) for x in str(r.biolip_residues).split(";") if x.isdigit()}
            m, o = mo.get(acc, (set(), set()))
            add(acc, "V", r.sequence, {"site": site, "metal": m - o, "organic": o - m}, ra[acc]["logits"],
                rf[acc]["logits"])
    d = pd.read_csv(GOLIATH, low_memory=False)
    d = d[(d.dataset_flag == "test") & (d.label_source == "goliath_original")]
    d = d.assign(acc=d["alphafolddb-id"].astype(str).str.strip()).drop_duplicates("acc").set_index("acc")
    ra, rf = load(f"{args.any}_pos", "raw_goliath_test.pt"), load(f"{args.final}_pos", "raw_goliath_test.pt")
    for acc, r in d.iterrows():
        if acc in ra and acc in rf:
            s = site_type_sets(r.site_labels, r.site_types, "goliath_original")
            cat, bind = s["catalytic"] or set(), s["binding"] or set()
            add(acc, "goliath", r.aa_sequence, {"site": {i + 1 for i in np.nonzero(ra[acc]["labels"].numpy())[0]},
                                                "catalytic": cat - bind, "binding": bind - cat},
                ra[acc]["logits"], rf[acc]["logits"])
    m = pd.read_csv(MCSA_TEST)
    seqs = dict(zip(m["alphafolddb-id"].astype(str).str.strip(), m.aa_sequence))
    ra, rf = load(f"{args.any}_pos", "raw_mcsa_test.pt"), load(f"{args.final}_pos", "raw_mcsa_test.pt")
    for acc in ra:
        if acc in rf and acc in seqs:
            add(acc, "mcsa", seqs[acc], {"site": {i + 1 for i in np.nonzero(ra[acc]["labels"].numpy())[0]}},
                ra[acc]["logits"], rf[acc]["logits"])
    torch.save(records, os.path.join(args.out_dir, "records.pt"))
    print(f"[+] {len(records)} proteins, skipped {skipped}")


def as_predictions(sub, t, score_of):
    """One --save-predictions dict for a task; residues of the other class are left out, not counted negative."""
    out = {"protein_ids": [], "logits": [], "labels": []}
    for r in sub:
        keep = np.ones(r["length"], dtype=bool)
        if t in OTHER:
            keep[[i for i in r["truth"].get(OTHER[t], []) if i < r["length"]]] = False
        y = np.zeros(r["length"], dtype=np.float32)
        y[r["truth"][t]] = 1.0
        x = np.clip(np.asarray(score_of(r), dtype=np.float64), EPS, 1 - EPS)
        out["protein_ids"].append(r["accession"])
        out["logits"].append(torch.tensor(np.log(x / (1 - x))[keep], dtype=torch.float32))
        out["labels"].append(torch.from_numpy(y[keep]))
    return out


def paired(a, la, b, lb, out, n_bootstrap, extra=()):
    subprocess.run([sys.executable, "-m", "siteseer.eval.paired_bootstrap", "--a", a, "--label-a", la, "--b", b,
                    "--label-b", lb, "--n-bootstrap", str(n_bootstrap), "--out-json", out, *extra],
                   check=True, stdout=subprocess.DEVNULL)
    return json.load(open(out))["metrics"]


def load_records(args):
    recs = torch.load(os.path.join(args.out_dir, "records.pt"), weights_only=False)
    for r in recs:
        p, q = r["ours_any"].astype(np.float64), r["p2rank"].astype(np.float64)
        r["mean"], r["noisy_or"] = (p + q) / 2, 1 - (1 - p) * (1 - q)
    return recs


def evaluate(args):
    recs = load_records(args)
    preds = os.path.join(args.out_dir, "preds")
    os.makedirs(preds, exist_ok=True)
    results = {}
    for group, t in TASKS:
        sub = [r for r in recs if r["group"] == group and r["truth"].get(t)]
        files = {}
        for s in SCORES:
            files[s] = os.path.join(preds, f"{group}_{t}_{s}.pt")
            torch.save(as_predictions(sub, t, lambda r: r[s]), files[s])
        top5 = {s: float(np.mean([np.mean(np.isin(np.argsort(-np.asarray(r[s]))[:5], r["truth"][t])) for r in sub]))
                for s in SCORES}
        pairs = {f"{b} vs {a}": paired(files[a], a, files[b], b, os.path.join(preds, f"{group}_{t}_{a}_vs_{b}.json"),
                                       args.n_bootstrap)
                 for a, b in (("p2rank", "ours_any"), ("p2rank", "ours_final"), ("p2rank", "mean"),
                              ("p2rank", "noisy_or"), ("ours_any", "mean"), ("ours_any", "noisy_or"))}
        results[f"{group}/{t}"] = {"n": len(sub), "top5_precision": top5, "pairs": pairs}
        print(f"[{group}/{t}] {len(sub)} proteins", flush=True)
    pocket = {}
    for group in ("V", "goliath", "mcsa"):
        sub = [r for r in recs if r["group"] == group and r["pockets"] and r["truth"].get("site")]
        res = {"n": len(sub)}
        for name, key in (("P2Rank order", lambda r, pk: pk[0]),
                          ("ours order", lambda r, pk: float(np.mean(r["ours_any"][pk[1]]))),
                          ("combined order", lambda r, pk: float(np.mean(r["mean"][pk[1]])))):
            hit, prec, rec = [], [], []
            for r in sub:
                top, t = set(max(r["pockets"], key=lambda pk: key(r, pk))[1]), set(r["truth"]["site"])
                hit.append(bool(t & top))
                prec.append(len(t & top) / len(top))
                rec.append(len(t & top) / len(t))
            res[name] = {"top1_hit": float(np.mean(hit)), "precision": float(np.mean(prec)), "recall": float(np.mean(rec))}
        res["no_pocket"] = sum(1 for r in recs if r["group"] == group and not r["pockets"])
        pocket[group] = res
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({"residue": results, "pocket": pocket}, f, indent=1)


def features(r):
    n = r["length"]
    ours = np.clip(r["ours_any"].astype(np.float64), EPS, 1 - EPS)
    p2 = np.clip(r["p2rank"].astype(np.float64), EPS, 1 - EPS)
    ps, pr, pm, inp = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for k, (score, res) in enumerate(r["pockets"], 1):            # P2Rank order: rank 1 first
        res = [i for i in res if i < n]
        if res:
            m = ours[res].mean()
            for i in res:
                if score > ps[i]:
                    ps[i], pr[i], pm[i], inp[i] = score, 1.0 / k, m, 1.0
    lo = lambda x: np.log(x / (1 - x))
    return np.stack([lo(ours), lo(p2), ours * p2, np.log1p(ps), pr, pm, inp], axis=1)


def learned(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    recs = load_records(args)
    fam = pd.read_csv(args.candidates).set_index("accession")["family"].to_dict()
    V = [r for r in recs if r["group"] == "V" and r["truth"].get("site")]
    fams = sorted({fam[r["accession"]] for r in V})
    half = set(np.random.default_rng(42).permutation(fams)[: len(fams) // 2])
    fold = {r["accession"]: int(fam[r["accession"]] in half) for r in V}

    def fit(rs):
        X = np.concatenate([features(r) for r in rs])
        y = np.concatenate([np.isin(np.arange(r["length"]), r["truth"]["site"]).astype(int) for r in rs])
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")).fit(X, y)

    score = {}
    for k in (0, 1):
        clf = fit([r for r in V if fold[r["accession"]] != k])
        for r in V:
            if fold[r["accession"]] == k:
                score[(r["group"], r["accession"])] = clf.predict_proba(features(r))[:, 1]
    full = fit(V)
    coef = dict(zip(FEATURES, np.round(full[-1].coef_[0], 3)))
    for r in recs:
        if r["group"] != "V":
            score[(r["group"], r["accession"])] = full.predict_proba(features(r))[:, 1]
    preds, res = os.path.join(args.out_dir, "preds"), {}
    for group, t in TASKS:
        sub = [r for r in recs if r["group"] == group and r["truth"].get(t)]
        f_l = os.path.join(preds, f"{group}_{t}_learned.pt")
        torch.save(as_predictions(sub, t, lambda r: score[(group, r["accession"])]), f_l)
        res[f"{group}/{t}"] = {a: paired(os.path.join(preds, f"{group}_{t}_{a}.pt"), a, f_l, "learned",
                                         os.path.join(preds, f"{group}_{t}_{a}_vs_learned.json"), args.n_bootstrap,
                                         ["--drop-label-mismatch"])
                               for a in ("p2rank", "ours_any", "noisy_or")}
        m = res[f"{group}/{t}"]["p2rank"]["macro_auprc"]
        print(f"[{group}/{t}] learned {m['b']:.3f} vs P2Rank {m['a']:.3f}", flush=True)
    with open(os.path.join(args.out_dir, "learned_results.json"), "w") as f:
        json.dump({"coefficients": {k: float(v) for k, v in coef.items()}, "results": res}, f, indent=1)


def protein(args):
    from sklearn.metrics import roc_auc_score
    recs = load_records(args)
    rows = [{"group": r["group"], "accession": r["accession"], "ours_any": float(r["ours_any"].max()),
             "ours_final": float(r["ours_final"].max()), "p2rank_pocket": max([pk[0] for pk in r["pockets"]], default=0.0),
             "p2rank_max": float(r["p2rank"].max())} for r in recs]
    load = lambda d: torch.load(os.path.join(args.scores_dir, d, "raw_negatives.pt"), weights_only=False)
    ga, gf = load(f"{args.any}_gold"), load(f"{args.final}_gold")
    missing = 0
    for acc in ga:
        p2 = p2rank_of(acc, args.p2rank_dir)
        if p2 is None or acc not in gf:
            missing += 1
            continue
        rows.append({"group": "G", "accession": acc, "ours_any": float(torch.sigmoid(ga[acc]["logits"]).max()),
                     "ours_final": float(torch.sigmoid(gf[acc]["logits"]).max()),
                     "p2rank_pocket": max([pk[0] for pk in p2["pockets"]], default=0.0),
                     "p2rank_max": float(p2["prob"].max())})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "protein_scores.csv"), index=False)
    rng = np.random.default_rng(42)
    res = {"G_without_p2rank": missing, "n": df.group.value_counts().to_dict()}
    for pos, neg in (("goliath", "G"), ("mcsa", "G"), ("V", "G")):
        a, b = df[df.group == pos], df[df.group == neg]
        y = np.r_[np.ones(len(a)), np.zeros(len(b))]
        res[f"{pos}_vs_{neg}"] = {}
        for s in ("ours_any", "ours_final", "p2rank_pocket", "p2rank_max"):
            boots = []
            for _ in range(1000):
                ia, ib = rng.integers(0, len(a), len(a)), rng.integers(0, len(b), len(b))
                boots.append(roc_auc_score(y, np.r_[a[s].values[ia], b[s].values[ib]]))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            res[f"{pos}_vs_{neg}"][s] = {"auroc": float(roc_auc_score(y, np.r_[a[s].values, b[s].values])),
                                         "ci95": [float(lo), float(hi)]}
        print(f"  {pos} vs G: " + ", ".join(f"{s} {v['auroc']:.3f}" for s, v in res[f"{pos}_vs_{neg}"].items()))
    res["share_with_pocket"] = {g: float((df[df.group == g].p2rank_pocket > 0).mean()) for g in df.group.unique()}
    with open(os.path.join(args.out_dir, "protein_results.json"), "w") as f:
        json.dump(res, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["assemble", "evaluate", "learned", "protein", "all"])
    ap.add_argument("--p2rank-dir", default="data/comparisons/p2rank/out")
    ap.add_argument("--scores-dir", default="results/negatives")
    ap.add_argument("--any", default="sitetypes", help="prefix of the site-type model's scoring runs")
    ap.add_argument("--final", default="main", help="prefix of the main model's scoring runs")
    ap.add_argument("--candidates", default="dataset/negatives/pocket_candidates.csv")
    ap.add_argument("--out-dir", default="results/reports/p2rank")
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    steps = {"assemble": assemble, "evaluate": evaluate, "learned": learned, "protein": protein}
    for name in (list(steps) if args.step == "all" else [args.step]):
        steps[name](args)


if __name__ == "__main__":
    main()
