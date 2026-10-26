"""Step 13: EasIFA's Swiss-Prot E-RXN ASA test set as an evaluation set, EasIFA's predictions on it, and protein
subsets for the comparison.

Test proteins: EasIFA's 851 test proteins that occur with an identical sequence in our stage-1 pool (825; all flagged
excluded_easifa_test there) and have <= 1022 residues; one row per protein (its first protein-reaction pair). Labels
are EasIFA's binary site labels (binding, catalytic and other sites merged), range convention, on both sides. Valid
= stage-1 validation proteins (threshold calibration).

Subsets (dataset/easifa/swissprot_ids_<name>.txt):
  all           825 proteins
  lt80          < 80% identity to everything our models trained on (stage 1, stage 1.5, M-CSA train+valid;
                mmseqs easy-search -s 7.5, best hit with query coverage >= 0.5)
  noleak        not in EasIFA's own train/valid split (EasIFA splits protein-reaction pairs, so 107 test proteins
                also occur in its training data)
  lt80_noleak   both: clean for both training sets (648)
  band_<b>      EasIFA's similarity_index_level (identity to EasIFA's training set), also within noleak

  python pipeline/13_easifa_swissprot.py
"""
import argparse
import json
import os
import tempfile

import pandas as pd
import torch

from common import EXTERNAL, GOLIATH_CSV, LABELS, MCSA_CSV, PIPELINE, run, tool
from siteseer.comparisons.easifa import ID, first_rows, range_labels, to_predictions

RESULTS = f"{EXTERNAL}/easifa_results/results"
S1 = f"{LABELS}/cached_dataframe_fully_excl_fix1.csv"
S15 = f"{LABELS}/cached_dataframe_real_labels_only_fix1.csv"
MODELS = {"esm_bin": "active-site-position-prediction_results.csv",
          "saprot_bin": "active-site-position-prediction_saprot_results.csv",
          "abl1_no_reaction": "ablation-experiment-1_results.csv",
          "abl2_reaction_scratch": "ablation-experiment-2_results.csv",
          "ng_bin": "ablation-experiment-3_results.csv",
          "abl4_rxnfp": "ablation-experiment-4_results.csv"}


def write_ids(path, ids):
    with open(path, "w") as f:
        f.write("\n".join(ids) + "\n")


def identity_to_training(query: dict, threads: int) -> pd.Series:
    """Highest identity of each query protein to our training proteins (0 without a hit)."""
    train = pd.concat([pd.read_csv(S1, usecols=[ID, "aa_sequence", "dataset_flag"], low_memory=False).query("dataset_flag == 'train'"),
                       pd.read_csv(S15, usecols=[ID, "aa_sequence", "dataset_flag"], low_memory=False).query("dataset_flag == 'train'")]
                      + [pd.read_csv(MCSA_CSV[s], usecols=[ID, "aa_sequence"]) for s in ("train", "valid")])
    train[ID] = train[ID].astype(str).str.strip()
    train = train.drop_duplicates(ID)
    with tempfile.TemporaryDirectory() as w:
        for fn, items in (("q", query.items()), ("t", zip(train[ID], train["aa_sequence"]))):
            with open(f"{w}/{fn}.fasta", "w") as f:
                for a, s in items:
                    f.write(f">{a}\n{s}\n")
        run([tool("mmseqs"), "easy-search", f"{w}/q.fasta", f"{w}/t.fasta", f"{w}/h.m8", f"{w}/tmp", "-s", "7.5",
             "--format-output", "query,target,fident,qcov,tcov,evalue", "--threads", threads])
        h = pd.read_csv(f"{w}/h.m8", sep="\t", names=["q", "t", "fident", "qcov", "tcov", "evalue"])
    h = h[h.qcov >= 0.5].sort_values("fident", ascending=False).drop_duplicates("q")
    return pd.Series(dict(zip(h.q, h.fident))).reindex(list(query)).fillna(0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=RESULTS)
    ap.add_argument("--eval-csv", default=f"{LABELS}/easifa_swissprot_eval_fix1.csv")
    ap.add_argument("--ids-dir", default="dataset/easifa")
    ap.add_argument("--pred-dir", default="data/comparisons/easifa")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    for d in (args.ids_dir, args.pred_dir, PIPELINE):
        os.makedirs(d, exist_ok=True)

    ref = first_rows(os.path.join(args.results_dir, MODELS["esm_bin"]))
    pool = pd.read_csv(S1, low_memory=False).reset_index()            # index = row key of the embedding caches
    pool[ID] = pool[ID].astype(str).str.strip()
    row = {}
    for i, a, s, flag in zip(pool["index"], pool[ID], pool["aa_sequence"], pool["dataset_flag"]):
        if a in ref.index and a not in row and s == ref.loc[a, "aa_sequence"]:
            row[a] = (i, flag)
    keep = [a for a in ref.index if a in row and len(ref.loc[a, "aa_sequence"]) <= 1022]
    leaked = [a for a in keep if row[a][1] in ("train", "valid")]
    if leaked:
        raise SystemExit(f"{len(leaked)} EasIFA test proteins are in our train/valid split, e.g. {leaked[:5]}")

    r = ref.loc[keep]
    test = pd.DataFrame({ID: keep, "aa_sequence": r["aa_sequence"].values, "site_labels": r["site_labels"].values,
                         "site_types": r["site_types"].values, "reaction": r["reaction"].values,
                         "label_source": "goliath_original", "dataset_flag": "test",
                         "_original_row_index": [row[a][0] for a in keep]})
    valid = pool[pool.dataset_flag == "valid"].drop_duplicates(ID)
    valid = valid.assign(_original_row_index=valid["index"])[list(test.columns)]
    pd.concat([test, valid]).to_csv(args.eval_csv, index=False)

    n_preds = {}
    for name, f in MODELS.items():
        rows = first_rows(os.path.join(args.results_dir, f))
        rows = rows.loc[[a for a in keep if a in rows.index and rows.loc[a, "aa_sequence"] == ref.loc[a, "aa_sequence"]]]
        preds = to_predictions(rows, labels_from=ref, label_fn=range_labels)
        torch.save(preds, os.path.join(args.pred_dir, f"swissprot_{name}_preds.pt"))
        n_preds[name] = len(preds["protein_ids"])

    ident = identity_to_training(dict(zip(keep, test["aa_sequence"])), args.threads)
    ident.rename("max_identity").to_csv(f"{PIPELINE}/easifa_swissprot_identity_to_training.csv", index_label=ID)
    easifa_trainvalid = set(pd.read_csv(GOLIATH_CSV, usecols=[ID, "dataset_flag"], low_memory=False)
                            .query("dataset_flag in ['train', 'valid']")[ID].astype(str).str.strip())
    lt80 = {a for a in keep if ident[a] < 0.8}
    noleak = set(keep) - easifa_trainvalid
    subsets = {"all": set(keep), "lt80": lt80, "noleak": noleak, "lt80_noleak": lt80 & noleak}
    for band in sorted(r["similarity_index_level"].unique()):
        ids = set(r.index[r["similarity_index_level"] == band])
        tag = band.replace("~", "-").replace("%", "")
        subsets[f"band_{tag}"] = ids
        subsets[f"noleak_band_{tag}"] = ids & noleak
    for name, ids in subsets.items():
        write_ids(os.path.join(args.ids_dir, f"swissprot_ids_{name}.txt"), [a for a in keep if a in ids])
    rep = {"easifa_proteins": len(ref), "compared": len(keep), "eval_csv": {"test": len(test), "valid": len(valid)},
           "easifa_predictions": n_preds, "subsets": {k: len(v) for k, v in subsets.items()}}
    with open(f"{PIPELINE}/easifa_swissprot_report.json", "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
