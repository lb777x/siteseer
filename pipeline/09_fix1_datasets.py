"""Step 09: stage-1 and stage-1.5 training CSVs.

Labels from step 06, split from step 08 (row-aligned). Duplicate ids with conflicting labels take the curated label.
Every protein of the M-CSA pool (train+valid+test) and every >= 90% homolog (`mmseqs easy-search --min-seq-id 0.9
-c 0.8 --cov-mode 0`) is flagged 'excluded_mcsa_homolog90' in all splits. Stage 1.5 = rows with curated labels.

  python pipeline/09_fix1_datasets.py
"""
import argparse
import json
import os
import subprocess

import pandas as pd

from common import LABELS, MCSA_CSV, PIPELINE, tool

ID = "alphafolddb-id"
SPLITS = ("train", "valid", "test")
EXCLUDED_FLAG = "excluded_mcsa_homolog90"
WEAK_SOURCE = "goliath_original"
LABEL_COLUMNS = ["site_labels", "site_types", "label_source", "label_confidence"]

LABELS_CSV = f"{LABELS}/cached_dataframe_final_resplit90.csv"
SPLIT_CSV = f"{LABELS}/cached_dataframe_resplit90_fully_excl.csv"
MCSA_CSVS = [MCSA_CSV[s] for s in SPLITS]
OUT_STAGE1 = f"{LABELS}/cached_dataframe_fully_excl_fix1.csv"
OUT_STAGE15 = f"{LABELS}/cached_dataframe_real_labels_only_fix1.csv"
OUT_REPORT = f"{PIPELINE}/fix1_datasets_report.json"


def unique_ids(df, flag):
    return set(df.loc[df["dataset_flag"] == flag, ID])


def split_counts(df):
    return {flag: len(unique_ids(df, flag)) for flag in sorted(df["dataset_flag"].dropna().unique())}


def harmonise_duplicate_labels(df):
    """Rows of an id with a real-labelled row take that row's label columns."""
    src = df["label_source"].fillna(WEAK_SOURCE)
    n_labels_per_id = df.groupby(ID)["site_labels"].nunique()
    conflicting = set(n_labels_per_id.index[n_labels_per_id > 1])
    real_rows = df[df[ID].isin(conflicting) & (src != WEAK_SOURCE)].drop_duplicates(ID)
    donor = real_rows.set_index(ID)[LABEL_COLUMNS]
    fixable = df[ID].isin(donor.index) & (src == WEAK_SOURCE)
    df.loc[fixable, LABEL_COLUMNS] = donor.loc[df.loc[fixable, ID]].values
    still = df.groupby(ID)["site_labels"].nunique()
    return {"ids_with_conflicting_labels_before": len(conflicting),
            "weak_rows_rewritten": int(fixable.sum()),
            "ids_fixed": int(df.loc[fixable, ID].nunique()),
            "ids_with_conflicting_labels_after (real vs real, first row kept)": int((still > 1).sum())}


def mmseqs_hits(mmseqs, queries, targets, threads, work):
    """query id -> set(target ids) at >=90% identity, >=80% bidirectional coverage."""
    os.makedirs(work, exist_ok=True)
    for name, seqs in (("q.fasta", queries), ("t.fasta", targets)):
        with open(os.path.join(work, name), "w") as f:
            f.writelines(f">{i}\n{s}\n" for i, s in seqs.items())
    out = os.path.join(work, "hits.m8")
    subprocess.run([mmseqs, "easy-search", os.path.join(work, "q.fasta"), os.path.join(work, "t.fasta"),
                    out, os.path.join(work, "tmp"), "--min-seq-id", "0.9", "-c", "0.8", "--cov-mode", "0",
                    "--format-output", "query,target,fident,qcov,tcov", "--threads", str(threads), "-v", "1"],
                   check=True)
    hits = {}
    with open(out) as f:
        for line in f:
            q, t = line.split("\t")[:2]
            hits.setdefault(q, set()).add(t)
    return hits, out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--labels-csv", default=LABELS_CSV)
    p.add_argument("--split-csv", default=SPLIT_CSV)
    p.add_argument("--mcsa-csv", nargs="+", default=MCSA_CSVS)
    p.add_argument("--out-stage1", default=OUT_STAGE1)
    p.add_argument("--out-stage15", default=OUT_STAGE15)
    p.add_argument("--report", default=OUT_REPORT)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--work-dir", default=f"{PIPELINE}/fix1_mmseqs")
    p.add_argument("--easifa-test-ids", default=f"{PIPELINE}/easifa_test_protein_ids.txt")
    p.add_argument("--no-harmonise", action="store_true", help="keep duplicate-id label conflicts as they are")
    args = p.parse_args()
    report = {"inputs": {"labels": args.labels_csv, "split": args.split_csv, "mcsa": args.mcsa_csv}}

    print(f"[*] loading {args.labels_csv} ...", flush=True)
    df = pd.read_csv((args.labels_csv), low_memory=False)
    df[ID] = df[ID].astype(str).str.strip()
    print(f"[*] loading {args.split_csv} (dataset_flag only) ...", flush=True)
    split = pd.read_csv((args.split_csv), usecols=[ID, "dataset_flag"], low_memory=False)
    split[ID] = split[ID].astype(str).str.strip()
    if len(df) != len(split) or not (df[ID].values == split[ID].values).all():
        raise RuntimeError("labels and split CSVs are not row-aligned -- the embedding cache would desync")
    print(f"[+] {len(df)} rows, identical id at every position")
    report["rows"] = int(len(df))
    report["split_before"] = split_counts(split)
    df["dataset_flag"] = split["dataset_flag"].values

    if not args.no_harmonise:
        report["duplicate_label_harmonisation"] = harmonise_duplicate_labels(df)
        print("[+] duplicate-id labels:", report["duplicate_label_harmonisation"])

    # --- M-CSA pool: exact ids and sequences
    mcsa = pd.concat([pd.read_csv((c)) for c in args.mcsa_csv], ignore_index=True)
    mcsa[ID] = mcsa[ID].astype(str).str.strip()
    mcsa = mcsa.drop_duplicates(ID)
    mcsa_ids = set(mcsa[ID])
    queries = dict(zip(mcsa[ID], mcsa["aa_sequence"]))
    report["mcsa_pool_unique_ids"] = len(mcsa_ids)

    pool = df[df["dataset_flag"].isin(SPLITS)].drop_duplicates(ID)
    targets = {i: s for i, s in zip(pool[ID], pool["aa_sequence"]) if isinstance(s, str) and s}
    print(f"[*] mmseqs easy-search: {len(queries)} M-CSA queries vs {len(targets)} stage-1 proteins ...", flush=True)
    hits, hits_path = mmseqs_hits(tool("mmseqs"), queries, targets, args.threads, args.work_dir)
    homolog_ids = set().union(*hits.values()) if hits else set()
    exact_ids = mcsa_ids & set(pool[ID])
    to_exclude = homolog_ids | exact_ids
    mask = df[ID].isin(to_exclude) & df["dataset_flag"].isin(SPLITS)
    per_split_removed = {s: len(unique_ids(df, s) & to_exclude) for s in SPLITS}
    df.loc[mask, "dataset_flag"] = EXCLUDED_FLAG
    report["mcsa_exclusion"] = {
        "method": "mmseqs easy-search --min-seq-id 0.9 -c 0.8 --cov-mode 0 (M-CSA pool vs stage-1 train/valid/test) + exact id",
        "hits_file": hits_path,
        "mcsa_queries_with_hit": len(hits),
        "exact_ids_in_stage1": len(exact_ids),
        "homolog_ids_in_stage1 (incl. exact)": len(homolog_ids),
        "homolog_only_ids (other UniProt id)": len(homolog_ids - mcsa_ids),
        "unique_ids_excluded": len(to_exclude),
        "rows_flagged": int(mask.sum()),
        "removed_per_split": per_split_removed,
        "flag": EXCLUDED_FLAG,
    }
    print("[+] M-CSA exclusion:", {k: v for k, v in report["mcsa_exclusion"].items() if k != "hits_file"})
    report["split_after"] = split_counts(df)

    # --- verification: nothing from the M-CSA pool, nor any of its hits, in any split
    for s in SPLITS:
        left = unique_ids(df, s) & to_exclude
        if left:
            raise RuntimeError(f"{len(left)} excluded ids still in {s}")
    easifa = args.easifa_test_ids
    if os.path.exists(easifa):
        with open(easifa) as f:
            ez = {l.strip() for l in f if l.strip()}
        report["easifa_test_ids_in_splits"] = {s: len(unique_ids(df, s) & ez) for s in SPLITS}
        if any(report["easifa_test_ids_in_splits"].values()):
            raise RuntimeError("EasIFA test proteins in a split -- split CSV is not the fully_excl file?")

    os.makedirs(os.path.dirname((args.out_stage1)), exist_ok=True)
    df.to_csv((args.out_stage1), index=False)
    print(f"[+] wrote {args.out_stage1}")

    # --- stage 1.5: real-labelled rows only, original row index preserved for the embedding cache
    real = df[df["label_source"].fillna(WEAK_SOURCE) != WEAK_SOURCE].copy()
    real["_original_row_index"] = real.index
    real = real.reset_index(drop=True)
    real.to_csv((args.out_stage15), index=False)
    report["stage15"] = {"rows": int(len(real)), "split": split_counts(real)}
    print(f"[+] wrote {args.out_stage15}: {report['stage15']}")

    # --- cross-checks a reader will ask for
    s1 = {s: unique_ids(df, s) for s in SPLITS}
    s15 = {s: unique_ids(real, s) for s in SPLITS}
    report["stage15_test_in_stage1"] = {s: len(s15["test"] & s1[s]) for s in SPLITS}
    report["stage15_train_in_stage1_test"] = len(s15["train"] & s1["test"])
    os.makedirs(os.path.dirname((args.report)), exist_ok=True)
    with open((args.report), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[+] wrote {args.report}")


if __name__ == "__main__":
    main()
