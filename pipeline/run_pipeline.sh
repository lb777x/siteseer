#!/usr/bin/env bash
# Rebuild every training and evaluation split from the raw sources, then verify the result against the checksums of
# the reported runs. Run from the data root (the directory that holds dataset/ and data/), with the siteseer
# environment active and BLAST+/MMseqs2 on PATH (envs/siteseer-tools.yml), after pipeline/00_download.py.
#   bash /path/to/siteseer/pipeline/run_pipeline.sh
set -euo pipefail
P="$(cd "$(dirname "$0")" && pwd)"
L=dataset/label_diffusion
E=dataset/ec_site_dataset/uniprot_ecreact_cluster_split_merge_dataset_limit_100
M=dataset/mcsa_fine_tune/normal_mcsa
D=data/pipeline

# label branch: curated labels for the Goliath pool, extended pool, 90%-identity split
python "$P/01_mcsa_label_transfer.py"
python "$P/02_m3site_labels.py"
python "$P/03_biolip_labels.py"
python "$P/04_expand_pool.py"
python "$P/05_resplit.py"
python "$P/06_apply_split.py"
# split branch: the same pool without EasIFA's test set, minus M-CSA and EasIFA-test cluster members in train
python "$P/07_easifa_test_ids.py"
python "$P/05_resplit.py" --exclude-ids-file $D/easifa_test_protein_ids.txt --resplit-dir $D/resplit_excl_easifa_test \
    --out-csv $E/cached_dataframe_resplit90_excl_easifa_test.csv
python "$P/06_apply_split.py" --source expanded --mapping $D/resplit_excl_easifa_test/goliath_cluster_assignment_90.csv \
    --exclude-ids-file $D/easifa_test_protein_ids.txt --out-csv $L/cached_dataframe_final_resplit90_excl_easifa_test.csv
python "$P/08_cluster_exclusion.py" --source-csv $L/cached_dataframe_final_resplit90_excl_easifa_test.csv \
    --out-csv $L/cached_dataframe_resplit90_excl_easifa_test_excl_mcsa_cluster90.csv --flag excluded_mcsa_cluster90 \
    --protect-csv $M/valid_dataset/mcsa_valid.csv $M/test_dataset/mcsa_test.csv
python "$P/08_cluster_exclusion.py" --source-csv $L/cached_dataframe_resplit90_excl_easifa_test_excl_mcsa_cluster90.csv \
    --out-csv $L/cached_dataframe_resplit90_fully_excl.csv --flag excluded_easifa_test_cluster90 \
    --protect-ids $D/easifa_test_protein_ids.txt
# training data of every reported model: merge, M-CSA exclusion (fix1), label-transfer removal (fix2), M-CSA folds
python "$P/09_fix1_datasets.py"
python "$P/10_fix2_datasets.py"
python "$P/11_mcsa_kfold.py"
# comparison sets: EasIFA's M-CSA and Swiss-Prot test sets with its predictions, EasIFA's and Squidly's training data
python "$P/12_easifa82.py"
python "$P/13_easifa_swissprot.py"
python "$P/14_easifa_own_split.py"
python "$P/15_catalodb_split.py"

sha256sum -c "$P/expected_sha256.txt"
