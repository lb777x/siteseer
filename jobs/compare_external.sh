#!/bin/bash
#SBATCH --job-name=siteseer-compare-external
#SBATCH --output=%x-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
# Comparisons with EasIFA and Squidly from saved predictions (paired bootstrap, 2,000 draws over proteins):
#   EasIFA ESM-bin / SaProt-bin on EasIFA's 82 M-CSA test proteins (main model, stage 2)
#   EasIFA on its Swiss-Prot benchmark (same-data model and main model; all subsets), and EasIFA's own ablation
#   Squidly 3B (threshold 0.6, its recommendation; ours 0.5) on CataloDB test (same-data model) and on M-CSA
#   (main model, 5-fold out-of-fold; all proteins and those < 30% / < 50% identical to Squidly's training data)
#   bash jobs/compare_external.sh        (CPU; needs jobs/chain.sh, jobs/same_data.sh, jobs/squidly.sh)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export LD_LIBRARY_PATH="${CONDA_PREFIX:+$CONDA_PREFIX/lib:}${LD_LIBRARY_PATH:-}"   # the env's libstdc++ before the system one
PRED=results/predictions
REP=results/reports
EA=data/comparisons/easifa
SQ=data/comparisons/squidly
mkdir -p "$REP"
paired () { python -u -m siteseer.eval.paired_bootstrap --n-bootstrap 2000 "$@"; }

echo "=== EasIFA, 82 M-CSA test proteins ==="
for V in esm_bin saprot_bin; do
    paired --a "$EA/easifa82_${V}_preds.pt" --label-a "easifa_$V" --b "$PRED/main_stage2_easifa82_test.pt" \
        --label-b main --out-json "$REP/easifa82_${V}_vs_main.json"
done

echo "=== EasIFA, Swiss-Prot benchmark ==="
python -u -m siteseer.comparisons.easifa_swissprot --ours "$PRED/easifa_own_swissprot_test.pt" \
    "$PRED"/main_{stage1,stage15,stage2_mcsa}_swissprot_test.pt --out-dir "$REP/easifa_swissprot"
python -u -m siteseer.comparisons.easifa_ablation --results-dir data/external/easifa_results/results \
    --out-json "$REP/easifa_ablation.json"

echo "=== Squidly ==="
python -u -m siteseer.comparisons.squidly --pkl "$SQ/catalodb_run/test_squidly_ensemble.pkl" \
    --labels-csv dataset/catalodb_squidly/catalodb_squidly.csv --out "$SQ/catalodb_preds.pt"
python -u -m siteseer.comparisons.squidly --pkl "$SQ/mcsa_run/mcsa_all_squidly_ensemble.pkl" --out "$SQ/mcsa_preds.pt"
paired --a "$SQ/catalodb_preds.pt" --label-a squidly_3B --threshold-a 0.6 --b "$PRED/catalodb_test.pt" \
    --label-b catalodb --out-json "$REP/squidly_catalodb_vs_ours.json"
for S in all lt30 lt50; do
    RESTRICT=()
    if [ "$S" != all ]; then RESTRICT=(--restrict-ids "dataset/squidly/mcsa_ids_$S.txt"); fi
    paired --a "$PRED"/main_kfold{0,1,2,3,4}_test.pt --label-a main_kfold --b "$SQ/mcsa_preds.pt" --label-b squidly_3B \
        --threshold-b 0.6 "${RESTRICT[@]}" --out-json "$REP/squidly_mcsa_$S.json"
done
echo "=== done: $REP/ ==="
