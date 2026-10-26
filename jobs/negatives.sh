#!/bin/bash
#SBATCH --job-name=siteseer-negatives
#SBATCH --output=%x-%j.out
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
# Negative control and the inputs of the P2Rank comparison: the main model (M-CSA stage) and the site-type model
# (stage 1) score the non-enzyme sets, the pocket candidates and the enzyme test sets (raw residue logits, no
# fallback); then the negative-control numbers (figures 3.18-3.20).
#   sbatch jobs/negatives.sh        (environment: siteseer; needs jobs/chain.sh MODEL=main, jobs/sitetypes.sh and
#                                    pipeline steps 17-19)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
N=dataset/negatives
OUT=results/negatives
CK=checkpoints
mkdir -p "$OUT" results/reports
POS=(--positives "goliath_test=dataset/label_diffusion/cached_dataframe_fully_excl_fix2.csv@test"
     --positives "reallabels_test=dataset/label_diffusion/cached_dataframe_real_labels_only_fix2.csv@test")
score () {   # checkpoint  out  [infer args...]
    local CKPT=$1 O=$2; shift 2
    if [ -f "$O/run_info.json" ]; then echo "[*] $O exists"; return; fi
    python -u -m siteseer.negatives.infer --checkpoint "$CKPT" --embedder esm2_650m_l30 --out-dir "$O" "$@"
}
for M in "main $CK/main_stage2_mcsa/best.pt" "sitetypes $CK/sitetypes_stage1/best.pt"; do
    set -- $M
    [ "$1" = main ] && score "$2" "$OUT/${1}_abc" --negatives "$N/negatives_abc.csv"
    score "$2" "$OUT/${1}_gold" --negatives "$N/negatives_gold.csv"
    score "$2" "$OUT/${1}_pocket" --negatives "$N/pocket_candidates.csv"
    score "$2" "$OUT/${1}_pos" "${POS[@]}"
done
python -u -m siteseer.negatives.analyze --abc "$OUT/main_abc" --gold "$OUT/main_gold" --pos "$OUT/main_pos" \
    --gold-csv "$N/negatives_gold.csv" --out-json results/reports/negative_control.json
echo "=== done: $OUT/, results/reports/negative_control.json ==="
