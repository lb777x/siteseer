#!/bin/bash
#SBATCH --job-name=siteseer-seeds
#SBATCH --output=%x-%j.out
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
# Seed spread of the main model: the M-CSA stage (single split and 5-fold) retrained with seeds 43-46 from the same
# stage-1.5 checkpoint (seed 42 is the main run of jobs/chain.sh). Then mean, SD and range over the five seeds, the
# five-seed ensemble, and each seed against seed 42 (paired bootstrap, 5-fold).
#   sbatch jobs/seeds.sh        (environment: siteseer; needs MODEL=main of jobs/chain.sh)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
PRED=results/predictions
REP=results/reports/seeds
CK=checkpoints
KF=dataset/mcsa_fine_tune/kfold
EASIFA82=dataset/easifa/easifa82_protein_ids.txt
INIT=$CK/main_stage15/best.pt
mkdir -p "$PRED" "$REP"
for f in "$INIT" "$PRED"/main_kfold{0,1,2,3,4}_{test,valid}.pt "$PRED/main_stage2_mcsa_test.pt"; do
    [ -f "$f" ] || { echo "missing $f: run jobs/chain.sh (MODEL=main) first"; exit 1; }
done

train () {   # run  config  [--set ...]
    local RUN=$1 C=$2; shift 2
    if [ -f "$CK/$RUN/best.pt" ]; then echo "=== $RUN: done ==="; return; fi
    echo "=== $RUN ==="
    python -u -m siteseer.train.train_probe --config "$C" --set "run_name=$RUN" "$@" \
        --init-probe-checkpoint "$INIT" --checkpoint-every-n-steps 500
}
evaluate () {   # run  config  split  out  [eval flags...] -- set by SETS
    local RUN=$1 C=$2 SPLIT=$3 OUT=$4; shift 4
    if [ -f "$OUT" ]; then echo "[*] $OUT exists"; return; fi
    python -u -m siteseer.eval.evaluate --config "$C" --set "run_name=$RUN" "${SETS[@]}" \
        --probe-checkpoint "$CK/$RUN/best.pt" --split "$SPLIT" --save-predictions "$OUT" "$@"
}

for SEED in 43 44 45 46; do
    R=main_seed${SEED}_stage2_mcsa
    SETS=("seed=$SEED")
    train "$R" configs/main/stage2_mcsa.yaml "${SETS[@]}"
    evaluate "$R" configs/main/stage2_mcsa.yaml test  "$PRED/main_seed${SEED}_stage2_mcsa_test.pt" --calibrate-threshold
    evaluate "$R" configs/main/stage2_mcsa.yaml valid "$PRED/main_seed${SEED}_stage2_mcsa_valid.pt"
    if [ -f "$EASIFA82" ]; then
        evaluate "$R" configs/main/stage2_mcsa.yaml test "$PRED/main_seed${SEED}_stage2_easifa82_test.pt" \
            --filter-ids-file "$EASIFA82"
    fi
    for i in 0 1 2 3 4; do
        R=main_seed${SEED}_kfold$i
        SETS=("seed=$SEED" "data.train_csv=$KF/fold_${i}_train.csv" "data.valid_csv=$KF/fold_${i}_valid.csv"
              "data.test_csv=$KF/fold_${i}_test.csv")
        train "$R" configs/main/kfold.yaml "${SETS[@]}"
        evaluate "$R" configs/main/kfold.yaml test  "$PRED/${R}_test.pt" --calibrate-threshold
        evaluate "$R" configs/main/kfold.yaml valid "$PRED/${R}_valid.pt"
    done
done

RUNS=(main main_seed43 main_seed44 main_seed45 main_seed46)
SUMMARY_SETS=(kfold_test kfold_valid stage2_mcsa_test)
if [ -f "$EASIFA82" ]; then SUMMARY_SETS+=(stage2_easifa82_test); fi
python -u -m siteseer.eval.seed_summary --runs "${RUNS[@]}" --sets "${SUMMARY_SETS[@]}" --out-json "$REP/seed_summary.json"
python -u -m siteseer.eval.ensemble --out-json "$REP/ensemble_mcsa95.json" \
    --predictions "$PRED/main_stage2_mcsa_test.pt" "$PRED"/main_seed{43,44,45,46}_stage2_mcsa_test.pt
for SEED in 43 44 45 46; do
    for S in test valid; do
        python -u -m siteseer.eval.paired_bootstrap --label-a seed42 --label-b seed$SEED --n-bootstrap 2000 \
            --a "$PRED"/main_kfold{0,1,2,3,4}_$S.pt --b "$PRED"/main_seed${SEED}_kfold{0,1,2,3,4}_$S.pt \
            --out-json "$REP/kfold_${S}_seed42_vs_seed$SEED.json"
    done
done
echo "=== done: $PRED/main_seed*.pt, $REP/, $CK/main_seed* ==="
