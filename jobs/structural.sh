#!/bin/bash
#SBATCH --job-name=siteseer-structural
#SBATCH --output=%x-%j.out
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
# Structural ablation (figure 3.15), on ESMC-6B and fix1 data:
#   structures (AlphaFold DB) -> self-supervised GVP pretraining -> structural probe stage 1, 1.5, M-CSA and 5-fold
#   -> the same chain with the Flat probe (the controlled comparison) -> paired bootstraps (Structural minus Flat,
#   and Structural minus the main model).
#   sbatch jobs/structural.sh        (environment: siteseer-esmc; needs the ESMC cache of jobs/chain.sh MODEL=esmc_6b
#                                    and, for the second comparison, jobs/chain.sh MODEL=main). Each step resumes;
#                                    pretraining alone takes ~2 days on one A30, so resubmit until it reports done.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
PRED=results/predictions
REP=results/reports/structural
CK=checkpoints
KF=dataset/mcsa_fine_tune/kfold
L=dataset/label_diffusion
M=dataset/mcsa_fine_tune/normal_mcsa
GVP=$CK/gvp_ssl/final.pt
mkdir -p "$PRED" "$REP"

echo "=== structures and pretraining ==="
python -u -m siteseer.structural.download --out-dir data/structures --csv "$L/cached_dataframe_fully_excl_fix1.csv" \
    "$L/cached_dataframe_real_labels_only_fix1.csv" "$M"/{train,valid,test}_dataset/mcsa_{train,valid,test}.csv
if [ ! -f "$GVP" ]; then
    LAST=$(ls -t "$CK"/gvp_ssl/step_*.pt 2>/dev/null | head -1 || true)
    python -u -m siteseer.structural.pretrain --config configs/structural/pretrain_gvp.yaml \
        ${LAST:+--resume-from "$LAST"}
fi

train () {   # run  [train_probe flags...]
    local RUN=$1; shift
    if [ -f "$CK/$RUN/best.pt" ]; then echo "=== $RUN: done ==="; return; fi
    local RESUME=()
    [ -f "$CK/$RUN/latest.pt" ] && RESUME=(--resume-from "$CK/$RUN/latest.pt")
    python -u -m siteseer.train.train_probe --set "run_name=$RUN" "$@" "${RESUME[@]}"
}
evaluate () {   # run  out  [evaluate flags...]
    local RUN=$1 OUT=$2; shift 2
    if [ -f "$OUT" ]; then echo "[*] $OUT exists"; return; fi
    python -u -m siteseer.eval.evaluate --probe-checkpoint "$CK/$RUN/best.pt" --split test --calibrate-threshold \
        --save-predictions "$OUT" "$@"
}
fold () { echo "data.train_csv=$KF/fold_$1_train.csv" "data.valid_csv=$KF/fold_$1_valid.csv" "data.test_csv=$KF/fold_$1_test.csv"; }

echo "=== structural probe ==="
S=(--probe-architecture graph_v2 --pretrained-gvp-checkpoint "$GVP")
C=configs/structural
train structural_stage1 --config $C/stage1.yaml "${S[@]}" --checkpoint-every-n-steps 2000
train structural_stage15 --config $C/stage15.yaml "${S[@]}" --checkpoint-every-n-steps 2000 \
    --init-probe-checkpoint "$CK/structural_stage1/best.pt"
train structural_stage2_mcsa --config $C/stage2_mcsa.yaml "${S[@]}" --checkpoint-every-n-steps 500 \
    --init-probe-checkpoint "$CK/structural_stage15/best.pt"
for X in stage1 stage15 stage2_mcsa; do
    evaluate "structural_$X" "$PRED/structural_${X}_test.pt" --config "$C/$X.yaml" --probe-architecture graph_v2
done
for i in 0 1 2 3 4; do
    train "structural_kfold$i" --config $C/kfold.yaml --set $(fold $i) "${S[@]}" --checkpoint-every-n-steps 500 \
        --init-probe-checkpoint "$CK/structural_stage15/best.pt"
    evaluate "structural_kfold$i" "$PRED/structural_kfold${i}_test.pt" --config $C/kfold.yaml --set $(fold $i) \
        --probe-architecture graph_v2
done

echo "=== Flat probe on the same features and data (ESMC-6B, fix1) ==="
E=(configs/plm/esmc_6b.yaml)
train esmc_fix1_stage1 --config configs/main/stage1.yaml "${E[@]}" --set "data.train_csv=$L/cached_dataframe_fully_excl_fix1.csv" \
    --checkpoint-every-n-steps 2000
train esmc_fix1_stage15 --config configs/main/stage15.yaml "${E[@]}" \
    --set "data.train_csv=$L/cached_dataframe_real_labels_only_fix1.csv" --checkpoint-every-n-steps 2000 \
    --init-probe-checkpoint "$CK/esmc_fix1_stage1/best.pt"
evaluate esmc_fix1_stage1 "$PRED/esmc_fix1_stage1_test.pt" --config configs/main/stage1.yaml "${E[@]}" \
    --set "data.train_csv=$L/cached_dataframe_fully_excl_fix1.csv"
evaluate esmc_fix1_stage15 "$PRED/esmc_fix1_stage15_test.pt" --config configs/main/stage15.yaml "${E[@]}" \
    --set "data.train_csv=$L/cached_dataframe_real_labels_only_fix1.csv"
for i in 0 1 2 3 4; do
    train "esmc_fix1_kfold$i" --config configs/main/kfold.yaml "${E[@]}" --set $(fold $i) --checkpoint-every-n-steps 500 \
        --init-probe-checkpoint "$CK/esmc_fix1_stage15/best.pt"
    evaluate "esmc_fix1_kfold$i" "$PRED/esmc_fix1_kfold${i}_test.pt" --config configs/main/kfold.yaml "${E[@]}" --set $(fold $i)
done

echo "=== paired comparisons (pooled AUPRC in figure 3.15) ==="
for A in esmc_fix1 main; do
    for X in stage1 stage15; do
        python -u -m siteseer.eval.paired_bootstrap --a "$PRED/${A}_${X}_test.pt" --label-a "$A" \
            --b "$PRED/structural_${X}_test.pt" --label-b structural --drop-label-mismatch --n-bootstrap 1000 \
            --out-json "$REP/${A}_vs_structural_$X.json"
    done
    python -u -m siteseer.eval.paired_bootstrap --a "$PRED/${A}"_kfold{0,1,2,3,4}_test.pt --label-a "$A" \
        --b "$PRED"/structural_kfold{0,1,2,3,4}_test.pt --label-b structural --drop-label-mismatch --n-bootstrap 1000 \
        --out-json "$REP/${A}_vs_structural_kfold.json"
done
echo "=== done: $PRED/{structural,esmc_fix1}_*.pt, $REP/, $CK/{gvp_ssl,structural_*,esmc_fix1_*} ==="
