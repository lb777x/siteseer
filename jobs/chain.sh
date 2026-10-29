#!/bin/bash
#SBATCH --job-name=siteseer-chain
#SBATCH --output=%x-%j.out
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
# Training chain for one language model: embedding cache, stage 1 (Goliath), stage 1.5 (curated labels), stage 2
# (M-CSA split) and the 5-fold M-CSA cross-validation, each with saved predictions; then the pooled 5-fold metrics.
#   MODEL=main (ESM-2 650M layer 30, default) | esm2_l33 | esm2_3b | esmc_6b
#   sbatch --export=ALL,MODEL=esm2_3b jobs/chain.sh        or        MODEL=esm2_3b bash jobs/chain.sh
# Run from the repository root with the data in place (pipeline/ or the data release) and the environment active:
# siteseer for the ESM-2 models, siteseer-esmc for esmc_6b. Finished steps are skipped, so a resubmitted job resumes.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export LD_LIBRARY_PATH="${CONDA_PREFIX:+$CONDA_PREFIX/lib:}${LD_LIBRARY_PATH:-}"   # the env's libstdc++ before the system one
MODEL=${MODEL:-main}
case "$MODEL" in
    main)     OVERLAY=();                          CACHE=data/embeddings_goliath_esm2_l30 ;;
    esm2_l33) OVERLAY=(configs/plm/esm2_l33.yaml); CACHE=data/embeddings_goliath_esm2_l33 ;;
    esm2_3b)  OVERLAY=(configs/plm/esm2_3b.yaml);  CACHE=data/embeddings_goliath_esm2_3b ;;
    esmc_6b)  OVERLAY=(configs/plm/esmc_6b.yaml);  CACHE=data/embeddings_goliath_esmc ;;
    *) echo "MODEL must be main, esm2_l33, esm2_3b or esmc_6b"; exit 1 ;;
esac
PRED=results/predictions
REP=results/reports
CK=checkpoints
mkdir -p "$PRED" "$REP"
S1_CSV=dataset/label_diffusion/cached_dataframe_fully_excl_fix2.csv
S15_CSV=dataset/label_diffusion/cached_dataframe_real_labels_only_fix2.csv
EASIFA82=dataset/easifa/easifa82_protein_ids.txt
KF=dataset/mcsa_fine_tune/kfold

echo "=== embedding cache $CACHE ==="
case "$MODEL" in
    main)     python -u -m siteseer.embeddings.extract_esm2_cache --layer 30 --out-dir "$CACHE" ;;
    esm2_l33) python -u -m siteseer.embeddings.extract_esm2_cache --layer 33 --out-dir "$CACHE" ;;
    esm2_3b)  python -u -m siteseer.embeddings.extract_esm2_cache --model esm2_t36_3B_UR50D --layer 36 --out-dir "$CACHE" ;;
    esmc_6b)  for C in "$S1_CSV" "$S15_CSV"; do
                  python -u -m siteseer.embeddings.extract_esmc_cache --csv "$C" --out-dir "$CACHE"
              done ;;
esac
python -u -m siteseer.embeddings.check_cache --embedding-dir "$CACHE" --csv "$S1_CSV" "$S15_CSV"
if [ "$MODEL" = main ] && [ ! -f "$CACHE/norm_stats.pt" ]; then
    python -u -m siteseer.embeddings.norm_stats --cache-dir "$CACHE"
fi

train () {   # run  base-config  checkpoint-every  [extra flags...]
    local RUN=$1 C=$2 EVERY=$3; shift 3
    if [ -f "$CK/$RUN/best.pt" ]; then echo "=== $RUN: done ==="; return; fi
    echo "=== $RUN ==="
    local RESUME=()
    [ -f "$CK/$RUN/latest.pt" ] && RESUME=(--resume-from "$CK/$RUN/latest.pt")
    python -u -m siteseer.train.train_probe --config "$C" "${OVERLAY[@]}" --set "run_name=$RUN" "${SETS[@]}" \
        --checkpoint-every-n-steps "$EVERY" "${RESUME[@]}" "$@"
}
evaluate () {   # run  base-config  split  out  [extra flags...]
    local RUN=$1 C=$2 SPLIT=$3 OUT=$4; shift 4
    if [ -f "$OUT" ]; then echo "[*] $OUT exists"; return; fi
    python -u -m siteseer.eval.evaluate --config "$C" "${OVERLAY[@]}" --set "run_name=$RUN" "${SETS[@]}" \
        --probe-checkpoint "$CK/$RUN/best.pt" --split "$SPLIT" --save-predictions "$OUT" "$@"
}

SETS=()
train "${MODEL}_stage1"  configs/main/stage1.yaml      2000
train "${MODEL}_stage15" configs/main/stage15.yaml     2000 --init-probe-checkpoint "$CK/${MODEL}_stage1/best.pt"
train "${MODEL}_stage2_mcsa" configs/main/stage2_mcsa.yaml 500 --init-probe-checkpoint "$CK/${MODEL}_stage15/best.pt"
for S in stage1 stage15 stage2_mcsa; do
    evaluate "${MODEL}_$S" configs/main/$S.yaml test  "$PRED/${MODEL}_${S}_test.pt" --calibrate-threshold
    evaluate "${MODEL}_$S" configs/main/$S.yaml valid "$PRED/${MODEL}_${S}_valid.pt"
done
if [ -f "$EASIFA82" ]; then
    evaluate "${MODEL}_stage2_mcsa" configs/main/stage2_mcsa.yaml test "$PRED/${MODEL}_stage2_easifa82_test.pt" \
        --calibrate-threshold --filter-ids-file "$EASIFA82"
else
    echo "[*] $EASIFA82 missing: EasIFA-82 evaluation skipped"
fi

echo "=== 5-fold M-CSA (every fold starts from stage 1.5) ==="
for i in 0 1 2 3 4; do
    SETS=("data.train_csv=$KF/fold_${i}_train.csv" "data.valid_csv=$KF/fold_${i}_valid.csv"
          "data.test_csv=$KF/fold_${i}_test.csv")
    R="${MODEL}_kfold$i"
    train "$R" configs/main/kfold.yaml 500 --init-probe-checkpoint "$CK/${MODEL}_stage15/best.pt"
    evaluate "$R" configs/main/kfold.yaml test  "$PRED/${R}_test.pt" --calibrate-threshold
    evaluate "$R" configs/main/kfold.yaml valid "$PRED/${R}_valid.pt"
done
python -u -m siteseer.eval.aggregate_kfold --predictions "$PRED/${MODEL}"_kfold{0,1,2,3,4}_test.pt \
    --valid-predictions "$PRED/${MODEL}"_kfold{0,1,2,3,4}_valid.pt --out-json "$REP/${MODEL}_kfold.json"
echo "=== done: $PRED/${MODEL}_*.pt, $REP/${MODEL}_kfold.json, $CK/${MODEL}_* ==="
