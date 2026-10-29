#!/bin/bash
#SBATCH --job-name=siteseer-sitetypes
#SBATCH --output=%x-%j.out
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
# Site-type model: the main chain (stage 1 -> 1.5 -> M-CSA) with two extra outputs, catalytic and binding, under a
# masked loss (configs/sitetypes.yaml). Main output paired against the main model; type outputs evaluated per type
# and for the catalytic-vs-binding distinction.
#   sbatch jobs/sitetypes.sh        (environment: siteseer; needs jobs/chain.sh MODEL=main)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export LD_LIBRARY_PATH="${CONDA_PREFIX:+$CONDA_PREFIX/lib:}${LD_LIBRARY_PATH:-}"   # the env's libstdc++ before the system one
PRED=results/predictions
REP=results/reports/sitetypes
CK=checkpoints
EASIFA82=dataset/easifa/easifa82_protein_ids.txt
mkdir -p "$PRED" "$REP"
OV=configs/sitetypes.yaml

train () {   # run  base-config  every  [extra flags...]
    local RUN=$1 C=$2 EVERY=$3; shift 3
    if [ -f "$CK/$RUN/best.pt" ]; then echo "=== $RUN: done ==="; return; fi
    local RESUME=()
    [ -f "$CK/$RUN/latest.pt" ] && RESUME=(--resume-from "$CK/$RUN/latest.pt")
    python -u -m siteseer.train.train_probe --config "$C" "$OV" --set "run_name=$RUN" --checkpoint-every-n-steps "$EVERY" \
        "${RESUME[@]}" "$@"
}
evaluate () {   # run  base-config  out  [extra flags...]
    local RUN=$1 C=$2 OUT=$3; shift 3
    if [ -f "$OUT" ]; then echo "[*] $OUT exists"; return; fi
    python -u -m siteseer.eval.evaluate --config "$C" "$OV" --probe-checkpoint "$CK/$RUN/best.pt" --split test \
        --calibrate-threshold --save-predictions "$OUT" "$@"
}
types () {   # run  base-config  baseline-run  name
    if [ -f "$REP/site_type_eval_$4.json" ]; then echo "[*] $4 exists"; return; fi
    python -u -m siteseer.eval.site_type_eval --config "$2" "$OV" --probe-checkpoint "$CK/$1/best.pt" \
        --baseline-checkpoint "$CK/$3/best.pt" --split test --save-predictions "$PRED/sitetypes_types_$4.pt" \
        --out-json "$REP/site_type_eval_$4.json"
}

train sitetypes_stage1 configs/main/stage1.yaml 2000
train sitetypes_stage15 configs/main/stage15.yaml 2000 --init-probe-checkpoint "$CK/sitetypes_stage1/best.pt"
train sitetypes_stage2_mcsa configs/main/stage2_mcsa.yaml 500 --init-probe-checkpoint "$CK/sitetypes_stage15/best.pt"
for S in stage1 stage15 stage2_mcsa; do
    evaluate "sitetypes_$S" "configs/main/$S.yaml" "$PRED/sitetypes_${S}_test.pt"
done
if [ -f "$EASIFA82" ]; then
    evaluate sitetypes_stage2_mcsa configs/main/stage2_mcsa.yaml "$PRED/sitetypes_stage2_easifa82_test.pt" \
        --filter-ids-file "$EASIFA82"
fi
for S in stage1 stage15 stage2_mcsa stage2_easifa82; do
    [ -f "$PRED/sitetypes_${S}_test.pt" ] || continue
    python -u -m siteseer.eval.paired_bootstrap --a "$PRED/main_${S}_test.pt" --label-a main \
        --b "$PRED/sitetypes_${S}_test.pt" --label-b sitetypes --out-json "$REP/paired_main_vs_sitetypes_$S.json"
done
types sitetypes_stage1 configs/main/stage1.yaml main_stage1 stage1_test
types sitetypes_stage15 configs/main/stage15.yaml main_stage15 stage15_test
types sitetypes_stage2_mcsa configs/main/stage1.yaml main_stage2_mcsa final_on_stage1_test
types sitetypes_stage2_mcsa configs/main/stage2_mcsa.yaml main_stage2_mcsa final_mcsa_test
echo "=== done: $PRED/sitetypes_*.pt, $REP/, $CK/sitetypes_* ==="
