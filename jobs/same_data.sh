#!/bin/bash
#SBATCH --job-name=siteseer-same-data
#SBATCH --output=%x-%j.out
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
# Same-data comparisons: the main model's recipe trained only on Squidly's CataloDB training data (tested on Squidly's
# test set) and only on EasIFA's training data (tested on EasIFA's Swiss-Prot benchmark); and the main model's own
# checkpoints on that benchmark.
#   sbatch jobs/same_data.sh        (environment: siteseer; needs jobs/chain.sh MODEL=main and pipeline steps 13-15)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export LD_LIBRARY_PATH="${CONDA_PREFIX:+$CONDA_PREFIX/lib:}${LD_LIBRARY_PATH:-}"   # the env's libstdc++ before the system one
PRED=results/predictions
CK=checkpoints
CAT=dataset/catalodb_squidly/catalodb_squidly.csv
OWN=dataset/label_diffusion/easifa_own_split_fix1.csv
SPE=dataset/label_diffusion/easifa_swissprot_eval_fix1.csv
L30=data/embeddings_goliath_esm2_l30
mkdir -p "$PRED"
for f in "$CAT" "$OWN" "$SPE" "$L30/norm_stats.pt" "$CK/main_stage2_mcsa/best.pt"; do
    [ -f "$f" ] || { echo "missing $f"; exit 1; }
done

echo "=== layer-30 embeddings: EasIFA rows missing from the main cache, CataloDB in its own cache ==="
python -u -m siteseer.embeddings.extract_esm2_cache --layer 30 --out-dir "$L30" --need-csv "$OWN" "$SPE"
python -u -m siteseer.embeddings.check_cache --embedding-dir "$L30" --csv "$OWN" "$SPE"
python -u -m siteseer.embeddings.extract_esm2_cache --layer 30 --own-sequences --need-csv "$CAT" \
    --out-dir data/embeddings_catalodb_esm2_l30
python -u -m siteseer.embeddings.check_cache --embedding-dir data/embeddings_catalodb_esm2_l30 --csv "$CAT"

train () {   # run  config  [extra flags...]
    local RUN=$1 C=$2; shift 2
    if [ -f "$CK/$RUN/best.pt" ]; then echo "=== $RUN: done ==="; return; fi
    local RESUME=()
    [ -f "$CK/$RUN/latest.pt" ] && RESUME=(--resume-from "$CK/$RUN/latest.pt")
    python -u -m siteseer.train.train_probe --config "$C" --checkpoint-every-n-steps 2000 "${RESUME[@]}" "$@"
}
evaluate () {   # checkpoint  config  out
    if [ -f "$3" ]; then echo "[*] $3 exists"; return; fi
    python -u -m siteseer.eval.evaluate --config "$2" --probe-checkpoint "$1" --split test --calibrate-threshold \
        --save-predictions "$3"
}

echo "=== CataloDB (Squidly's training data) ==="
train catalodb configs/same_data/catalodb.yaml --early-stop-patience 5
evaluate "$CK/catalodb/best.pt" configs/same_data/catalodb.yaml "$PRED/catalodb_test.pt"

echo "=== EasIFA's training data ==="
train easifa_own configs/same_data/easifa_own.yaml
evaluate "$CK/easifa_own/best.pt" configs/same_data/easifa_own.yaml "$PRED/easifa_own_swissprot_test.pt"

echo "=== main-model checkpoints on EasIFA's Swiss-Prot benchmark ==="
for S in stage1 stage15 stage2_mcsa; do
    evaluate "$CK/main_$S/best.pt" configs/same_data/easifa_swissprot_eval.yaml "$PRED/main_${S}_swissprot_test.pt"
done
echo "=== done: $PRED/{catalodb_test,easifa_own_swissprot_test,main_*_swissprot_test}.pt, $CK/{catalodb,easifa_own} ==="
