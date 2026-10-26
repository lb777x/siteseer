#!/bin/bash
#SBATCH --job-name=siteseer-layer-screens
#SBATCH --output=%x-%j.out
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
# Linear screens of hidden layers (logistic regression on standardised per-residue embeddings):
#   ESM-2 650M layers 32, 31, 30, 28, 24, 18 against layer 33 (2,000 train / 400 test proteins, fix1 curated labels);
#   ESMC-6B blocks 80, 76, 72, 64 against the ESMC cache, and ESMC + ESM-2 layer 33 concatenated (4,000 / 1,000,
#   test proteins from valid+test, fix2 curated labels).
#   sbatch jobs/layer_screens.sh        (environment: siteseer-esmc; the ESMC part needs the esmc_6b and esm2_l33 caches
#                                        of jobs/chain.sh). Extraction ~2 h on one GPU, the screens run on CPU.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
REP=results/reports/layer_screens
mkdir -p "$REP"
screen () {   # csv  dir-a  label-a  dir-b  label-b  out  n-train  n-test  test-splits...
    local CSV=$1 A=$2 LA=$3 B=$4 LB=$5 OUT=$6 NTR=$7 NTE=$8; shift 8
    if [ -f "$OUT" ]; then echo "[*] $OUT exists"; return; fi
    python -u -m siteseer.eval.linear_screen --csv "$CSV" --dir-a "$A" --label-a "$LA" --dir-b "$B" --label-b "$LB" \
        --n-train "$NTR" --n-test "$NTE" --test-splits "$@" --out-json "$OUT"
}

echo "=== ESM-2 650M layers ==="
CSV=dataset/label_diffusion/cached_dataframe_real_labels_only_fix1.csv
L=data/esm2_layers
python -u -m siteseer.embeddings.extract_layers --model esm2_650m --layers 18 24 28 30 31 32 33 --csv "$CSV" \
    --out-root "$L" --n-train 2500 --n-test 500
for K in 32 31 30 28 24 18; do
    screen "$CSV" "$L/layer_33" "ESM-2 layer 33" "$L/layer_$K" "ESM-2 layer $K" \
        "$REP/esm2_layer33_vs_layer$K.json" 2000 400 test
done

echo "=== ESMC-6B blocks ==="
CSV=dataset/label_diffusion/cached_dataframe_real_labels_only_fix2.csv
L=data/esmc_layers
python -u -m siteseer.embeddings.extract_layers --model esmc_6b --layers 64 72 76 80 --csv "$CSV" \
    --out-root "$L" --n-train 4000 --n-test 1000 --test-splits valid test
for K in 80 76 72 64; do
    screen "$CSV" data/embeddings_goliath_esmc "ESMC cache" "$L/layer_$K" "ESMC block $K" \
        "$REP/esmc_cache_vs_block$K.json" 4000 1000 valid test
done
screen "$CSV" data/embeddings_goliath_esmc "ESMC-6B" data/embeddings_goliath_esm2_l33 "ESM-2 650M layer 33" \
    "$REP/esmc_vs_esm2_concat.json" 4000 1000 valid test
echo "=== done: $REP/ (data/esm2_layers and data/esmc_layers can be deleted) ==="
