#!/bin/bash
#SBATCH --job-name=siteseer-compare-plms
#SBATCH --output=%x-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
# Each language model against the main model on the 5-fold M-CSA predictions (962 proteins), paired bootstrap with
# 2,000 draws over proteins, on the test folds and on the validation folds (the layer was chosen on validation).
#   bash jobs/compare_plms.sh        (CPU; needs jobs/chain.sh for MODEL=main and each variant)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export LD_LIBRARY_PATH="${CONDA_PREFIX:+$CONDA_PREFIX/lib:}${LD_LIBRARY_PATH:-}"   # the env's libstdc++ before the system one
PRED=results/predictions
REP=results/reports
mkdir -p "$REP"
for V in esm2_l33 esm2_3b esmc_6b; do
    for S in test valid; do
        if ! ls "$PRED/${V}"_kfold{0,1,2,3,4}_$S.pt >/dev/null 2>&1; then echo "[*] $V $S: predictions missing"; continue; fi
        python -u -m siteseer.eval.paired_bootstrap --label-a main --label-b "$V" --n-bootstrap 2000 \
            --a "$PRED"/main_kfold{0,1,2,3,4}_$S.pt --b "$PRED/${V}"_kfold{0,1,2,3,4}_$S.pt \
            --out-json "$REP/plm_kfold_${S}_main_vs_$V.json"
    done
done
