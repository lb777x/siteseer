#!/bin/bash
#SBATCH --job-name=siteseer-squidly
#SBATCH --output=%x-%j.out
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
# Squidly's released 3B ensemble (trained on CataloDB) on Squidly's CataloDB test set and on all 962 M-CSA proteins,
# through Squidly's own ensemble script (per-residue probabilities of all five models).
#   SQUIDLY_WEIGHTS=external/squidly sbatch jobs/squidly.sh
# Environment: siteseer-squidly (envs/install_squidly.sh, which also downloads the weights). ESM-2 3B in fp32 needs
# ~11 GB of GPU memory; minutes per set.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export LD_LIBRARY_PATH="${CONDA_PREFIX:+$CONDA_PREFIX/lib:}${LD_LIBRARY_PATH:-}"   # the env's libstdc++ before the system one
MODELS=${SQUIDLY_WEIGHTS:-external/squidly}/models/3B
OUT=data/comparisons/squidly
[ -d "$MODELS" ] || { echo "missing $MODELS: run envs/install_squidly.sh"; exit 1; }
SQ=$(python -c "import importlib.util, os; print(os.path.join(os.path.dirname(importlib.util.find_spec('squidly').origin), 'squidly.py'))")
mkdir -p "$OUT/catalodb_run" "$OUT/mcsa_run"
python -u "$SQ" dataset/catalodb_squidly/test.fasta esm2_t36_3B_UR50D "$MODELS" "$OUT/catalodb_run"
python -u "$SQ" dataset/squidly/mcsa_all.fasta esm2_t36_3B_UR50D "$MODELS" "$OUT/mcsa_run"
echo "=== done: $OUT/catalodb_run/test_squidly_ensemble.pkl, $OUT/mcsa_run/mcsa_all_squidly_ensemble.pkl ==="
