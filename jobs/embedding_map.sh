#!/bin/bash
#SBATCH --job-name=siteseer-embedding-map
#SBATCH --output=%x-%j.out
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
# Embedding maps (figures 3.21, 3.22): ESM-2 650M layers 30 and 33 of the stage-1 test proteins, then kNN purity and
# UMAP maps for four residue representations and for whole proteins by EC class.
#   sbatch jobs/embedding_map.sh        (environment: siteseer; needs jobs/chain.sh MODEL=main and jobs/sitetypes.sh)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[ -f data/embedding_map_cache.pt ] || python -u -m siteseer.eval.embedding_map extract
python -u -m siteseer.eval.embedding_map project --out results/reports/embedding_map.json
