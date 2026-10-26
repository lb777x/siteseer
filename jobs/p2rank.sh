#!/bin/bash
#SBATCH --job-name=siteseer-p2rank
#SBATCH --output=%x-%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
# P2Rank 2.5.1 (alphafold model) on the AlphaFold models of every protein in the comparison, then the comparison
# (residue, pocket and protein level) against the main model and the site-type model.
#   bash jobs/p2rank.sh        (CPU; environment: siteseer with Java on PATH; P2Rank from envs/install_p2rank.sh;
#                               needs jobs/negatives.sh). Skip the first two steps by placing P2Rank's outputs from
#                               the data release in data/comparisons/p2rank/out.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
PRANK=${PRANK:-external/p2rank_2.5.1/prank}
OUT=data/comparisons/p2rank
mkdir -p "$OUT"
if [ ! -d "$OUT/out" ]; then
    python -u -m siteseer.comparisons.alphafold_models --out-dir "$OUT/pdb" --dataset-file "$OUT/all.ds"
    "$PRANK" predict -c alphafold -threads "${SLURM_CPUS_PER_TASK:-8}" -o "$OUT/out" "$OUT/all.ds"
fi
python -u -m siteseer.comparisons.p2rank all --p2rank-dir "$OUT/out" --out-dir results/reports/p2rank
echo "=== done: results/reports/p2rank/ ==="
