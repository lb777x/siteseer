#!/bin/bash
#SBATCH --job-name=siteseer-identity-bands
#SBATCH --output=%x-%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
# Identity bands (plot_identity_bands): the main model's 5-fold macro AUPRC by sequence identity of each M-CSA protein
# to the stage-1 training set and to its own M-CSA training fold. CPU only.
#   sbatch jobs/identity_bands.sh       (environment: siteseer; MMseqs2 from PATH or MMSEQS_BIN; needs jobs/chain.sh
#                                        MODEL=main and pipeline step 21)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
export LD_LIBRARY_PATH="${CONDA_PREFIX:+$CONDA_PREFIX/lib:}${LD_LIBRARY_PATH:-}"   # the env's libstdc++ before the system one
python -u -m siteseer.eval.identity_bands --threads "${SLURM_CPUS_PER_TASK:-8}"
