#!/bin/bash
#SBATCH --job-name=siteseer-docking
#SBATCH --output=%x-%A_%a.out
#SBATCH --array=0-1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
# Docking application (figures 3.16, 3.17): DiffDock on the 757 docking proteins, three conditions per protein on
# the same node (whole structure; 15 A crop around the main model's predicted site; the same with a
# probability-weighted centre). Array task K of N docks every N-th protein; conditions interleaved in blocks of 20.
#   sbatch jobs/docking.sh          then, when all tasks are done:   bash jobs/docking.sh post
# Environments: SITESEER_ENV (default siteseer) for the set, crops and analysis; DIFFDOCK_ENV (default diffdock,
# envs/install_diffdock.sh) and DIFFDOCK_HOME (default external/DiffDock) for docking. Needs jobs/chain.sh
# MODEL=main and the structures of jobs/structural.sh (siteseer.structural.download). Resumable.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
ROOT=$PWD
K=${SLURM_ARRAY_TASK_ID:-0}
N=${SLURM_ARRAY_TASK_COUNT:-1}
SET=dataset/docking/goliath_test_dock.csv
LIG=dataset/docking/ligand_smiles.json
CROPS=data/docking
OUT=results/docking
RUNS=$OUT/runs
export DIFFDOCK_HOME=${DIFFDOCK_HOME:-$ROOT/external/DiffDock}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SITESEER_ENV:-siteseer}"
mkdir -p "$CROPS" "$RUNS"

if [ "${1:-}" = post ]; then
    python -u -m siteseer.docking.report --out-dir "$RUNS" --tracker-csv "$OUT"/tracker_shard*.csv \
        --conditions _global _predicted_cropped _predicted_cropped_weighted --output-csv "$OUT/comparison.csv"
    python -u -m siteseer.docking.paired --csv "$OUT/comparison.csv" --out-json "$OUT/paired.json"
    python -u -m siteseer.docking.site_accuracy --set "$SET" --raw-dir data/structures --out-dir "$RUNS" \
        --pred-residues "$CROPS/predicted_residues_shard*.csv" --out-csv "$OUT/site_accuracy.csv"
    exit 0
fi

[ -f "$SET" ] || python -u pipeline/20_docking_set.py --structure-root data/structures
COMMON=(--csv "$SET" --structure-root data/structures --probe-checkpoint checkpoints/main_stage2_mcsa/best.pt
        --shard-index "$K" --num-shards "$N")
python -u -m siteseer.docking.crops "${COMMON[@]}" --output-dir "$CROPS/predicted_cropped" \
    --residues-out-csv "$CROPS/predicted_residues_shard$K.csv"
python -u -m siteseer.docking.crops "${COMMON[@]}" --output-dir "$CROPS/predicted_cropped_weighted" --weighted-centroid

conda activate "${DIFFDOCK_ENV:-diffdock}"
TOTAL=$(python -c "import pandas as pd; print(len(pd.read_csv('$SET')))")
D=(--csv "$SET" --out-dir "$RUNS" --tracker-csv "$OUT/tracker_shard$K.csv" --smiles-json "$LIG"
   --shard-index "$K" --num-shards "$N")
for L in $(seq 20 20 "$TOTAL") "$TOTAL"; do
    python -u -m siteseer.docking.run_diffdock "${D[@]}" --limit "$L" --condition global --raw-dir data/structures \
        --complex-suffix _global
    python -u -m siteseer.docking.run_diffdock "${D[@]}" --limit "$L" --condition cropped \
        --structure-dir "$CROPS/predicted_cropped" --complex-suffix _predicted_cropped
    python -u -m siteseer.docking.run_diffdock "${D[@]}" --limit "$L" --condition cropped \
        --structure-dir "$CROPS/predicted_cropped_weighted" --complex-suffix _predicted_cropped_weighted
done
echo "=== task $K/$N done; when all tasks are done: bash jobs/docking.sh post ==="
