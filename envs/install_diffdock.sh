#!/usr/bin/env bash
# DiffDock for the docking application: conda environment, upstream code at the pinned commit, and one local fix.
#   bash envs/install_diffdock.sh [TARGET_DIR]     (default: ./external/DiffDock)
# The fix (diffdock_parse_chi.patch): side-chain torsion indices can be NaN for incomplete residues; they are set to
# 0 before the integer cast, which otherwise crashes those proteins. The reported runs used this patched copy.
# Model weights are downloaded by DiffDock itself on the first run. Afterwards: export DIFFDOCK_HOME=<TARGET_DIR>.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:-external/DiffDock}"
COMMIT=85c49b6

conda env list | grep -q "^siteseer-diffdock " || conda env create -f "$HERE/diffdock.yml"
if [ ! -d "$TARGET/.git" ]; then
    git clone https://github.com/gcorso/DiffDock.git "$TARGET"
fi
git -C "$TARGET" checkout -q "$COMMIT"
if git -C "$TARGET" apply --check "$HERE/diffdock_parse_chi.patch" 2>/dev/null; then
    git -C "$TARGET" apply "$HERE/diffdock_parse_chi.patch"
fi
echo "DiffDock ready in $TARGET (commit $COMMIT, patched). export DIFFDOCK_HOME=$(cd "$TARGET" && pwd)"
