#!/usr/bin/env bash
# Squidly 0.1.0 and its released 3B weights (Hugging Face WillRieger/Squidly, pinned revision) plus the ESM-2 3B
# weights it embeds with.
#   bash envs/install_squidly.sh [WEIGHTS_DIR]      (default: ./external/squidly)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:-external/squidly}"
REVISION=f5a4b52a0bf46cf900ddc5207c92b6a62a595b29

conda env list | grep -q "^compass-squidly " || conda env create -f "$HERE/squidly.yml"
conda run -n compass-squidly pip install --no-deps squidly==0.1.0
conda run -n compass-squidly python - "$TARGET" "$REVISION" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
path = snapshot_download("WillRieger/Squidly", revision=sys.argv[2], local_dir=sys.argv[1],
                         allow_patterns=["models/3B/*"])
print("Squidly 3B weights:", sorted(os.listdir(os.path.join(path, "models", "3B"))))
import esm
esm.pretrained.esm2_t36_3B_UR50D()   # downloads the ESM-2 3B weights into the torch hub cache once
PY
echo "Squidly ready (weights in $TARGET/models/3B)"
