#!/usr/bin/env bash
# ESMC-6B environment: envs/siteseer-esmc.yml plus EvolutionaryScale's `esm` SDK (model code and tokenizer for
# biohub/ESMC-6B). esm 3.4.1 declares torch>=2.11; the reported runs used torch 2.5.1, so it is installed without
# its declared dependencies and only the ones the ESMC path imports are added.
#   bash envs/install_esmc.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
conda env list | grep -q "^siteseer-esmc " || conda env create -f "$HERE/siteseer-esmc.yml"
conda run -n siteseer-esmc pip install --no-deps esm==3.4.1.post1
conda run -n siteseer-esmc pip install attrs==26.1.0 einops==0.8.2 cloudpathlib==0.25.0 zstd==1.5.7.2 \
    biotite==1.7.1 brotli==1.2.0 msgpack==1.2.2 msgpack-numpy==0.4.8 httpx==0.28.1 tenacity==9.1.4
conda run -n siteseer-esmc python -c "from esm.models import esmc; from esm.tokenization import EsmSequenceTokenizer; print('ESMC environment ready')"
