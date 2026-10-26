#!/usr/bin/env bash
# P2Rank 2.5.1 (needs Java, e.g. from envs/siteseer-tools.yml).
#   bash envs/install_p2rank.sh [TARGET_DIR]       (default: ./external)
set -euo pipefail
TARGET="${1:-external}"
URL=https://github.com/rdk/p2rank/releases/download/2.5.1/p2rank_2.5.1.tar.gz
SHA256=d243f2d9036ac053fefb9407b5fe1c85f4fe077c519fd975ac585e995feab274
mkdir -p "$TARGET"
cd "$TARGET"
[ -f p2rank_2.5.1.tar.gz ] || curl -L -o p2rank_2.5.1.tar.gz "$URL"
echo "$SHA256  p2rank_2.5.1.tar.gz" | sha256sum -c -
[ -d p2rank_2.5.1 ] || tar xzf p2rank_2.5.1.tar.gz
echo "P2Rank ready: $(pwd)/p2rank_2.5.1/prank (export P2RANK_HOME=$(pwd)/p2rank_2.5.1)"
