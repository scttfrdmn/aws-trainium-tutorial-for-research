#!/usr/bin/env bash
# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
#
# Download the ColabFold databases once, stage them through S3, and build the index each host
# needs. Same database contents everywhere; only the index differs (GPU-padded on g7e,
# CPU index on r7a).
#
#   stage_databases.sh upload  <workdir> s3://BUCKET/PREFIX   # any host with ~1 TB free
#   stage_databases.sh restore <workdir> s3://BUCKET/PREFIX gpu   # g7e.2xlarge
#   stage_databases.sh restore <workdir> s3://BUCKET/PREFIX cpu   # r7a.48xlarge
#
# Templates are skipped (SKIP_TEMPLATES): Boltz-2 runs MSA-only by default and ColabFold's
# default is no templates.
set -euo pipefail

COLABFOLD_TAG="v1.6.3"
SETUP_URL="https://raw.githubusercontent.com/sokrypton/ColabFold/${COLABFOLD_TAG}/setup_databases.sh"
mode="${1:?upload|restore}"; workdir="${2:?workdir}"; s3="${3:?s3://bucket/prefix}"
mkdir -p "$workdir"; cd "$workdir"
curl -fsSL "$SETUP_URL" -o setup_databases.sh
chmod +x setup_databases.sh
touch SKIP_TEMPLATES

case "$mode" in
  upload)
    DOWNLOADS_ONLY=1 ./setup_databases.sh "$workdir"
    aws s3 sync "$workdir" "$s3" --exclude '*' --include '*.tar.gz' --include 'DOWNLOADS_READY' \
      --include 'setup_databases.sh' --include 'SKIP_TEMPLATES'
    ;;
  restore)
    kind="${4:?gpu|cpu}"
    aws s3 sync "$s3" "$workdir"
    test -f DOWNLOADS_READY || { echo "S3 staging incomplete (no DOWNLOADS_READY)"; exit 1; }
    if [ "$kind" = gpu ]; then GPU=1 ./setup_databases.sh "$workdir"; else ./setup_databases.sh "$workdir"; fi
    ;;
  *) echo "unknown mode $mode"; exit 2 ;;
esac

# Record what was built (names, sizes, mmseqs version) next to the databases.
{
  echo "colabfold_setup=${COLABFOLD_TAG}"
  echo "mmseqs=$(mmseqs version 2>/dev/null || echo unknown)"
  ls -l --time-style=+%F ./*_READY 2>/dev/null || true
  du -sh "$workdir"
} > "$workdir/DATABASES.txt"
cat "$workdir/DATABASES.txt"
