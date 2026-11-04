#!/usr/bin/env bash
# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
#
# g7e.2xlarge: MMseqs2-GPU MSAs, one query at a time, GPU server holding both databases.
#   msa_gpu.sh <db_dir> <targets.json> <out_msa_dir> <timing.json>
set -euo pipefail
db="${1:?db dir}"; targets="${2:?targets.json}"; out="${3:?msa out dir}"; timing="${4:?timing json}"
here="$(cd "$(dirname "$0")" && pwd)"
export CUDA_VISIBLE_DEVICES=0

mmseqs gpuserver "$db/colabfold_envdb_202108_db" --max-seqs 10000 --db-load-mode 0 --prefilter-mode 1 &
p1=$!
mmseqs gpuserver "$db/uniref30_2302_db" --max-seqs 10000 --db-load-mode 0 --prefilter-mode 1 &
p2=$!
trap 'kill $p1 $p2 2>/dev/null || true' EXIT
sleep 60  # servers load the databases before the first query

python "$here/run_search.py" --mode gpu --targets "$targets" --db "$db" --out "$out" --timing "$timing"
ls "$out"/*.a3m | wc -l
