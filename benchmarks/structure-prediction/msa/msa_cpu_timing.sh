#!/usr/bin/env bash
# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
#
# r7a.48xlarge: MMseqs2-CPU timing on the CPU-built index (output discarded). The index is
# preloaded into RAM with vmtouch (ColabFold: fast single-query CPU search needs the index in
# memory, 768 GB-1 TB for ColabFoldDB). Never run this on trn2.3xlarge (124 GB host RAM).
#   msa_cpu_timing.sh <db_dir> <targets.json> <timing.json>
set -euo pipefail
db="${1:?db dir}"; targets="${2:?targets.json}"; timing="${3:?timing json}"
here="$(cd "$(dirname "$0")" && pwd)"

ram_gib=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
if [ "$ram_gib" -lt 768 ]; then
  echo "host has ${ram_gib} GiB RAM; the in-memory index needs >= 768 GiB (use r7a.48xlarge)"; exit 1
fi
command -v vmtouch >/dev/null || sudo apt-get install -y vmtouch
sudo vmtouch -f -t -l -d -m 1500G "$db"/*.idx*
sleep 5

python "$here/run_search.py" --mode cpu --targets "$targets" --db "$db" \
  --out /tmp/msa-cpu-scratch --timing "$timing" --discard
