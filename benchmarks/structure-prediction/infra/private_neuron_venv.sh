#!/usr/bin/env bash
# Copyright 2026 Scott Friedman
# SPDX-License-Identifier: Apache-2.0
#
# Private Neuron venv for a SHARED trn2 host (#107).
#
#   bash infra/private_neuron_venv.sh DEST           # before boltz-neuron's scripts/setup.sh
#   BOLTZ_NEURON_VENV=DEST/venv BOLTZ_NEURON_SKIP_SWAP=1 bash ~/boltz-neuron/scripts/setup.sh
#   bash infra/private_neuron_venv.sh --check DEST   # after: shared source untouched, venv private
#
# On the Neuron 2.33 DLAMI, torch_neuronx is an *editable* install from a shared source tree
# (/source-repos/...). A plain `cp -a` of the DLAMI venv still imports that tree, and
# boltz-neuron's setup.sh patches it in place (neuron_dynamo_backend/decompositions.py), which
# changes torch_neuronx for every user of the host. This script copies the venv and the
# torch_neuronx package into DEST and points the copy's editable finder at the private package.
set -euo pipefail
CHECK=0
if [ "${1:-}" = "--check" ]; then CHECK=1; shift; fi
DEST=${1:?usage: private_neuron_venv.sh [--check] DEST}
V=${NEURON_VENV:-/opt/aws_neuronx_venv_torch_neuronx_2_14_0}
mkdir -p "$DEST/tn-src"
finder() { ls "$DEST"/venv/lib/python3*/site-packages/__editable___torch_neuronx_*_finder.py 2>/dev/null | head -1; }
subst() {  # subst FILE OLD NEW: literal in-place replacement (portable, no sed -i)
  python3 - "$@" <<'PY'
import pathlib, sys
p, old, new = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
p.write_text(p.read_text().replace(old, new))
PY
}
mapped() {  # MAPPING['torch_neuronx'] of an editable finder
  python3 - "$1" <<'PY'
import ast, re, sys
s = open(sys.argv[1]).read()
m = re.search(r"^MAPPING[^=]*=\s*(\{.*?\})\s*$", s, re.M)
print(ast.literal_eval(m.group(1))["torch_neuronx"])
PY
}
if [ $CHECK = 1 ]; then
  SHARED=$(cat "$DEST/tn-src/SHARED_SOURCE")
  ROOT=$(git -C "$SHARED" rev-parse --show-toplevel 2>/dev/null || true)
  if [ -n "$ROOT" ] && ! git -C "$ROOT" diff --quiet -- "$SHARED"; then
    echo "FAIL: shared torch_neuronx source modified:"; git -C "$ROOT" diff --stat -- "$SHARED"; exit 1
  fi
  [ "$(mapped "$(finder)")" = "$DEST/tn-src/torch_neuronx" ] || { echo "FAIL: venv not private"; exit 1; }
  if [ -z "${SKIP_IMPORT_CHECK:-}" ]; then
    got=$("$DEST/venv/bin/python" -c "import torch_neuronx, os; print(os.path.dirname(torch_neuronx.__file__))")
    [ "$got" = "$DEST/tn-src/torch_neuronx" ] || { echo "FAIL: imports $got"; exit 1; }
  fi
  echo "OK: private torch_neuronx at $DEST/tn-src/torch_neuronx; shared source $SHARED unmodified"
  exit 0
fi
[ -d "$DEST/venv" ] || cp -a "$V" "$DEST/venv"
subst "$DEST/venv/bin/activate" "$V" "$DEST/venv"
F=$(finder); [ -n "$F" ] || { echo "no editable torch_neuronx finder in $DEST/venv (not a 2.33 DLAMI venv?)"; exit 1; }
SRC=$(mapped "$F")
if [ "$SRC" != "$DEST/tn-src/torch_neuronx" ]; then
  [ -d "$DEST/tn-src/torch_neuronx" ] || cp -a "$SRC" "$DEST/tn-src/torch_neuronx"
  echo "$SRC" > "$DEST/tn-src/SHARED_SOURCE"
  subst "$F" "'torch_neuronx': '$SRC'" "'torch_neuronx': '$DEST/tn-src/torch_neuronx'"
fi
[ "$(mapped "$F")" = "$DEST/tn-src/torch_neuronx" ] || { echo "remap failed"; exit 1; }
echo "private venv $DEST/venv; torch_neuronx -> $DEST/tn-src/torch_neuronx (shared: $(cat "$DEST/tn-src/SHARED_SOURCE"))"
