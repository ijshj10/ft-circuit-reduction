#!/usr/bin/env bash
# One Cloud Batch task: sample the p this task was assigned and write it to GCS.
# BATCH_TASK_INDEX picks the point, so a preempted task is retried on its own and
# nothing else is lost.
set -euo pipefail

: "${CODE_TAR:?set CODE_TAR to gs://BUCKET/code.tar.gz}"
: "${OUT:?set OUT to gs://BUCKET/run1}"
: "${IDLE_ARGS:=}"          # e.g. "--idle-noise DEPOLARIZE1 --idle-scale 3"

apt-get update -qq && apt-get install -y -qq python3-pip > /dev/null
workdir=$(mktemp -d); cd "$workdir"
gsutil -q cp "$CODE_TAR" code.tar.gz && tar xzf code.tar.gz
pip3 install -q -r cloud/requirements.txt

# The 21-point grid the reference scripts use, 1e-2 down to 1e-4.
P=$(python3 -c "import numpy as np; print('%.10e' % (10**np.linspace(-2,-4,21))[${BATCH_TASK_INDEX:-0}])")
echo "task ${BATCH_TASK_INDEX:-0}: p=$P"

# Skip if this point is already in the output -- makes a re-submit cheap.
if gsutil -q stat "$OUT/point_p${P}.json" 2>/dev/null; then
  echo "already done"; exit 0
fi

python3 cloud/run_point.py --p "$P" --out "$OUT" $IDLE_ARGS \
    --workers "$(nproc)" --target-errors "${TARGET_ERRORS:-4000}"
