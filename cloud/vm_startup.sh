#!/usr/bin/env bash
# Runs on the Spot VM at boot: fetch code, sample every p that is not already in the
# bucket, then shut down so a preempted-and-restarted VM resumes rather than repeats,
# and an idle VM stops costing money.
#
# Watch it with:
#   gcloud compute instances get-serial-port-output NAME --zone ZONE | tail -40
set -uo pipefail
# Plain redirection, not `exec > >(tee ...)`: that puts a pipe between the script
# and its log, and when the pipe broke mid-run the next write killed the script --
# losing 1.6 h of sampling with the VM left idle and billing. A file has no such
# failure mode; use `gcloud compute ssh ... tail -f /var/log/pscan.log` to follow.
exec >> /var/log/pscan.log 2>&1

meta() { curl -sf -H "Metadata-Flavor: Google" \
         "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
CODE_TAR=$(meta code-tar); OUT=$(meta out)
IDLE_ARGS=$(meta idle-args || echo ""); TARGET=$(meta target-errors || echo 4000)

apt-get update -qq && apt-get install -y -qq python3-pip > /dev/null
cd "$(mktemp -d)"
gsutil -q cp "$CODE_TAR" code.tar.gz && tar xzf code.tar.gz
pip3 install -q --break-system-packages -r cloud/requirements.txt

mapfile -t PS < <(python3 -c "
import numpy as np
print('\n'.join('%.10e' % x for x in 10 ** np.linspace(-2, -4, 21)))")

# Cheapest first: if the VM is preempted early, the expensive tail is what is left,
# and the next boot picks it up.
for P in "${PS[@]}"; do
  if gsutil -q stat "$OUT/point_p${P}.json" 2>/dev/null; then
    echo "skip  p=$P (already done)"; continue
  fi
  echo "=== p=$P  $(date -u +%H:%M:%S)"
  python3 cloud/run_point.py --p "$P" --out "$OUT" $IDLE_ARGS \
      --workers "$(nproc)" --target-errors "$TARGET" || echo "FAILED p=$P"
done

echo "=== all points done, shutting down"
gsutil -q cp /var/log/pscan.log "$OUT/pscan.log" || true
shutdown -h now
