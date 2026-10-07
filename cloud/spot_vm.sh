#!/usr/bin/env bash
# Create a Spot VM that runs the whole p scan and deletes itself when finished.
#
#   ./cloud/spot_vm.sh BUCKET [ZONE] [MACHINE]
#
# Re-run it after a preemption: finished points are in the bucket and get skipped.
set -euo pipefail
BUCKET=${1:?usage: spot_vm.sh BUCKET [ZONE] [MACHINE]}

# The SDK installer only appends to ~/.bashrc, so a fresh non-login shell does not
# see it; pick it up here rather than failing on a bare "command not found".
if ! command -v gsutil > /dev/null && [ -d "$HOME/google-cloud-sdk/bin" ]; then
  PATH="$HOME/google-cloud-sdk/bin:$PATH"
fi
command -v gcloud > /dev/null || { echo "gcloud not found: install the SDK, or source ~/.bashrc"; exit 1; }
gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q . \
  || { echo "not authenticated: run  gcloud auth login --no-launch-browser"; exit 1; }
gsutil ls -b "gs://$BUCKET" > /dev/null 2>&1 \
  || { echo "bucket gs://$BUCKET not reachable: create it with  gsutil mb -l us-central1 gs://$BUCKET"; exit 1; }
ZONE=${2:-us-central1-a}
MACHINE=${3:-c3-highcpu-44}
# Not NAME: WSL exports that as the Windows machine name, which is not a valid
# instance name and silently overrode the default.
VM_NAME=${VM_NAME:-pscan-$(date +%m%d-%H%M)}
IDLE_ARGS=${IDLE_ARGS:-}            # "" dephasing | "--idle-noise DEPOLARIZE1" | + "--idle-scale 3"

tar czf /tmp/code.tar.gz evaluate.py sweep.py bell_reduction.py circuit.py \
    low/simulations/utils.py low/simulations/floqetified_steane_custom.py cloud/
gsutil -q cp /tmp/code.tar.gz "gs://$BUCKET/code.tar.gz"

gcloud compute instances create "$VM_NAME" \
  --zone "$ZONE" --machine-type "$MACHINE" \
  ${PROVISIONING:---provisioning-model=SPOT --instance-termination-action=DELETE} \
  --image-family=debian-12 --image-project=debian-cloud --boot-disk-size=20GB \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --metadata-from-file startup-script=cloud/vm_startup.sh \
  --metadata code-tar="gs://$BUCKET/code.tar.gz",out="gs://$BUCKET/run1",idle-args="$IDLE_ARGS",target-errors="${TARGET_ERRORS:-4000}"

cat <<EOF

created $VM_NAME ($MACHINE, ${PROVISIONING:+standard}${PROVISIONING:-SPOT}) in $ZONE

  progress   gcloud compute instances get-serial-port-output $VM_NAME --zone $ZONE | tail -40
  points     gsutil ls gs://$BUCKET/run1/
  collect    python cloud/merge_points.py --points gs://$BUCKET/run1 --out result.json
  stop early gcloud compute instances delete $VM_NAME --zone $ZONE --quiet

The VM deletes itself when the scan finishes, so it stops billing on its own.
EOF
