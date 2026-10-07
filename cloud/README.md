# Running a p scan on Spot VMs

The scan is 21 independent points. Each is one task, writes its own file, and skips
itself if that file already exists -- so a preempted task costs one point and a
re-submit is cheap. That is the whole reason for `run_point.py`: `evaluate.py`'s own
p scan is a single adaptive loop that writes nothing until all 21 points finish,
which on Spot means an 80%-complete scan can vanish.

## Before anything else: check the environment matches

A different stim version can change sampling, so `requirements.txt` pins it. Confirm
the cloud environment reproduces the published data before trusting a run from it:

    IDLE_NOISE=Z_ERROR python3 scan_p_depol.py --circuits threeflag \
        --idle-scale 1 --target-errors 300 --p-min 5e-4 --tag envcheck

and compare against `low/simulations/data/flagged_three_qubit_data_by_p.json`
at the same p. Ratios should sit inside the ~10% the event counts allow. This takes
minutes and would have caught every wrong turn this project has taken.

## 1. Package the code

    tar czf code.tar.gz evaluate.py sweep.py bell_reduction.py circuit.py \
        low/simulations/utils.py low/simulations/floqetified_steane_custom.py cloud/
    gsutil cp code.tar.gz gs://BUCKET/code.tar.gz
    gsutil cp cloud/task.sh gs://BUCKET/task.sh

112 KB total; there is no dataset to move.

## 2. Submit

Edit `BUCKET` in `batch_job.json`, then

    gcloud batch jobs submit pscan-deph --location us-central1 \
        --config cloud/batch_job.json

For a different noise model, set `IDLE_ARGS` in the job's environment:

    ""                                        dephasing p/10 (the reference model)
    "--idle-noise DEPOLARIZE1"                depolarizing p/10
    "--idle-noise DEPOLARIZE1 --idle-scale 3" depolarizing 3p/10

## 3. Merge

    python cloud/merge_points.py --points gs://BUCKET/run1 --out scan_pscandeph_29.json

It warns if any of the 21 are missing rather than silently producing a short curve.
Re-submitting the same job re-runs only those.

## Sizing

Work is very unevenly distributed: cost per point goes as 1/p^2, so the last two
points are most of the scan. With `parallelism: 21` every point gets its own VM and
the job finishes in the time of the slowest one -- about 2 h on 8 vCPU for p = 1e-4,
against ~9 h for the whole scan on 12 local cores. Fewer, larger machines also work;
tasks are independent either way.

Spot prices vary, but 21 x c3-highcpu-8 for ~2 h is a few dollars.

## Fallback: one VM, no Batch

    gcloud compute instances create pscan --machine-type c3-highcpu-88 \
        --provisioning-model SPOT --image-family debian-12 --image-project debian-cloud
    # then, on the VM, after unpacking the code:
    for P in $(python3 -c "import numpy as np; print(' '.join('%.10e'%x for x in 10**np.linspace(-2,-4,21)))"); do
        python3 cloud/run_point.py --p $P --out gs://BUCKET/run1 --workers $(nproc)
    done

Same per-point files, same merge, no Batch config -- but a preemption stops the loop,
so re-run it and the finished points are skipped.
