#!/bin/bash -l
# Report GPU (GCD) hours for the retraining job.
#   usage: bash lumi/gpu_hours.sh <ARRAY_JOBID> [RETRAINED_DIR]
# Prints two numbers:
#   1) billed GCD-hours from sacct (Elapsed x AllocGPUS, incl. load/queue-inside-job overhead)
#   2) pure training seconds summed from each model's train_metrics.json (compute only)
set -euo pipefail

JOBID="${1:?usage: gpu_hours.sh <ARRAY_JOBID> [RETRAINED_DIR]}"
RETRAINED_DIR="${2:-/scratch/project_465002928/$(whoami)/retrained}"

echo "== pure train_runtime sum (train_metrics.json in $RETRAINED_DIR) =="
python3 - "$RETRAINED_DIR" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
files = sorted(glob.glob(os.path.join(root, "*", "train_metrics.json")))
tot = 0.0
for f in files:
    d = json.load(open(f))
    tot += float(d.get("train_runtime_s", 0.0))
print(f"  models = {len(files)}")
print(f"  sum train_runtime = {tot:.0f} s = {tot / 3600:.2f} h")
PY
