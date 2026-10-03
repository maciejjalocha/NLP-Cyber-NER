#!/bin/bash -l
# Report GPU (GCD) hours for the COMBINED RoBERTa fine-tuning job.
#   usage: bash lumi/gpu_hours_combined.sh [MODEL_DIR]
# MODEL_DIR defaults to the transient checkpoint dir used by lumi/train_combined.sh.
# sacct reports zero GCD-hours on LUMI, so we rely on the pure train_runtime written to
# train_metrics.json by nlp_cyber_ner/modeling/combined_hf_ner.py (compute only).
set -euo pipefail

DEFAULT_DIR="/scratch/project_465002928/$(whoami)/retrained/combined-roberta"
MODEL_DIR="${1:-$DEFAULT_DIR}"

echo "== pure train_runtime (train_metrics.json in $MODEL_DIR) =="
python3 - "$MODEL_DIR" <<'PY'
import json, os, sys
path = os.path.join(sys.argv[1], "train_metrics.json")
if not os.path.exists(path):
    sys.exit(f"  no train_metrics.json at {path} (did the job finish?)")
d = json.load(open(path))
t = float(d.get("train_runtime_s", 0.0))
print(f"  base_model      = {d.get('base_model')}")
print(f"  train_sentences = {d.get('train_sentences')}")
print(f"  epochs          = {d.get('epochs')}")
print(f"  train_runtime   = {t:.0f} s = {t / 3600:.2f} h")
PY
