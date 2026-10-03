#!/bin/bash -l
# Pre-stage the 5 base encoders into the scratch HF cache so compute nodes (offline) can load them.
# Run on a LOGIN node (has internet), from the repo root, AFTER setup_venv.sh.
set -euo pipefail

module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF="${SIF:-/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260319_153422/lumi-multitorch-full-u24r64f21m43t29-20260319_153422.sif}"
VENV="${VENV:-$HOME/venvs/nlp-cyber-ner}"
HF_SCRATCH="${HF_SCRATCH:-/scratch/project_465002928/$(whoami)/hf}"
mkdir -p "$HF_SCRATCH"

MODELS="ehsanaghaei/SecureBERT SynamicTechnologies/CYBERT jackaduma/SecBERT microsoft/deberta-v3-base FacebookAI/roberta-base"

echo "HF_SCRATCH=$HF_SCRATCH"
singularity exec "$SIF" bash -c "
    set -euo pipefail
    source '$VENV/bin/activate'
    export HF_HOME='$HF_SCRATCH'
    for m in $MODELS; do
        echo \"=== staging \$m ===\"
        huggingface-cli download \"\$m\" --resume-download
    done
"
echo "staged base models into $HF_SCRATCH"
