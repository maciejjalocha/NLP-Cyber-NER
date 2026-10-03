#!/bin/bash -l
#SBATCH --job-name=ner_multihead_roberta
#SBATCH --account=project_465002928
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --output=lumi/logs/multihead_%j.log
#SBATCH --error=lumi/logs/multihead_%j.err
#
# Single job = fine-tune a multi-head RoBERTa: one SHARED encoder + one head per dataset over that
# dataset's ORIGINAL label set. Per-epoch dataset sampling is WITH replacement, proportional to each
# dataset's batch count (reproduces the BiLSTM train_tokenmodel_* scheme). Eval = per-dataset span-F1
# on each dataset's original-label dev set. Same fine-tuning recipe as train_hf_ner.py / train_combined.sh.
# Model weights + label maps + dev metrics are written under --output-dir (scratch on LUMI).
# MLflow-free: metrics land in dev_metrics.json. Ensure stage.sh has pre-staged FacebookAI/roberta-base
# into $HF_SCRATCH. Submit from the repo root:  sbatch lumi/train_multihead.sh
set -euo pipefail

module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF="${SIF:-/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260319_153422/lumi-multitorch-full-u24r64f21m43t29-20260319_153422.sif}"
VENV="${VENV:-$HOME/venvs/nlp-cyber-ner}"
PROJ_ROOT="${PROJ_ROOT:-$SLURM_SUBMIT_DIR}"
HF_SCRATCH="${HF_SCRATCH:-/scratch/project_465002928/$(whoami)/hf}"
RETRAINED_DIR="${RETRAINED_DIR:-/scratch/project_465002928/$(whoami)/retrained}"

mkdir -p lumi/logs "$RETRAINED_DIR"

# MIOPEN kernel cache on node-local /tmp (per LUMI guidance).
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-${SLURM_JOB_ID:-0}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
rm -rf "$MIOPEN_USER_DB_PATH"; mkdir -p "$MIOPEN_USER_DB_PATH"

srun singularity exec "$SIF" bash -c "
    set -euo pipefail
    # SLURM exposes the GCD via ROCR_VISIBLE_DEVICES; torch wants HIP_VISIBLE_DEVICES.
    if [ -n \"\${ROCR_VISIBLE_DEVICES:-}\" ] && [ -z \"\${HIP_VISIBLE_DEVICES:-}\" ]; then
        export HIP_VISIBLE_DEVICES=\"\$ROCR_VISIBLE_DEVICES\"
    fi
    unset ROCR_VISIBLE_DEVICES

    export HF_HOME='$HF_SCRATCH'
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1

    source '$VENV/bin/activate'
    cd '$PROJ_ROOT'
    python -m nlp_cyber_ner.modeling.train_hf_multihead \
        --arch RoBERTa \
        --output-dir '$RETRAINED_DIR/multihead-roberta' \
        --batch-size 2 --lr 2e-5 --epochs 10 --seed 42 --max-length 512 --bf16
"
