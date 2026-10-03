#!/bin/bash -l
#SBATCH --job-name=ner_combined_roberta
#SBATCH --account=project_465002928
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --output=lumi/logs/combined_%j.log
#SBATCH --error=lumi/logs/combined_%j.err
#
# Single job = fine-tune RoBERTa on the COMBINED dataset (unified labels), eval all 5 dev packs.
# Same recipe as train_hf_ner.py / train_array.sh, but one model over the union of datasets.
# The fine-tuned model is NOT persisted; --output-dir is a transient scratch checkpoint used only
# to run inference for predictions. MLflow logging works over the compute node's outbound network
# (creds from .env). Ensure stage.sh has pre-staged FacebookAI/roberta-base into $HF_SCRATCH.
# Submit from the repo root:  sbatch lumi/train_combined.sh
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
    python nlp_cyber_ner/train_combined_roberta.py \
        --base-model FacebookAI/roberta-base \
        --output-dir '$RETRAINED_DIR/combined-roberta' \
        --batch-size 2 --lr 2e-5 --epochs 10 --seed 42 --max-length 512 --bf16
"
