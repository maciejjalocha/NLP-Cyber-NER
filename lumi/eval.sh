#!/bin/bash -l
#SBATCH --job-name=ner_eval
#SBATCH --account=project_465002928
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=lumi/logs/eval_%j.log
#SBATCH --error=lumi/logs/eval_%j.err
#
# Score the 20 leakage-clean checkpoints with the late-merge cross-dataset sweep.
# Writes artifacts/cross_retrained/ (+ DELTA vs the hub baseline in artifacts/cross_pretrained/).
# Run AFTER train_array.sh has finished:  sbatch lumi/eval.sh
set -euo pipefail

module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF="${SIF:-/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260319_153422/lumi-multitorch-full-u24r64f21m43t29-20260319_153422.sif}"
VENV="${VENV:-$HOME/venvs/nlp-cyber-ner}"
PROJ_ROOT="${PROJ_ROOT:-$SLURM_SUBMIT_DIR}"
HF_SCRATCH="${HF_SCRATCH:-/scratch/project_465002928/$(whoami)/hf}"
RETRAINED_DIR="${RETRAINED_DIR:-/scratch/project_465002928/$(whoami)/retrained}"

mkdir -p lumi/logs

export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-eval"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
rm -rf "$MIOPEN_USER_DB_PATH"; mkdir -p "$MIOPEN_USER_DB_PATH"

srun singularity exec "$SIF" bash -c "
    set -euo pipefail
    if [ -n \"\${ROCR_VISIBLE_DEVICES:-}\" ] && [ -z \"\${HIP_VISIBLE_DEVICES:-}\" ]; then
        export HIP_VISIBLE_DEVICES=\"\$ROCR_VISIBLE_DEVICES\"
    fi
    unset ROCR_VISIBLE_DEVICES
    export HF_HOME='$HF_SCRATCH'
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    # Compute nodes have outbound network, so mlflow→DagsHub logging works. The sweep reads
    # MLFLOW_TRACKING_URI (+ creds) from .env via python-dotenv, so nothing to set here.
    source '$VENV/bin/activate'
    cd '$PROJ_ROOT'
    python -m nlp_cyber_ner.modeling.cross_dataset_pretrained \
        --source local --models-dir '$RETRAINED_DIR'
"
