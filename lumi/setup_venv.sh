#!/bin/bash -l
# One-time setup: build a venv layered on the LUMI AI singularity container and install the
# Python deps our NER fine-tuning needs (torch comes from the container via --system-site-packages).
# Run on a LOGIN node (has internet), from the repo root.
set -euo pipefail

module --force purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

export SIF="${SIF:-/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260319_153422/lumi-multitorch-full-u24r64f21m43t29-20260319_153422.sif}"
VENV="${VENV:-$HOME/venvs/nlp-cyber-ner}"
PROJ_ROOT="${PROJ_ROOT:-$PWD}"

echo "SIF=$SIF"
echo "VENV=$VENV"
echo "PROJ_ROOT=$PROJ_ROOT"

singularity exec "$SIF" bash -c "
    set -euo pipefail
    python -m venv --system-site-packages '$VENV'
    source '$VENV/bin/activate'
    pip install --upgrade pip
    # torch is inherited from the container; do not reinstall it.
    pip install 'transformers>=4.44' 'accelerate>=1.1.0' sentencepiece seqeval \
                loguru python-dotenv jsonlines mlflow tqdm
    pip install -e '$PROJ_ROOT' --no-deps
    python -c 'import torch, transformers, accelerate, sentencepiece; \
print(\"torch\", torch.__version__, \"| transformers\", transformers.__version__, \
\"| rocm\", getattr(torch.version, \"hip\", None))'
"
echo "venv ready: $VENV"
