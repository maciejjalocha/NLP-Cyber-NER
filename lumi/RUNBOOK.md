# LUMI runbook — leakage-clean retraining of the 20 encoders

Reproduces the 20 Cyber-ThreaD token-classification checkpoints from their base encoders, on
**leakage-cleaned** data (union of all 4 valid sets removed from each train set), with a **uniform
recipe** (batch 2, lr 2e-5, 10 epochs, seed 42 — the AttackER/Deka recipe), then scores them with the
late-merge cross-dataset sweep and reports GPU-hours.

Account: `project_465002928`. Layout:
- **Repo (code + `data/` + small result CSVs)** lives in `~/repos/NLP-Cyber-NER`. Submit all jobs
  from there so `PROJ_ROOT` (= sbatch submit dir) resolves correctly.
- **Heavy paths on scratch** (`/scratch/project_465002928/jalocham`): HF cache in `hf/` and the 20
  checkpoints in `retrained/`. The script defaults already point here — nothing to set.

Base models are pre-staged from a login node so training doesn't re-download them (and starts fast);
the compute nodes do have outbound network (MLflow→DagsHub works from them), so `HF_HUB_OFFLINE=1` in
the job scripts is a speed/reproducibility choice, not a hard requirement.

Overridable env vars (defaults in the scripts): `SIF`, `VENV` (`$HOME/venvs/nlp-cyber-ner`),
`HF_SCRATCH` (`/scratch/project_465002928/$USER/hf`), `RETRAINED_DIR`
(`/scratch/project_465002928/$USER/retrained`), `PROJ_ROOT` (defaults to the sbatch submit dir).

> If the container path (`SIF`) has rotated since this was written, update it or export `SIF=...`.
> Find it under `/appl/local/laifs/containers/lumi-multitorch-*/`.

## 0. Get the repo onto LUMI (with `data/`)
```bash
# from your Mac
rsync -av --exclude .venv --exclude .git \
  /Users/m-proj/repositories/NLP-Cyber-NER/ \
  lumi:repos/NLP-Cyber-NER/
```
Then `ssh lumi` and `cd ~/repos/NLP-Cyber-NER`. Run every step below from here.
(Checkpoints and HF cache still go to `/scratch` via the script defaults; only code, `data/`,
and the small result CSVs live in the `$HOME` repo.)

## 1. Build the venv (login node, one-time)
```bash
bash lumi/setup_venv.sh
```

## 2. Stage the 5 base models (login node, needs internet)
```bash
bash lumi/stage.sh          # ~a few GB into $HF_SCRATCH; safe to re-run (resumes)
```

## 3. Smoke test — one array task on a single GCD
```bash
sbatch --array=0 lumi/train_array.sh     # index 0 = SecureBERT/dnrti
# watch it:
tail -f lumi/logs/retrain_*_0.log
```
Confirm it logs the leakage removal count, trains, and writes
`$RETRAINED_DIR/SecureBERT__dnrti/` (config.json + model + tokenizer + train_metrics.json).

## 4. Full retraining — all 20 models
```bash
sbatch lumi/train_array.sh               # --array=0-19 baked in; note the ARRAY jobid it prints
squeue --me
```
~14 GPU-hours total; wall-clock depends on how many array tasks run concurrently.

## 5. Score them (after all 20 finish)
```bash
sbatch lumi/eval.sh
tail -f lumi/logs/eval_*.log
```
Writes `artifacts/cross_retrained/`: 25 CSVs (`<arch>__<metric>.csv`), `REPORT.md`,
`DELTA_<arch>__<metric>.csv` (retrained − hub, i.e. the leakage effect; expect DNRTI & ATTACKER
diagonals and APTNER→DNRTI to be clearly negative), and **`predictions/`** — two CoNLL files per cell:
`<arch>-train-<train>-eval-<eval>.txt` (token + predicted **unified** tag, cross_dataset_model.py
style) and `.raw.txt` (token + predicted **original** tag, pre-merge — e.g. DNRTI's `SamFile` before
it becomes `Malware`). Token order matches `data/processed/<eval>/valid.unified` for easy diffing.

## 6. GPU-hours
```bash
bash lumi/gpu_hours.sh <ARRAY_JOBID>     # the jobid from step 4
```
Prints billed GCD-hours (sacct) and the pure `train_runtime` sum (compute-only cross-check).

## 7. Bring results back
```bash
# from your Mac
rsync -av lumi:repos/NLP-Cyber-NER/artifacts/cross_retrained/ \
  /Users/m-proj/repositories/NLP-Cyber-NER/artifacts/cross_retrained/
```
MLflow logging to DagsHub works from the compute nodes (they have outbound network) — set
`MLFLOW_TRACKING_URI` (and DagsHub creds) in your `.env`/environment and the sweep logs metrics live.

## Multi-head model (shared RoBERTa encoder, one head per dataset)
Separate single-job experiment (not part of the 20-encoder array). Fine-tunes **one shared
RoBERTa-base encoder** with **one token-classification head per dataset over that dataset's ORIGINAL
label set** — the transformer counterpart of the four BiLSTM `train_tokenmodel_*` scripts. Per-epoch
dataset sampling is **with replacement, proportional to each dataset's batch count** (reproduces the
BiLSTM scheme, so larger datasets' heads get more updates), and batches are homogeneous (one head per
forward pass). Same recipe (batch 2, lr 2e-5, 10 epochs, seed 42, bf16) and the same union-leakage
removal as the array jobs. Eval is per-dataset span-F1 on each **original-label** dev set.
```bash
sbatch lumi/train_multihead.sh           # ~single GCD; writes to $RETRAINED_DIR/multihead-roberta
tail -f lumi/logs/multihead_*.log
```
Writes `$RETRAINED_DIR/multihead-roberta/`: `model.pt` (encoder + heads), `label_maps.json` (per-dataset
original label lists), `dev_metrics.json` (per-dataset span-F1), and `train_metrics.json` (timing).
Per-dataset predictions + metrics are logged to MLflow as runs `train-multihead-roberta-eval-<ds>`
(needs `MLFLOW_TRACKING_URI` + creds in `.env`; prediction CoNLLs land in `models/predictions/`).
GPU-hours:
```bash
bash lumi/gpu_hours_multihead.sh         # reads train_metrics.json; sacct reads zero GCD-hours on LUMI
```

## Notes
- One GCD per fine-tune (models are 110–184M params); no torchrun/RCCL needed.
- `--array=0-19` maps row-major over `ARCHES × DATASETS` (see `train_hf_ner.resolve_index`).
- To retrain a single model interactively: `python -m nlp_cyber_ner.modeling.train_hf_ner
  --arch RoBERTa --dataset cyner --bf16 --output-root $RETRAINED_DIR`.
