# LUMI GPU-hours — RoBERTa fine-tuning

This report summarizes the measured training time for the three RoBERTa fine-tuning experiments run on the LUMI-G partition: cross-dataset, combined, and multi-head training.

## Hardware and accounting unit

Each SLURM training job requested one accelerator with `--gpus-per-node=1`. On LUMI, the relevant accelerator accounting unit is the AMD MI250X **Graphics Compute Die (GCD)**. A physical MI250X module contains two GCDs, but these experiments were configured to use one GCD per running job.

Accordingly, the values below are reported as **GCD-hours**. They are calculated from the `train_runtime_s` values written by the training scripts:

$$
\text{GCD-hours} = \frac{\text{training runtime in seconds}}{3600}
$$

These are pure measured training runtimes, not necessarily the final billed allocation reported by SLURM. The provided `gpu_hours.sh` workflow notes that `sacct` reports zero GCD-hours on LUMI for these jobs, so the runtime values are used as the compute-only comparison.

## Training cost

| Experiment | Training configuration | Training runtime | Compute-only cost |
|---|---|---:|---:|
| Cross-dataset | 20 array tasks, one architecture/dataset model per task | 3,925 s | **1.09 GCD-hours** |
| Combined | One RoBERTa model trained on the unified union of four datasets | 4,145 s | **1.15 GCD-hours** |
| Multi-head | One shared RoBERTa encoder with four dataset-specific heads | 4,232 s | **1.18 GCD-hours** |

The cross-dataset value is the sum of the training runtimes from all 20 array tasks. It therefore represents aggregate accelerator time across the experiment, not elapsed wall-clock time for the SLURM array. Its maximum simultaneous accelerator usage depended on how many array tasks LUMI scheduled concurrently.

## Interpretation

The combined and multi-head experiments each used one GCD for a single training job. The multi-head model took 87 seconds longer than the combined model, an increase of approximately 2.1\%. The cross-dataset experiment required 3,925 aggregate training seconds across its 20 models, or 1.09 aggregate GCD-hours.

The appropriate summary for the experiments is therefore:

> RoBERTa fine-tuning used 1.09 aggregate GCD-hours for the cross-dataset experiment, 1.15 GCD-hours for the combined experiment, and 1.18 GCD-hours for the multi-head experiment, based on measured training runtime on one LUMI MI250X GCD per job.

## MI250X module-equivalent hours

Dividing the GCD-hours by two produces physical MI250X-module-equivalent hours, because each MI250X module contains two GCDs:

| Experiment | GCD-hours | MI250X-module-equivalent hours |
|---|---:|---:|
| Cross-dataset | 1.09 | 0.55 |
| Combined | 1.15 | 0.58 |
| Multi-head | 1.18 | 0.59 |

These divided values should only be used when reporting normalized full-module-equivalent time. They should not replace GCD-hours in the primary report, since the jobs allocated one GCD rather than a complete two-GCD MI250X module.

## Reproducibility

The measurements were obtained with the following LUMI helper scripts:

- `lumi/gpu_hours.sh` for the cross-dataset array
- `lumi/gpu_hours_combined.sh` for the combined model
- `lumi/gpu_hours_multihead.sh` for the multi-head model

The first attempted multi-head command included an extra `.sh` suffix and failed because `lumi/gpu_hours_multihead.sh.sh` does not exist. The subsequent invocation of the correct script produced the reported 4,232-second runtime.
