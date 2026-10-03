# Multi-head RoBERTa — results

Fine-tuned **multi-head RoBERTa**: one shared `roberta-base` encoder with one token-classification
head per dataset, each over that dataset's **original** label set (no label unification). Recipe:
batch 2, lr 2e-5, 10 epochs, linear schedule, warmup 0, seed 42, max-length 512, bf16. Cross-dataset
batching reproduces the BiLSTM multi-head scheme (per-epoch dataset sampling **with replacement**,
proportional to each dataset's batch count). Union-leakage removal applied (train sentences whose
tokens appear in any of the four valid sets dropped). Scores are **span-F1 on each dataset's
original-label dev set**. Source: `models/multihead_roberta/dev_metrics.json`.

Only the **Shared: Both** (fully-shared encoder) variant maps cleanly to a monolithic transformer, so
the emb-only / LSTM-only columns of the paper's BiLSTM table have no RoBERTa analog.

## Multi-head span-F1 vs. the BiLSTM multi-head (paper Table `tab:multi_head_matrix`)

BiLSTM columns are reproduced from the paper for comparison; the RoBERTa column is this run.

| Dataset  | BiLSTM Shared: LSTM | BiLSTM Shared: EMB | BiLSTM Shared: Both | BiLSTM Reference | **RoBERTa multi-head (Shared: Both)** |
|----------|:---:|:---:|:---:|:---:|:---:|
| DNRTI    | 0.43 | 0.52 | 0.52 | 0.45 | **0.61** |
| ATTACKER | 0.01 | 0.19 | 0.21 | 0.04 | **0.42** |
| APTNER   | 0.36 | 0.38 | 0.37 | 0.35 | **0.60** |
| CYNER    | 0.34 | 0.41 | 0.39 | 0.40 | **0.80** |

The fully-shared RoBERTa multi-head improves span-F1 over the best BiLSTM multi-head variant on every
dataset, with the largest gains on ATTACKER (0.21 → 0.42) and CYNER (0.41 → 0.80).

## Full metric breakdown — RoBERTa multi-head (Shared: Both)

Strict span-F1 with its precision/recall, plus the unlabelled and loose (partial-overlap, same-label)
span-F1 variants — analogous to the paper's cross-dataset metric tables.

| Dataset  | Precision | Recall | Span-F1 | Unlabelled F1 | Loose F1 |
|----------|:---:|:---:|:---:|:---:|:---:|
| DNRTI    | 0.63 | 0.59 | 0.61 | 0.68 | 0.69 |
| ATTACKER | 0.44 | 0.41 | 0.42 | 0.54 | 0.54 |
| APTNER   | 0.57 | 0.63 | 0.60 | 0.71 | 0.67 |
| CYNER    | 0.80 | 0.79 | 0.80 | 0.84 | 0.84 |

APTNER was rescored after converting its native BIOES tags (`S-X` to `B-X` and `E-X` to `I-X`),
because `span_f1` expects BIO tags. Its corrected score is precision 0.57, recall 0.63, and
span-F1 0.60. The original reported APTNER value (0.44) was therefore an incompatible BIOES
scoring result. The other datasets use BIO-compatible labels and are unchanged. CYNER is strongest
across every metric.

## Single-dataset non-merged RoBERTa reference

For comparison, the diagonal single-dataset RoBERTa predictions were rescored against each
dataset's original-label development set. APTNER `S/E` tags were normalized to `B/I` before
calling `span_f1`; the models and predictions were not changed.

| Dataset  | Precision | Recall | Span-F1 | Unlabelled F1 | Loose F1 |
|----------|:---:|:---:|:---:|:---:|:---:|
| DNRTI    | 0.6164 | 0.6243 | **0.6203** | 0.7004 | 0.6919 |
| ATTACKER | 0.4257 | 0.4246 | **0.4252** | 0.5338 | 0.5635 |
| APTNER   | 0.5609 | 0.6189 | **0.5885** | 0.7084 | 0.6618 |
| CYNER    | 0.7738 | 0.7624 | **0.7681** | 0.8411 | 0.8093 |

Source: `models/single_dataset_reference_metrics.json`.

## RoBERTa multi-head versus single-dataset reference

| Dataset  | Reference span-F1 | Multi-head span-F1 | Difference |
|----------|:---:|:---:|:---:|
| DNRTI    | 0.6203 | 0.61 | -0.0103 |
| ATTACKER | 0.4252 | 0.42 | -0.0052 |
| APTNER   | 0.5885 | 0.60 | **+0.0115** |
| CYNER    | 0.7681 | 0.80 | **+0.0319** |

Difference is multi-head minus single-dataset reference. Both columns use original-label
evaluation; APTNER tags were normalized from BIOES to BIO before scoring.
