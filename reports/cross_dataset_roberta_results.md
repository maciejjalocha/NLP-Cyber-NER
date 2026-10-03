# Cross-dataset RoBERTa — notes

Transformer counterpart of the BiLSTM cross-dataset experiment (paper Section *Cross-Dataset
Evaluation Setup*, Tables `tab:cross_eval_matrix`, `_precision`, `_recall`,
`tab:unlabelled-span-f1-cross-dataset`, `tab:loose-span-f1-cross-dataset`).

- Fine-tuned **RoBERTa** separately on each of the four datasets, on each dataset's **original**
  label set; evaluated every model on all four dev sets → 4×4 span-F1 matrices (diagonal =
  intra-dataset, off-diagonal = cross-dataset).
- Unified 4-class space (Organization / System / Vulnerability / Malware) applied **after inference**
  by *prob-sum late merge*: sum softmax probabilities of each coarse class's fine children, then argmax.
- Recipe: batch 2, lr 2e-5, 10 epochs, linear schedule, warmup 0, seed 42, max-length 512, bf16.
- **Union-leakage removal**: each train set drops sentences whose tokens appear in any of the four dev
  sets, so all eval columns are full and leakage-free (counts in appendix).
- Source: `models/cross_retrained/REPORT.md` (RoBERTa), MLflow `retrained-cross-RoBERTa`.

Train datasets on rows, dev datasets on columns; diagonal in **bold**.

## Span-F1 (vs. paper Table `tab:cross_eval_matrix`)

**RoBERTa**

| Train \ Dev | DNRTI | ATTACKER | APTNER | CYNER |
|-------------|:---:|:---:|:---:|:---:|
| **DNRTI**    | **0.63** | 0.30 | 0.21 | 0.16 |
| **ATTACKER** | 0.34 | **0.61** | 0.26 | 0.37 |
| **APTNER**   | 0.39 | 0.32 | **0.59** | 0.55 |
| **CYNER**    | 0.25 | 0.24 | 0.32 | **0.73** |

**BiLSTM (paper)**

| Train \ Dev | DNRTI | ATTACKER | APTNER | CYNER |
|-------------|:---:|:---:|:---:|:---:|
| **DNRTI**    | **0.41** | 0.16 | 0.19 | 0.07 |
| **ATTACKER** | 0.09 | **0.23** | 0.01 | 0.02 |
| **APTNER**   | 0.31 | 0.16 | **0.41** | 0.18 |
| **CYNER**    | 0.05 | 0.04 | 0.06 | **0.40** |

- Diagonals: DNRTI 0.41→0.63, ATTACKER 0.23→0.61, APTNER 0.41→0.59, CYNER 0.40→0.73.
- No off-diagonal beats its column diagonal (same as BiLSTM).
- Highest off-diagonal region: APTNER↔DNRTI (APTNER→DNRTI 0.39). Leakage-clean.

## Precision / Recall (vs. paper Tables `_precision`, `_recall`)

**RoBERTa precision**

| Train \ Dev | DNRTI | ATTACKER | APTNER | CYNER |
|-------------|:---:|:---:|:---:|:---:|
| **DNRTI**    | **0.63** | 0.29 | 0.15 | 0.13 |
| **ATTACKER** | 0.38 | **0.61** | 0.20 | 0.31 |
| **APTNER**   | 0.48 | 0.40 | **0.51** | 0.58 |
| **CYNER**    | 0.30 | 0.29 | 0.28 | **0.76** |

**RoBERTa recall**

| Train \ Dev | DNRTI | ATTACKER | APTNER | CYNER |
|-------------|:---:|:---:|:---:|:---:|
| **DNRTI**    | **0.64** | 0.32 | 0.38 | 0.21 |
| **ATTACKER** | 0.32 | **0.62** | 0.40 | 0.45 |
| **APTNER**   | 0.33 | 0.27 | **0.69** | 0.52 |
| **CYNER**    | 0.22 | 0.21 | 0.37 | **0.71** |

- Diagonal precision/recall roughly balanced (BiLSTM was precision≫recall). APTNER recall-leaning
  (0.69 vs 0.51).

## Unlabelled / loose span-F1 (vs. paper Tables `_unlabelled`, `_loose`)

**RoBERTa unlabelled** (boundaries correct, label ignored)

| Train \ Dev | DNRTI | ATTACKER | APTNER | CYNER |
|-------------|:---:|:---:|:---:|:---:|
| **DNRTI**    | **0.71** | 0.43 | 0.37 | 0.44 |
| **ATTACKER** | 0.55 | **0.67** | 0.36 | 0.48 |
| **APTNER**   | 0.55 | 0.45 | **0.66** | 0.60 |
| **CYNER**    | 0.57 | 0.42 | 0.58 | **0.82** |

**RoBERTa loose** (partial overlap, same label)

| Train \ Dev | DNRTI | ATTACKER | APTNER | CYNER |
|-------------|:---:|:---:|:---:|:---:|
| **DNRTI**    | **0.69** | 0.40 | 0.27 | 0.20 |
| **ATTACKER** | 0.44 | **0.69** | 0.39 | 0.48 |
| **APTNER**   | 0.44 | 0.41 | **0.67** | 0.59 |
| **CYNER**    | 0.29 | 0.38 | 0.35 | **0.78** |

- Largest strict→unlabelled jump on the CYNER row (CYNER→DNRTI 0.25→0.57, CYNER→APTNER 0.32→0.58):
  right spans, wrong coarse class.

## Appendix — leakage removal counts

Sentences removed per train set (union of all four dev sets): DNRTI 546, ATTACKER 207, APTNER 358,
CYNER 25.
