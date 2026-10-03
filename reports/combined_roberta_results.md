# Combined (unified) RoBERTa — results

Fine-tuned **RoBERTa** on the **union of all four datasets** (CYNER + APTNER + ATTACKER + DNRTI) in
the **unified** 9-tag BIO label space — the transformer counterpart of the paper's BiLSTM
"unified-datasets model" (Section *Unified-datasets model*, Table `tab:bigmodel_vs_original`). One
model is trained and then evaluated on each dataset's unified-label dev set plus the pooled
**combined** dev set. Recipe: batch 2, lr 2e-5, 10 epochs, linear schedule, warmup 0, seed 42,
max-length 512, bf16. Union-leakage removal applied (train sentences whose tokens appear in any of the
four valid sets are dropped): 18,222 → **17,086** train sentences (1,136 removed). Training wall time
**4,146 s** (~69 min) on one LUMI MI250X GCD. Source: MLflow experiments
`train-combined-roberta-eval-*` and `models/combined-roberta/train_metrics.json`.

## Combined-model span-F1 vs. the BiLSTM combined model (paper Table `tab:bigmodel_vs_original`)

The paper's table compares a BiLSTM trained on the combined data ("Combined") against BiLSTMs trained
on each single dataset ("Original"). Both BiLSTM columns are reproduced from the paper; **RoBERTa
Combined** is this run, and **RoBERTa Original** is the single-dataset RoBERTa diagonal reproduced
from `cross_dataset_roberta_results.md` (train and eval on the same dataset).

| val \ train | BiLSTM Combined | BiLSTM Original | **RoBERTa Combined** | RoBERTa Original |
|-------------|:---:|:---:|:---:|:---:|
| Combined    | 0.38 | –    | **0.53** | –    |
| DNRTI       | 0.49 | 0.41 | **0.52** | 0.63 |
| ATTACKER    | 0.33 | 0.23 | **0.51** | 0.61 |
| APTNER      | 0.30 | 0.41 | **0.49** | 0.59 |
| CYNER       | 0.37 | 0.39 | **0.67** | 0.73 |

The combined RoBERTa model beats the BiLSTM combined model on every evaluation set, most strongly on
CYNER (0.37 → 0.67) and ATTACKER (0.33 → 0.51). However, unlike the BiLSTM case (where combined
training helped on DNRTI and ATTACKER), the **single-dataset RoBERTa outperforms the combined RoBERTa
on every dataset** (e.g. CYNER 0.67 → 0.73, APTNER 0.49 → 0.59). For a pretrained encoder, pooling the
four datasets under unified labels is a net loss relative to per-dataset training — consistent with
the paper's conclusion that unification noise dominates the benefit of more data.

> **Comparability caveat.** The eval sets match exactly: both experiments use the same union-leakage
> removal (which drops only *training* sentences, leaving dev sets full), so each RoBERTa Combined /
> RoBERTa Original pair is scored on the identical dev tokens, unified gold spans, and `span_f1`.
> The two RoBERTa columns do differ in *training-time label handling*, though: RoBERTa Combined is
> trained natively on the unified label space, whereas RoBERTa Original is trained on each dataset's
> original labels and collapsed to the 4 coarse classes by prob-sum late merge after inference. This
> is an extra confound not present in the paper's BiLSTM columns (both unified-trained), so treat
> RoBERTa Original as a strong single-dataset reference rather than a perfectly clean ablation.

## Full metric breakdown — combined RoBERTa

Strict span-F1 with its precision/recall, plus the unlabelled and loose (partial-overlap, same-label)
span-F1 variants — analogous to the paper's cross-dataset metric tables
(`tab:cross_eval_matrix_precision`, `_recall`, `tab:unlabelled-span-f1-cross-dataset`,
`tab:loose-span-f1-cross-dataset`).

| Eval set | Precision | Recall | Span-F1 | Unlabelled F1 | Loose F1 |
|----------|:---:|:---:|:---:|:---:|:---:|
| Combined | 0.53 | 0.53 | 0.53 | 0.64 | 0.60 |
| DNRTI    | 0.60 | 0.46 | 0.52 | 0.63 | 0.58 |
| ATTACKER | 0.56 | 0.48 | 0.51 | 0.59 | 0.59 |
| APTNER   | 0.41 | 0.60 | 0.49 | 0.59 | 0.57 |
| CYNER    | 0.68 | 0.66 | 0.67 | 0.80 | 0.71 |

CYNER is strongest across every metric. DNRTI and ATTACKER are precision-leaning (precision >
recall), whereas APTNER is the one recall-leaning set (0.60 vs 0.41) — the model over-predicts
entities on APTNER, consistent with the definition-discrepancy trends discussed in the paper's
analysis. The gap between strict and unlabelled F1 (e.g. CYNER 0.67 → 0.80) reflects residual
label-assignment errors on otherwise correctly delimited spans.
