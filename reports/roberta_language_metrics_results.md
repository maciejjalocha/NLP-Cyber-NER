# RoBERTa Jensen-Shannon correlation results

Transformer counterpart of the paper's language-metrics correlation table (`tab:js-correlation`). Pearson correlations were computed over the 12 off-diagonal train-to-development dataset pairs, relating Jensen-Shannon (JS) divergence between dataset training distributions to cross-dataset span-F1.

The dataset order is DNRTI, ATTACKER, APTNER, and CYNER. The JS-divergence matrices are the paper's train-to-train matrices, reused from `notebooks/languagemetrics.ipynb`. The RoBERTa performance matrix is the leakage-clean, probability-sum late-merge cross-dataset span-F1 reported in `reports/cross_dataset_roberta_results.md`.

## Correlation results

| Distribution | BiLSTM (paper) | BiLSTM (recomputed) | RoBERTa | RoBERTa p-value |
|-------------|:--------------:|:-------------------:|:-------:|:---------------:|
| Words | -0.74 | -0.74 | **-0.02** | 0.95 |
| POS labels | -0.71 | -0.71 | **-0.23** | 0.48 |
| Entity span lengths | -0.36 | -0.36 | **0.00** | 0.99 |
| Entity span-based counts | 0.00 | -0.00 | **0.13** | 0.68 |

The BiLSTM recomputation reproduces the paper values, validating the masking and correlation pipeline. For RoBERTa, all correlations are close to zero and none is statistically significant at conventional thresholds. The strongest absolute association is the weak negative correlation for POS labels (`r=-0.23`, `p=0.48`).

## Interpretation

The strong negative word and POS correlations reported for the BiLSTM do not carry over to RoBERTa. Within this experiment, JS divergence therefore does not predict cross-dataset RoBERTa span-F1. This is consistent with the pretrained encoder being more robust to distributional shift than the from-scratch BiLSTM, though the small number of observations (12 off-diagonal pairs) limits statistical power.

## Comparability caveat

Only the performance matrix changes between the two analyses: the JS matrices are identical and model-independent. RoBERTa uses union-leakage removal and probability-sum late merge to the four coarse labels, while the BiLSTM reference was unified-trained. This training and label-handling difference is a confound, so the RoBERTa results should be presented as a transformer counterpart rather than a perfectly controlled ablation.

Source notebook: `notebooks/roberta_language_metrics.ipynb`.
