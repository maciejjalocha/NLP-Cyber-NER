"""
Paired bootstrap significance test: LST-NER vs. baseline, per dataset.

Why bootstrap instead of a t-test: each (dataset, model) config here has exactly
one training run, so there's no distribution of scores to hand a t-test - it
would have no variance estimate. Instead this uses the standard NLP approach
for comparing two systems on the same test set (Berg-Kirkpatrick et al. 2012;
Koehn 2004): resample the dev set with replacement many times, and for each
resample compute both systems' span-F1 on the exact same resampled sentences.
The p-value is the fraction of resamples where the observed improvement doesn't
hold up.

This requires predictions_dev.json (per-example gold/pred tags) written by the
patched evaluate_model() in both the LST-NER and baseline training scripts, in
each config's output_dir. Run this after both members of a pair have completed
training with the prediction-dumping patch.

Usage:
    python bootstrap_significance.py
"""

import json
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nlp_cyber_ner.span_f1 import toSpans

# (LST-NER output dir, baseline output dir) per dataset
DATASETS = {
    "APTNER":    ("lst_ner_output_roberta_aptner",    "bert_ner_baseline_output_roberta_aptner"),
    "DNRTI":     ("lst_ner_output_roberta_dnrti",     "bert_ner_baseline_output_roberta_dnrti"),
    "CYNER":     ("lst_ner_output_roberta_cyner",     "bert_ner_baseline_output_roberta_cyner"),
    "ATTACKNER": ("lst_ner_output_roberta_attackner", "bert_ner_baseline_output_roberta_attackner"),
}

N_BOOTSTRAP = 10000
SEED = 42
ALPHA = 0.05


def load_predictions(output_dir):
    path = os.path.join(output_dir, "predictions_dev.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["gold"], data["pred"]


def precompute_spans(tags_list):
    return [toSpans(tags) for tags in tags_list]


def f1_from_spans(gold_spans_list, pred_spans_list, indices):
    tp = fp = fn = 0
    for i in indices:
        g = gold_spans_list[i]
        p = pred_spans_list[i]
        overlap = len(g & p)
        tp += overlap
        fp += len(p) - overlap
        fn += len(g) - overlap
    prec = 0.0 if tp + fp == 0 else tp / (tp + fp)
    rec = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if prec + rec == 0.0 else 2 * (prec * rec) / (prec + rec)
    return f1


def paired_bootstrap(gold, pred_a, pred_b, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """
    Paired bootstrap significance test comparing system A (e.g. LST-NER) vs
    system B (e.g. baseline) on the same sentences. Returns
    (f1_a, f1_b, observed_diff, p_value).
    """
    n = len(gold)
    assert len(pred_a) == n and len(pred_b) == n, (
        f"gold/pred length mismatch: gold={n}, pred_a={len(pred_a)}, pred_b={len(pred_b)}"
    )

    gold_spans = precompute_spans(gold)
    a_spans = precompute_spans(pred_a)
    b_spans = precompute_spans(pred_b)

    full_idx = list(range(n))
    f1_a = f1_from_spans(gold_spans, a_spans, full_idx)
    f1_b = f1_from_spans(gold_spans, b_spans, full_idx)
    observed_diff = f1_a - f1_b

    rng = random.Random(seed)
    count = 0
    for _ in range(n_bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]
        diff = f1_from_spans(gold_spans, a_spans, idx) - f1_from_spans(gold_spans, b_spans, idx)
        # Two-sided paired bootstrap test: count resamples where the sign of the
        # difference flips relative to the observed direction.
        if observed_diff >= 0:
            if diff <= 0:
                count += 1
        else:
            if diff >= 0:
                count += 1

    p_value = count / n_bootstrap
    return f1_a, f1_b, observed_diff, p_value


def main():
    print(f"Paired bootstrap significance test (N={N_BOOTSTRAP}, seed={SEED}, alpha={ALPHA})")
    print(f"{'Dataset':<12} {'LST-NER F1':>11} {'Baseline F1':>12} {'Diff':>8} {'p-value':>9}  Significant?")
    print("-" * 82)

    results = {}
    for ds, (lst_dir, base_dir) in DATASETS.items():
        lst_path = os.path.join(lst_dir, "predictions_dev.json")
        base_path = os.path.join(base_dir, "predictions_dev.json")
        if not os.path.exists(lst_path):
            print(f"{ds:<12} SKIPPED - missing {lst_path}")
            continue
        if not os.path.exists(base_path):
            print(f"{ds:<12} SKIPPED - missing {base_path}")
            continue

        gold_lst, pred_lst = load_predictions(lst_dir)
        gold_base, pred_base = load_predictions(base_dir)

        if gold_lst != gold_base:
            print(
                f"{ds:<12} WARNING: gold labels differ between LST-NER ({len(gold_lst)} sentences) "
                f"and baseline ({len(gold_base)} sentences) dev sets - not validly paired, skipping."
            )
            continue

        f1_a, f1_b, diff, p = paired_bootstrap(gold_lst, pred_lst, pred_base)
        sig = "YES" if p < ALPHA else "no"
        results[ds] = {"lst_ner_f1": f1_a, "baseline_f1": f1_b, "diff": diff, "p_value": p, "significant": p < ALPHA}
        print(f"{ds:<12} {f1_a:>11.4f} {f1_b:>12.4f} {diff:>+8.4f} {p:>9.4f}  {sig}")

    print()
    with open("bootstrap_significance_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved full results to bootstrap_significance_results.json")


if __name__ == "__main__":
    main()
