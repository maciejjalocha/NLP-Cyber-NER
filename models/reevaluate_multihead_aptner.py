#!/usr/bin/env python3
"""Rescore the multi-head RoBERTa APTNER prediction with BIO-compatible tags."""

import json
from pathlib import Path

from nlp_cyber_ner.dataset import read_iob2_file
from nlp_cyber_ner.span_f1 import span_f1

ROOT = Path(__file__).resolve().parents[1]
PREDICTION_PATH = ROOT / "data" / "train-multihead-roberta-eval-APTNer.txt"
GOLD_PATH = ROOT / "data" / "interim" / "APTNer" / "APTNERdev.cleaned"
METRICS_PATH = ROOT / "models" / "multihead_aptner_metrics_bio.json"


def bioes_to_bio(tags: list[str]) -> list[str]:
    """Convert APTNER BIOES tags to the BIO representation used by span_f1."""
    converted = []
    for tag in tags:
        if tag == "O" or "-" not in tag:
            converted.append(tag)
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "S":
            prefix = "B"
        elif prefix == "E":
            prefix = "I"
        converted.append(f"{prefix}-{label}")
    return converted


def main() -> None:
    gold_data = read_iob2_file(GOLD_PATH)
    prediction_data = read_iob2_file(PREDICTION_PATH)

    if len(gold_data) != len(prediction_data):
        raise ValueError(
            f"sentence mismatch: {len(gold_data)} gold vs {len(prediction_data)} predictions"
        )

    gold_tags = []
    prediction_tags = []
    for sentence_number, ((gold_tokens, gold), (pred_tokens, prediction)) in enumerate(
        zip(gold_data, prediction_data), start=1
    ):
        if gold_tokens != pred_tokens:
            raise ValueError(f"token mismatch in sentence {sentence_number}")
        gold_tags.append(bioes_to_bio(gold))
        prediction_tags.append(bioes_to_bio(prediction))

    metrics = span_f1(gold_tags, prediction_tags)
    METRICS_PATH.write_text(
        json.dumps(
            {
                "dataset": "APTNER",
                "model": "multi-head RoBERTa",
                "prediction_file": str(PREDICTION_PATH.relative_to(ROOT)),
                "gold_file": str(GOLD_PATH.relative_to(ROOT)),
                "normalization": {"S": "B", "E": "I"},
                "metric": "nlp_cyber_ner.span_f1.span_f1",
                "metrics": metrics,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    print(f"wrote {METRICS_PATH}")


if __name__ == "__main__":
    main()