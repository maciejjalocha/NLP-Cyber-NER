"""Compute single-dataset RoBERTa reference metrics from stored predictions."""

from collections import Counter
from collections.abc import Callable
import json
from pathlib import Path
from typing import TypedDict

from nlp_cyber_ner.dataset import (
    read_iob2_file,
    unify_tag_aptner,
    unify_tag_attacker,
    unify_tag_cyner,
    unify_tag_dnrti,
)
from nlp_cyber_ner.span_f1 import span_f1

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "models" / "single_dataset_reference_metrics.json"


class DatasetConfig(TypedDict):
    prediction: Path
    gold: Path
    normalize_bioes: bool
    mapper: Callable[[str], str]


DATASETS: dict[str, DatasetConfig] = {
    "DNRTI": {
        "prediction": ROOT / "models" / "predictions" / "RoBERTa-train-dnrti-eval-dnrti.raw.txt",
        "gold": ROOT / "data" / "interim" / "DNRTI" / "valid.cleaned",
        "normalize_bioes": False,
        "mapper": unify_tag_dnrti,
    },
    "ATTACKER": {
        "prediction": ROOT / "models" / "predictions" / "RoBERTa-train-attacker-eval-attacker.raw.txt",
        "gold": ROOT / "data" / "interim" / "attacker" / "valid.cleaned",
        "normalize_bioes": False,
        "mapper": unify_tag_attacker,
    },
    "APTNER": {
        "prediction": ROOT / "models" / "predictions" / "RoBERTa-train-APTNer-eval-APTNer.raw.txt",
        "gold": ROOT / "data" / "interim" / "APTNer" / "APTNERdev.cleaned",
        "normalize_bioes": True,
        "mapper": unify_tag_aptner,
    },
    "CYNER": {
        "prediction": ROOT / "models" / "predictions" / "RoBERTa-train-cyner-eval-cyner.raw.txt",
        "gold": ROOT / "data" / "raw" / "cyner" / "valid.txt",
        "normalize_bioes": False,
        "mapper": unify_tag_cyner,
    },
}


def bioes_to_bio(tags: list[str]) -> list[str]:
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


def load_aligned_tags(gold_path: Path, prediction_path: Path, normalize_bioes: bool):
    gold_data = read_iob2_file(gold_path)
    prediction_data = read_iob2_file(prediction_path)
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
        if normalize_bioes:
            gold = bioes_to_bio(gold)
            prediction = bioes_to_bio(prediction)
        gold_tags.append(gold)
        prediction_tags.append(prediction)
    return gold_tags, prediction_tags


def dropped_entity_token_ratio(gold_path: Path, mapper) -> dict[str, float | int]:
    """Count original entity tokens mapped to O in one validation set."""
    data = read_iob2_file(gold_path)
    total_entity_tokens = 0
    dropped_entity_tokens = 0
    for _, tags in data:
        for tag in tags:
            if tag == "O":
                continue
            total_entity_tokens += 1
            if mapper(tag) == "O":
                dropped_entity_tokens += 1
    ratio = (
        dropped_entity_tokens / total_entity_tokens
        if total_entity_tokens
        else 0.0
    )
    return {
        "dropped_entity_tokens": dropped_entity_tokens,
        "total_entity_tokens": total_entity_tokens,
        "dropped_entity_token_ratio": ratio,
    }


def label_mapping_breakdown(gold_path: Path, mapper) -> dict[str, dict[str, str | int]]:
    """Summarize each original entity label and its unified destination."""
    counts: dict[str, int] = {}
    for _, tags in read_iob2_file(gold_path):
        for tag in tags:
            if tag == "O" or "-" not in tag:
                continue
            _, label = tag.split("-", 1)
            counts[label] = counts.get(label, 0) + 1

    breakdown = {}
    for label, count in sorted(counts.items()):
        destination = mapper(f"B-{label}")
        destination_label = destination.split("-", 1)[1] if destination != "O" else "O"
        breakdown[label] = {
            "entity_tokens": count,
            "maps_to": destination_label,
            "status": "dropped" if destination == "O" else "bundled/retained",
        }
    return breakdown


def label_confusion_summary(
    gold_tags: list[list[str]], prediction_tags: list[list[str]], mapper
) -> dict:
    """Summarize original-label token confusions for annotated gold tokens."""
    confusion = Counter()
    dropped_gold_counts = Counter()
    dropped_to_o = Counter()
    dropped_to_entity = Counter()

    def core(tag: str) -> str:
        return tag.split("-", 1)[1] if "-" in tag else tag

    for gold_sentence, prediction_sentence in zip(gold_tags, prediction_tags):
        for gold, prediction in zip(gold_sentence, prediction_sentence):
            if gold == "O":
                continue
            gold_label = core(gold)
            prediction_label = core(prediction)
            confusion[(gold_label, prediction_label)] += 1
            if mapper(gold) == "O":
                dropped_gold_counts[gold_label] += 1
                if prediction == "O":
                    dropped_to_o[gold_label] += 1
                else:
                    dropped_to_entity[gold_label] += 1

    incorrect = [
        {"gold": gold, "predicted": predicted, "tokens": count}
        for (gold, predicted), count in confusion.most_common()
        if gold != predicted
    ]
    dropped_summary = {}
    for label, count in dropped_gold_counts.most_common():
        dropped_summary[label] = {
            "gold_entity_tokens": count,
            "predicted_as_o": dropped_to_o[label],
            "predicted_as_entity": dropped_to_entity[label],
            "predicted_as_o_ratio": dropped_to_o[label] / count,
        }
    return {
        "top_original_label_confusions": incorrect[:20],
        "dropped_gold_label_predictions": dropped_summary,
    }


def main() -> None:
    results = {}
    for dataset, paths in DATASETS.items():
        for path in (paths["gold"], paths["prediction"]):
            if not path.exists():
                raise FileNotFoundError(path)
        gold_tags, prediction_tags = load_aligned_tags(
            paths["gold"], paths["prediction"], paths["normalize_bioes"]
        )
        results[dataset] = {
            "gold_file": str(paths["gold"].relative_to(ROOT)),
            "prediction_file": str(paths["prediction"].relative_to(ROOT)),
            "normalization": {"S": "B", "E": "I"} if paths["normalize_bioes"] else None,
            "validation_mapping": dropped_entity_token_ratio(paths["gold"], paths["mapper"]),
            "label_mapping_breakdown": label_mapping_breakdown(paths["gold"], paths["mapper"]),
            "label_confusion": label_confusion_summary(
                gold_tags, prediction_tags, paths["mapper"]
            ),
            "metrics": span_f1(gold_tags, prediction_tags),
        }
    output = {
        "model": "single-dataset RoBERTa",
        "evaluation": "original labels, non-merged",
        "metric": "nlp_cyber_ner.span_f1.span_f1",
        "results": results,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()