"""Fine-tune a RoBERTa encoder on the COMBINED Cyber-NER dataset (unified label space).

This is the transformer counterpart of ``combined_dataset_model.py``: instead of training a BiLSTM
tagger from scratch, we fine-tune a pretrained encoder (RoBERTa by default) on the union of the four
datasets (cyner + APTNer + attacker + DNRTI) in the *unified* 9-tag BIO space, then evaluate the one
model on all five dev packs (each dataset + combined).

The fine-tuning recipe is exactly the one from ``train_hf_ner.py`` (HF ``Trainer``, batch 2 / lr 2e-5
/ 10 epochs / linear / warmup 0 / seed 42 / max-length 512 / optional bf16), and the encoding/tokenizer
helpers are imported from there to avoid duplication. Leakage removal is "union": drop every train
sentence whose tokens appear in ANY of the four valid sets, so the single trained model serves all
five eval columns with leakage-free dev sets.

MLflow tracking + prediction logging mirror ``combined_dataset_model.py``. The fine-tuned model is
NOT persisted: it lives only in the Trainer's transient ``output_dir`` (on LUMI a scratch folder) long
enough to run inference for predictions, then is discarded.
"""

import json
import os
from pathlib import Path
import time

from loguru import logger
import mlflow
import torch
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from nlp_cyber_ner.config import MODELS_DIR, PROCESSED_DATA_DIR, load_dotenv
from nlp_cyber_ner.dataset import list_to_conll, read_iob2_file, remove_leakage
from nlp_cyber_ner.modeling.train_hf_ner import (
    BASE_MODELS,
    ListDataset,
    build_label_maps,
    encode,
    load_tokenizer_for_pretokenized_inputs,
    pick_device,
)
from nlp_cyber_ner.span_f1 import span_f1

# Unified label space guard (same as combined_dataset_model.py).
END_LABELS = {
    "B-Organization",
    "O",
    "I-Malware",
    "B-System",
    "I-Vulnerability",
    "I-Organization",
    "I-System",
    "B-Vulnerability",
    "B-Malware",
}


def load_datasets() -> tuple[
    list[tuple[list[str], list[str]]],
    list[tuple[str, list[tuple[list[str], list[str]]]]],
]:
    """Load the four unified datasets. Returns (combined_train, dev_packs).

    dev_packs is a list of (name, dev_data), including a synthetic "combined" pack.
    Mirrors combined_dataset_model.py:40-84.
    """
    # (display name, on-disk dir): dir case matters on LUMI's case-sensitive FS.
    packs = []
    for name, dir_name in (
        ("cyner", "cyner"),
        ("aptner", "APTNer"),
        ("attacker", "attacker"),
        ("dnrti", "dnrti"),
    ):
        ds_path = PROCESSED_DATA_DIR / dir_name
        train_data = read_iob2_file(ds_path / "train.unified", word_index=0, tag_index=1)
        dev_data = read_iob2_file(ds_path / "valid.unified", word_index=0, tag_index=1)
        a = {tag for _, tags in train_data for tag in tags}
        b = {tag for _, tags in dev_data for tag in tags}
        assert a == b == END_LABELS, f"{name}: labels differ from the unified space ({a} / {b})"
        logger.info(f"{name} loaded: train={len(train_data)} valid={len(dev_data)}")
        packs.append((name, train_data, dev_data))

    combined_train = [s for _, tr, _ in packs for s in tr]
    combined_dev = [s for _, _, dv in packs for s in dv]

    dev_packs = [(name, dv) for name, _, dv in packs]
    dev_packs.append(("combined", combined_dev))
    return combined_train, dev_packs


def union_valid_tokens(
    dev_packs: list[tuple[str, list[tuple[list[str], list[str]]]]],
) -> set[tuple[str, ...]]:
    """Token sequences of every sentence across the four (non-combined) valid sets."""
    union: set[tuple[str, ...]] = set()
    for name, dev_data in dev_packs:
        if name == "combined":
            continue
        for toks, _ in dev_data:
            union.add(tuple(toks))
    return union


def predict_word_tags(
    model,
    tokenizer,
    data: list[tuple[list[str], list[str]]],
    id2label: dict[int, str],
    max_length: int,
    device: str,
) -> list[list[str]]:
    """Word-level predicted tags aligned to the input tokens (first-subword decode).

    For each sentence, tokenizes with is_split_into_words=True, runs the model, argmaxes logits, and
    keeps only the FIRST subword of each word via word_ids(). Missing/truncated words default to "O",
    so the returned per-sentence list always matches len(words) — required by span_f1 and
    list_to_conll.
    """
    model.eval()
    all_tags: list[list[str]] = []
    with torch.no_grad():
        for words, _ in data:
            enc = tokenizer(
                words,
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            word_ids = enc.word_ids()
            inputs = {k: v.to(device) for k, v in enc.items()}
            logits = model(**inputs).logits[0]  # (seq_len, num_labels)
            pred_ids = logits.argmax(dim=-1).cpu().tolist()

            tags = ["O"] * len(words)
            prev = None
            for pos, wid in enumerate(word_ids):
                if wid is None or wid == prev:
                    prev = wid
                    continue
                if wid < len(tags):
                    tags[wid] = id2label[pred_ids[pos]]
                prev = wid
            all_tags.append(tags)
    return all_tags


def run(
    base_model: str = BASE_MODELS["RoBERTa"],
    batch_size: int = 2,
    lr: float = 2e-5,
    epochs: float = 10.0,
    max_length: int = 512,
    seed: int = 42,
    bf16: bool = False,
    max_steps: int = -1,
    output_dir: str | None = None,
    output_root: str | None = None,
) -> None:
    # Resolve the transient Trainer output dir (mirror train_hf_ner.py:154-180).
    if output_dir:
        out_dir = Path(output_dir)
    elif output_root:
        out_dir = Path(output_root) / "combined-roberta"
    else:
        out_dir = MODELS_DIR / "retrained" / "combined-roberta"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"=== fine-tune {base_model} on COMBINED dataset -> {out_dir} (transient) ===")

    combined_train, dev_packs = load_datasets()

    # Union-leakage removal, once: drop train sentences whose tokens appear in ANY valid set.
    union = union_valid_tokens(dev_packs)
    union_pseudo = [(list(t), []) for t in union]  # remove_leakage keys on tokens only
    n_before = len(combined_train)
    train_data, removed = remove_leakage(combined_train, union_pseudo)
    logger.info(f"leakage removal: {n_before} -> {len(train_data)} (removed {len(removed)})")
    train_tokens = {tuple(t) for t, _ in train_data}
    assert train_tokens.isdisjoint(union), "leakage remains after removal!"

    # Unified label space (train ∪ all dev so eval-only tags aren't unknown).
    all_dev = [s for _, dv in dev_packs for s in dv]
    labels, label2id = build_label_maps(train_data + all_dev)
    id2label = {i: label for label, i in label2id.items()}
    logger.info(f"{len(labels)} unified labels: {labels}")

    tokenizer = load_tokenizer_for_pretokenized_inputs(base_model)
    model = AutoModelForTokenClassification.from_pretrained(
        base_model, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    train_ds = ListDataset(encode(train_data, tokenizer, label2id, max_length))
    collator = DataCollatorForTokenClassification(tokenizer)

    targs = TrainingArguments(
        output_dir=str(out_dir / "hf_trainer"),
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=epochs,
        max_steps=max_steps,
        lr_scheduler_type="linear",
        warmup_ratio=0.0,
        weight_decay=0.0,
        optim="adamw_torch",
        seed=seed,
        bf16=bf16,
        logging_steps=100,
        save_strategy="no",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        data_collator=collator,
    )

    device = pick_device()
    logger.info(f"device={device} | training {len(train_ds)} examples, {epochs} epochs")
    t0 = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - t0
    logger.success(f"training done in {elapsed:.0f}s wall")
    # NOTE: the model is intentionally NOT saved (transient scratch checkpoint only).

    # Persist timing/config so lumi/gpu_hours_combined.sh can sum GPU-hours (sacct reads zero on
    # LUMI). Mirrors train_hf_ner.py's train_metrics.json (esp. the "train_runtime_s" key).
    train_metrics = {
        "base_model": base_model,
        "train_sentences": len(train_data),
        "removed_leaked": len(removed),
        "num_labels": len(labels),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "train_runtime_s": result.metrics.get("train_runtime", elapsed),
        "wall_s": elapsed,
    }
    (out_dir / "train_metrics.json").write_text(json.dumps(train_metrics, indent=2), encoding="utf-8")
    logger.info(f"wrote {out_dir / 'train_metrics.json'}")

    model.to(device)

    # MLflow config (mirror combined_dataset_model.py:248-252).
    load_dotenv()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri is not None:
        logger.info(f"MLFLOW_TRACKING_URI: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)

    hyperparams = {
        "base_model": base_model,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "max_length": max_length,
        "seed": seed,
        "bf16": bf16,
        "removed_leaked": len(removed),
        "train_size": len(train_data),
    }

    preds_dir = MODELS_DIR / "predictions"
    train_dir = MODELS_DIR / "train"
    preds_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    # One (train) CoNLL for the shared training set, logged per run.
    train_tokens_list = [toks for toks, _ in train_data]
    train_tags_list = [tags for _, tags in train_data]

    for dev_pack_name, dev_data in dev_packs:
        name = f"train-combined-roberta-eval-{dev_pack_name}"
        logger.info(f"evaluating {name}")

        dev_tokens = [toks for toks, _ in dev_data]
        dev_gold = [tags for _, tags in dev_data]
        dev_preds = predict_word_tags(model, tokenizer, dev_data, id2label, max_length, device)

        mlflow.set_experiment(name)
        with mlflow.start_run(run_name=name):
            metrics = span_f1(dev_gold, dev_preds)

            store_preds_path = preds_dir / f"{name}.txt"
            store_trains_path = train_dir / f"{name}.txt"
            list_to_conll(dev_tokens, dev_preds, store_preds_path)
            list_to_conll(train_tokens_list, train_tags_list, store_trains_path)

            mlflow.log_params(hyperparams)
            mlflow.log_params({"dev_size": len(dev_data)})
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(store_preds_path), artifact_path="predictions")
            mlflow.log_artifact(str(store_trains_path), artifact_path="train")
