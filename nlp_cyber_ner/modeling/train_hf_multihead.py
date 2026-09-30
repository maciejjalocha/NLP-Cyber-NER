"""Multi-head RoBERTa fine-tuning: shared encoder, one token-classification head per dataset.

The transformer counterpart of the four BiLSTM ``train_tokenmodel_*`` scripts (paper.tex §"Multi-head
Model", Table ``tab:multi_head_matrix``). Instead of a from-scratch per-dataset embedding + BiLSTM
stack, we fine-tune a SINGLE shared RoBERTa-base encoder with one linear head per dataset, each over
that dataset's ORIGINAL label set. This keeps each dataset's label specificity (no unification) while
letting all datasets share representations.

Fine-tuning recipe follows ``train_hf_ner.py`` (lr 2e-5 / linear / warmup 0 / seed 42, batch 2 by
default). Cross-dataset batching reproduces the original authors' scheme from ``train_tokenmodel_*``:
per-epoch dataset-level multinomial sampling WITH REPLACEMENT, proportional to each dataset's batch
count, pulled through ``itertools.cycle``. This means a larger dataset's head receives proportionally
more updates (size-proportional exposure) — matching how the BiLSTM multi-head and the combined model
are trained. Batches are homogeneous (each batch is a single dataset → one head per forward pass).

A cleaner alternative (train every head over the same number of batches) would balance the heads
better, but is intentionally NOT used here so results stay comparable to the already-run experiments.

Leakage removal = "union" (same as train_hf_ner): drop every train sentence whose tokens appear in ANY
of the four valid sets. Evaluation is per-dataset span-F1 on that dataset's original-label dev set.
"""

import argparse
from itertools import cycle
import json
import os
from pathlib import Path
import random
import time

from loguru import logger
import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    DataCollatorForTokenClassification,
    get_linear_schedule_with_warmup,
)

from nlp_cyber_ner.config import MODELS_DIR, load_dotenv
from nlp_cyber_ner.dataset import list_to_conll, read_iob2_file, remove_leakage
from nlp_cyber_ner.modeling.train_hf_ner import (
    BASE_MODELS,
    ORIG_SOURCES,
    ListDataset,
    build_label_maps,
    encode,
    load_tokenizer_for_pretokenized_inputs,
    pick_device,
    union_valid_tokens,
)
from nlp_cyber_ner.span_f1 import span_f1

# Head/eval order. Keys double as nn.ModuleDict head names (kept identical to train_hf_ner.DATASETS).
DATASETS = ["dnrti", "attacker", "APTNer", "cyner"]


class MultiHeadRoBERTa(nn.Module):
    """Shared RoBERTa encoder + one linear token-classification head per dataset."""

    def __init__(self, base_model: str, dataset_label_sizes: dict[str, int], dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict(
            {name: nn.Linear(hidden, n) for name, n in dataset_label_sizes.items()}
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, dataset_name, labels=None):
        if dataset_name not in self.heads:
            raise ValueError(
                f"Dataset '{dataset_name}' not recognized. Heads: {list(self.heads.keys())}"
            )
        sequence = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        logits = self.heads[dataset_name](self.dropout(sequence))  # (B, T, n_labels_ds)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return logits, loss


def get_batch_sampling_probs(train_loaders: dict) -> tuple[int, list[float]]:
    """Total batches and per-dataset sampling probability = len(loader)/total (mirror token models)."""
    total_batches = sum(len(loader) for loader in train_loaders.values())
    datasets = list(train_loaders.keys())
    probs = []
    logger.info(f"total batches {total_batches}")
    for name in datasets:
        logger.info(f"{name} individual batches: {len(train_loaders[name])}")
        probs.append(len(train_loaders[name]) / total_batches)
    logger.info(f"dataset sampling probabilities: {probs}")
    return total_batches, probs


def build_dataset_pack(tokenizer, max_length: int):
    """Per dataset: leakage-clean original-label train + valid, encode train, build label maps."""
    union = union_valid_tokens()
    union_pseudo = [(list(t), []) for t in union]  # remove_leakage keys on tokens only

    train_loaders_src, label_maps, dev_packs = {}, {}, {}
    for name in DATASETS:
        train_path, valid_path = ORIG_SOURCES[name]
        train_data = read_iob2_file(train_path, word_index=0, tag_index=1)
        valid_data = read_iob2_file(valid_path, word_index=0, tag_index=1)

        n_before = len(train_data)
        train_data, removed = remove_leakage(train_data, union_pseudo)
        assert {tuple(t) for t, _ in train_data}.isdisjoint(union), "leakage remains!"
        logger.info(
            f"{name}: leakage {n_before} -> {len(train_data)} (removed {len(removed)}), "
            f"valid={len(valid_data)}"
        )

        # Original-label space (train ∪ valid so eval-only tags aren't unknown).
        labels, label2id = build_label_maps(train_data + valid_data)
        id2label = {i: label for label, i in label2id.items()}
        label_maps[name] = {"labels": labels, "label2id": label2id, "id2label": id2label}

        encoded = encode(train_data, tokenizer, label2id, max_length)
        train_loaders_src[name] = ListDataset(encoded)
        dev_packs[name] = valid_data
        logger.info(f"{name}: {len(labels)} original labels")
    return train_loaders_src, label_maps, dev_packs


def train(model, train_loaders, total_batches, probs, epochs, device, lr, max_grad_norm,
          max_batches, bf16=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = int(total_batches * epochs) if max_batches < 0 else max_batches * int(epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
    datasets = list(train_loaders.keys())
    autocast = torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=bf16)

    for epoch in range(int(epochs)):
        model.train()
        logger.info(f"Epoch {epoch + 1}/{int(epochs)}")
        loader_iters = {name: cycle(loader) for name, loader in train_loaders.items()}
        sampled = np.random.choice(datasets, size=total_batches, p=probs)  # with replacement

        n_steps = total_batches if max_batches < 0 else min(max_batches, total_batches)
        losses = {name: [0.0, 0] for name in datasets}  # sum, count
        for i in range(n_steps):
            name = sampled[i]
            batch = next(loader_iters[name])
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with autocast:
                _, loss = model(input_ids, attention_mask, name, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            losses[name][0] += loss.item()
            losses[name][1] += 1
            if (i + 1) % 100 == 0:
                report = {n: round(s / c, 3) for n, (s, c) in losses.items() if c}
                logger.info(f"  step {i + 1}/{n_steps} avg loss/head: {report}")
        for name, (s, c) in losses.items():
            if c:
                logger.info(f"Epoch {epoch + 1} avg {name} loss: {s / c:.3f} ({c} batches)")
    return model


@torch.no_grad()
def predict_word_tags(model, tokenizer, data, dataset_name, id2label, max_length, device):
    """First-subword decode to word-level tags (mirrors combined_hf_ner.predict_word_tags)."""
    model.eval()
    all_tags = []
    for words, _ in data:
        enc = tokenizer(
            words, is_split_into_words=True, truncation=True, max_length=max_length,
            return_tensors="pt",
        )
        word_ids = enc.word_ids()
        logits, _ = model(
            enc["input_ids"].to(device), enc["attention_mask"].to(device), dataset_name
        )
        pred_ids = logits[0].argmax(dim=-1).cpu().tolist()
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="RoBERTa", choices=list(BASE_MODELS.keys()))
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=float, default=10.0)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-grad-norm", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-batches", type=int, default=-1, help="cap batches/epoch (smoke test); -1 off")
    p.add_argument("--bf16", action="store_true", help="enable bf16 autocast (set on LUMI/MI250X)")
    p.add_argument("--output-dir", type=str, default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    base_model = BASE_MODELS[args.arch]
    out_dir = Path(args.output_dir) if args.output_dir else MODELS_DIR / "multihead_roberta"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    logger.info(f"=== multi-head {args.arch} ({base_model}) -> {out_dir} | device={device} ===")

    tokenizer = load_tokenizer_for_pretokenized_inputs(base_model)
    train_sets, label_maps, dev_packs = build_dataset_pack(tokenizer, args.max_length)

    collator = DataCollatorForTokenClassification(tokenizer)
    train_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
        for name, ds in train_sets.items()
    }
    total_batches, probs = get_batch_sampling_probs(train_loaders)

    label_sizes = {name: len(label_maps[name]["labels"]) for name in DATASETS}
    model = MultiHeadRoBERTa(base_model, label_sizes).to(device)

    t0 = time.perf_counter()
    train(model, train_loaders, total_batches, probs, args.epochs, device,
          args.lr, args.max_grad_norm, args.max_batches, args.bf16)
    elapsed = time.perf_counter() - t0
    logger.success(f"training done in {elapsed:.0f}s wall")

    # Save encoder + heads (to scratch via --output-dir) + per-dataset label maps.
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "label_maps.json").write_text(
        json.dumps({n: label_maps[n]["labels"] for n in DATASETS}, indent=2), encoding="utf-8"
    )

    # Timing/config for lumi/gpu_hours_multihead.sh (sacct reads zero GCD-hours on LUMI).
    train_metrics = {
        "base_model": base_model,
        "train_sentences": sum(len(ds) for ds in train_sets.values()),
        "num_datasets": len(DATASETS),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "train_runtime_s": elapsed,
        "wall_s": elapsed,
    }
    (out_dir / "train_metrics.json").write_text(json.dumps(train_metrics, indent=2), encoding="utf-8")

    # MLflow config (mirror combined_hf_ner.py).
    load_dotenv()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri is not None:
        logger.info(f"MLFLOW_TRACKING_URI: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)

    hyperparams = {
        "base_model": base_model,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "seed": args.seed,
        "sampling": "with_replacement_size_proportional",
    }
    preds_dir = MODELS_DIR / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    # Per-dataset dev span-F1 (original labels), one MLflow run each + logged prediction CoNLL.
    results = {}
    for name in DATASETS:
        run_name = f"train-multihead-roberta-eval-{name}"
        gold = [tags for _, tags in dev_packs[name]]
        dev_tokens = [toks for toks, _ in dev_packs[name]]
        preds = predict_word_tags(
            model, tokenizer, dev_packs[name], name, label_maps[name]["id2label"],
            args.max_length, device,
        )
        metrics = span_f1(gold, preds)
        results[name] = metrics
        logger.success(f"{name} dev span-F1: {metrics}")

        store_preds_path = preds_dir / f"{run_name}.txt"
        list_to_conll(dev_tokens, preds, store_preds_path)

        mlflow.set_experiment(run_name)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(hyperparams)
            mlflow.log_params({"dataset": name, "dev_size": len(dev_packs[name]),
                               "n_labels": len(label_maps[name]["labels"])})
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(store_preds_path), artifact_path="predictions")

    (out_dir / "dev_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(f"wrote {out_dir / 'dev_metrics.json'}")


if __name__ == "__main__":
    main()
