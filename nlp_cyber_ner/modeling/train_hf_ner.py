"""Leakage-clean fine-tuning of one Cyber-ThreaD encoder (one arch x one dataset).

Reproduces the Cyber-ThreaD token-classification checkpoints from their *base* encoders, but on
leakage-cleaned data, so the downstream cross-dataset matrix (see cross_dataset_pretrained.py) is
free of train/valid overlap. Models are trained on each dataset's *original* label scheme; the
unified space is applied *after* inference (late merge) by the eval sweep.

Design choices (see plan): uniform recipe batch 2 / lr 2e-5 / 10 epochs / linear / warmup 0 / seed 42
(the AttackER = Deka et al. recipe, citable). Leakage removal = "union": from each train set we drop
every sentence whose token sequence appears in ANY of the four valid sets, so one model per
(arch, dataset) serves all four eval columns with full, comparable, leakage-free eval sets.

Meant to be driven per-index by a SLURM job array (--index 0..19 over ARCHES x DATASETS), one GCD each.
Runs offline on LUMI (HF_HUB_OFFLINE=1 + pre-staged base models); also runs locally on MPS/CPU.
"""

import argparse
import json
from pathlib import Path
import time

from loguru import logger
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from nlp_cyber_ner.config import INTERIM_DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from nlp_cyber_ner.dataset import read_iob2_file, remove_leakage

# Same order as cross_dataset_pretrained.py so --index lines up across the two scripts.
ARCHES = ["SecureBERT", "CyBERT", "SecBERT", "DeBERTa", "RoBERTa"]
DATASETS = ["dnrti", "attacker", "APTNer", "cyner"]

BASE_MODELS = {
    "SecureBERT": "ehsanaghaei/SecureBERT",
    "CyBERT": "SynamicTechnologies/CYBERT",
    "SecBERT": "jackaduma/SecBERT",
    "DeBERTa": "microsoft/deberta-v3-base",
    "RoBERTa": "FacebookAI/roberta-base",
}

# Original-label train/valid sources (tokens + ORIGINAL tags). read_iob2_file(word_index=0, tag_index=1)
# handles both space- and tab-separated CoNLL.
ORIG_SOURCES = {
    "dnrti": (INTERIM_DATA_DIR / "DNRTI" / "train.cleaned", INTERIM_DATA_DIR / "DNRTI" / "valid.cleaned"),
    "APTNer": (
        INTERIM_DATA_DIR / "APTNer" / "APTNERtrain.cleaned",
        INTERIM_DATA_DIR / "APTNer" / "APTNERdev.cleaned",
    ),
    "attacker": (
        INTERIM_DATA_DIR / "attacker" / "train.cleaned",
        INTERIM_DATA_DIR / "attacker" / "valid.cleaned",
    ),
    "cyner": (RAW_DATA_DIR / "cyner" / "train.txt", RAW_DATA_DIR / "cyner" / "valid.txt"),
}


def resolve_index(index: int) -> tuple[str, str]:
    """Row-major over ARCHES x DATASETS (arch outer, dataset inner)."""
    if not 0 <= index < len(ARCHES) * len(DATASETS):
        raise ValueError(f"--index must be in [0, {len(ARCHES) * len(DATASETS)})")
    return ARCHES[index // len(DATASETS)], DATASETS[index % len(DATASETS)]


def union_valid_tokens() -> set[tuple[str, ...]]:
    """Token sequences of every sentence across all four processed valid.unified files.

    Tokens are identical to the original-label valids, so this is a valid leakage filter regardless
    of label scheme.
    """
    union: set[tuple[str, ...]] = set()
    for ds in DATASETS:
        for toks, _ in read_iob2_file(PROCESSED_DATA_DIR / ds / "valid.unified", tag_index=1):
            union.add(tuple(toks))
    return union


def build_label_maps(datasets: list[tuple[list[str], list[str]]]) -> tuple[list[str], dict]:
    labels = sorted({tag for _, tags in datasets for tag in tags})
    if "O" in labels:  # keep O at index 0 for readability
        labels = ["O"] + [x for x in labels if x != "O"]
    label2id = {label: i for i, label in enumerate(labels)}
    return labels, label2id


def encode(data, tokenizer, label2id, max_length):
    """Tokenize pre-split sentences and align labels to the FIRST subword (rest -100)."""
    examples = []
    for words, tags in data:
        enc = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
        )
        label_ids, prev = [], None
        for wid in enc.word_ids():
            if wid is None:
                label_ids.append(-100)
            elif wid != prev:
                label_ids.append(label2id[tags[wid]])
            else:
                label_ids.append(-100)  # non-first subword
            prev = wid
        enc["labels"] = label_ids
        examples.append({k: enc[k] for k in enc})
    return examples


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokenizer_for_pretokenized_inputs(base_model: str):
    """Load a fast tokenizer compatible with is_split_into_words=True inputs.

    RoBERTa-family fast tokenizers require add_prefix_space=True for pretokenized input.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    cls_name = tokenizer.__class__.__name__.lower()
    is_roberta_family = "roberta" in cls_name
    if is_roberta_family and not getattr(tokenizer, "add_prefix_space", False):
        logger.info("reloading tokenizer with add_prefix_space=True for pretokenized inputs")
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, add_prefix_space=True)
    return tokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--index", type=int, help="0..19 over ARCHES x DATASETS")
    g.add_argument("--arch", choices=ARCHES)
    p.add_argument("--dataset", choices=DATASETS)
    p.add_argument("--output-dir", type=str, default=None, help="explicit output dir")
    p.add_argument("--output-root", type=str, default=None,
                   help="parent dir; checkpoint saved to <root>/<arch>__<ds> (for the SLURM array)")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=float, default=10.0)
    p.add_argument("--max-steps", type=int, default=-1, help="cap steps (dry-run); -1 disables")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true", help="enable bf16 (set on LUMI/MI250X)")
    args = p.parse_args()

    if args.index is not None:
        arch, ds = resolve_index(args.index)
    else:
        if args.dataset is None:
            p.error("--dataset is required with --arch")
        arch, ds = args.arch, args.dataset

    base_model = BASE_MODELS[arch]
    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif args.output_root:
        out_dir = Path(args.output_root) / f"{arch}__{ds}"
    else:
        out_dir = MODELS_DIR / "retrained" / f"{arch}__{ds}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"=== fine-tune {arch} on {ds} from {base_model} -> {out_dir} ===")

    train_path, valid_path = ORIG_SOURCES[ds]
    train_data = read_iob2_file(train_path, word_index=0, tag_index=1)
    valid_data = read_iob2_file(valid_path, word_index=0, tag_index=1)
    logger.info(f"loaded train={len(train_data)} valid={len(valid_data)} sentences")

    # Union-leakage removal: drop train sentences whose tokens appear in ANY valid set.
    union = union_valid_tokens()
    union_pseudo = [(list(t), []) for t in union]  # remove_leakage keys on tokens only
    n_before = len(train_data)
    train_data, removed = remove_leakage(train_data, union_pseudo)
    logger.info(f"leakage removal: {n_before} -> {len(train_data)} (removed {len(removed)})")
    train_tokens = {tuple(t) for t, _ in train_data}
    assert train_tokens.isdisjoint(union), "leakage remains after removal!"

    # Label space from the ORIGINAL labels (train ∪ valid so eval-only tags aren't unknown).
    labels, label2id = build_label_maps(train_data + valid_data)
    id2label = {i: label for label, i in label2id.items()}
    logger.info(f"{len(labels)} original labels: {labels}")

    tokenizer = load_tokenizer_for_pretokenized_inputs(base_model)
    model = AutoModelForTokenClassification.from_pretrained(
        base_model, num_labels=len(labels), id2label=id2label, label2id=label2id
    )

    train_ds = ListDataset(encode(train_data, tokenizer, label2id, args.max_length))
    collator = DataCollatorForTokenClassification(tokenizer)

    targs = TrainingArguments(
        output_dir=str(out_dir / "hf_trainer"),
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        lr_scheduler_type="linear",
        warmup_ratio=0.0,
        weight_decay=0.0,
        optim="adamw_torch",
        seed=args.seed,
        bf16=args.bf16,
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

    logger.info(f"device={pick_device()} | training {len(train_ds)} examples, {args.epochs} epochs")
    t0 = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - t0

    trainer.save_model(str(out_dir))  # config (incl. id2label) + weights
    tokenizer.save_pretrained(str(out_dir))

    metrics = {
        "arch": arch,
        "dataset": ds,
        "base_model": base_model,
        "train_sentences": len(train_data),
        "removed_leaked": len(removed),
        "num_labels": len(labels),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "train_runtime_s": result.metrics.get("train_runtime", elapsed),
        "wall_s": elapsed,
    }
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.success(
        f"done {arch}/{ds}: {len(train_data)} sents, {elapsed:.0f}s wall, saved to {out_dir}"
    )


if __name__ == "__main__":
    main()
