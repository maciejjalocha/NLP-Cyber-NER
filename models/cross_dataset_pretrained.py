"""Cross-dataset evaluation of the Cyber-ThreaD pretrained transformers with *late* merging.

This mirrors the paper's Table 6 cross-dataset matrix (train on rows, dev/valid on columns,
each cell a span-F1 score), but replaces the BiLSTM-trained-on-unified-labels setup with the
20 Cyber-ThreaD Hugging Face models (5 architectures x 4 dataset fine-tunes).

Difference from the paper: the paper trains on *unified* labels (early merge). Here each model
predicts in its training dataset's *original* label scheme, and the predictions are projected into
the unified space *after* inference ("late merge"). The projection is **prob-sum**: the coarse class
is the argmax of the summed softmax probabilities of its fine children (marginalization), not a map
of the top-1 fine class. The fine->coarse grouping comes from the per-tag logic in
``nlp_cyber_ner.dataset`` (``unify_tag_aptner/dnrti/attacker/cyner``). Gold labels are the
already-unified ``valid.unified`` files.

Caveat: the pretrained models saw their dataset's full *train* split, and we cannot apply the
paper's ``remove_leakage`` (there is no retraining). Diagonal (in-domain) cells may therefore be
optimistic if train/valid duplicate sentences exist. Off-diagonal cross-dataset cells - the focus
of this experiment - are unaffected. Inference only; no training happens here.
"""

import argparse
import csv
import gc
import os
from pathlib import Path
import time

from loguru import logger
import mlflow
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from nlp_cyber_ner.config import ARTIFACTS_DIR, MODELS_DIR, PROCESSED_DATA_DIR, load_dotenv
from nlp_cyber_ner.dataset import (
    list_to_conll,
    read_iob2_file,
    unify_tag_aptner,
    unify_tag_attacker,
    unify_tag_cyner,
    unify_tag_dnrti,
)
from nlp_cyber_ner.span_f1 import span_f1

MAX_LEN = 512
BATCH_SIZE = 16

# 5 base architectures, each fine-tuned on the 4 datasets below.
ARCHES = ["SecureBERT", "CyBERT", "SecBERT", "DeBERTa", "RoBERTa"]

# internal dataset keys (== processed/ dir names), in the paper's row/column order.
DATASETS = ["dnrti", "attacker", "APTNer", "cyner"]
DISPLAY = {"dnrti": "DNRTI", "attacker": "ATTACKER", "APTNer": "APTNER", "cyner": "CYNER"}

# our dir key -> Hugging Face repo suffix (casing differs from our dir names).
HF_SUFFIX = {"dnrti": "DNRTI", "attacker": "AttackER", "APTNer": "APTNER", "cyner": "CyNER"}

# per-training-dataset late-merge function (maps one raw model tag -> unified tag).
UNIFY = {
    "APTNer": unify_tag_aptner,
    "dnrti": unify_tag_dnrti,
    "attacker": unify_tag_attacker,
    "cyner": unify_tag_cyner,
}

# the 5 span-F1 metric variants we build a matrix for (keys returned by span_f1).
METRICS = ["slot-f1", "precision", "recall", "ul_slot-f1", "l_slot-f1"]

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def model_id(arch: str, ds: str) -> str:
    """Hugging Face repo id for ``arch`` fine-tuned on ``ds``.

    All four DeBERTa checkpoints are deberta-v3, but only the AttackER one carries the
    ``-v3`` in its repo name.
    """
    if arch == "DeBERTa" and ds == "attacker":
        return "Cyber-ThreaD/DeBERTa-v3-AttackER"
    return f"Cyber-ThreaD/{arch}-{HF_SUFFIX[ds]}"


def build_coarse_projection(id2label: dict, unify) -> tuple[torch.Tensor, list[str]]:
    """Fixed [num_fine x num_coarse] 0/1 matrix mapping each fine class to its coarse (unified)
    class via ``unify``. Drives prob-sum late merge: coarse_probs = softmax(fine_logits) @ proj.
    """
    num_fine = len(id2label)
    coarse_list = sorted({unify(id2label[i]) for i in range(num_fine)})
    coarse_index = {c: j for j, c in enumerate(coarse_list)}
    proj = torch.zeros(num_fine, len(coarse_list))
    for i in range(num_fine):
        proj[i, coarse_index[unify(id2label[i])]] = 1.0
    return proj, coarse_list


@torch.no_grad()
def predict(
    model, tokenizer, id2label, sentences: list[list[str]], proj: torch.Tensor,
    coarse_list: list[str],
) -> tuple[list[list[str]], list[list[str]]]:
    """Run the model over pre-tokenized ``sentences`` with first-subword alignment.

    Late merge is done by **probability marginalization** (prob-sum): the coarse class is the
    argmax of the summed softmax probabilities of its fine children, not a map of the top-1 fine
    class. Returns (raw_tags, unified_tags), one tag per input word:
    - raw_tags: the model's native (fine) argmax label (for inspection / .raw.txt).
    - unified_tags: the prob-sum coarse label.
    Truncated trailing words are padded with "O" so lengths match the input.
    """
    proj = proj.to(device)
    raw_all: list[list[str]] = []
    uni_all: list[list[str]] = []
    for start in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[start : start + BATCH_SIZE]
        enc = tokenizer(
            batch,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        )
        logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
        raw_ids = logits.argmax(-1).cpu().tolist()
        coarse_ids = (torch.softmax(logits, dim=-1) @ proj).argmax(-1).cpu().tolist()

        for i, words in enumerate(batch):
            raw_tags: list[str] = []
            uni_tags: list[str] = []
            prev = None
            for k, wid in enumerate(enc.word_ids(i)):
                if wid is None or wid == prev:  # special token or non-first subword
                    continue
                raw_tags.append(id2label[raw_ids[i][k]])
                uni_tags.append(coarse_list[coarse_ids[i][k]])
                prev = wid
            pad = len(words) - len(raw_tags)
            raw_all.append(raw_tags + ["O"] * pad)
            uni_all.append(uni_tags + ["O"] * pad)
    return raw_all, uni_all


def _safe_mlflow(fn, ctx: str) -> None:
    """Run an mlflow logging closure, turning any failure (e.g. remote tracking server
    unreachable) into a warning so the expensive sweep is never aborted by telemetry."""
    try:
        fn()
    except Exception as e:  # noqa: BLE001 - telemetry must never kill the run
        logger.warning(f"mlflow logging failed for {ctx}: {e}")
        try:
            if mlflow.active_run() is not None:
                mlflow.end_run()
        except Exception:
            pass


def _log_run(run_name: str, params: dict, metrics: dict) -> None:
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)


def free(model) -> None:
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_matrix_csv(path, matrix: dict[str, dict[str, float]]) -> None:
    """Write a train x eval matrix (rows=train dataset, cols=eval dataset) to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["train\\eval"] + [DISPLAY[d] for d in DATASETS])
        for train_ds in DATASETS:
            writer.writerow(
                [DISPLAY[train_ds]] + [f"{matrix[train_ds][eval_ds]:.4f}" for eval_ds in DATASETS]
            )


def matrix_to_markdown(title: str, matrix: dict[str, dict[str, float]]) -> str:
    header = "| train \\ eval | " + " | ".join(DISPLAY[d] for d in DATASETS) + " |"
    sep = "|" + "---|" * (len(DATASETS) + 1)
    lines = [f"### {title}", header, sep]
    for train_ds in DATASETS:
        cells = []
        for eval_ds in DATASETS:
            val = f"{matrix[train_ds][eval_ds]:.2f}"
            cells.append(f"**{val}**" if train_ds == eval_ds else val)
        lines.append(f"| **{DISPLAY[train_ds]}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def read_matrix_csv(path) -> dict[str, dict[str, float]] | None:
    """Read a matrix CSV written by save_matrix_csv back into a nested dict, or None if missing."""
    if not path.exists():
        return None
    inv = {v: k for k, v in DISPLAY.items()}
    matrix: dict[str, dict[str, float]] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)[1:]
        for row in reader:
            train_ds = inv[row[0]]
            matrix[train_ds] = {inv[h]: float(v) for h, v in zip(header, row[1:])}
    return matrix


def write_delta_csv(path, local_m, hub_m) -> None:
    """local - hub, per cell (skips cells missing on either side)."""
    delta = {
        t: {e: local_m[t][e] - hub_m[t][e] for e in DATASETS if e in hub_m.get(t, {})}
        for t in DATASETS
        if t in hub_m
    }
    save_matrix_csv(path, delta)


def local_model_path(models_dir, arch: str, ds: str):
    return models_dir / f"{arch}__{ds}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["hub", "local"], default="hub",
                   help="hub = Cyber-ThreaD checkpoints; local = our leakage-clean retrains")
    p.add_argument("--models-dir", type=str, default=None,
                   help="dir of local checkpoints <arch>__<ds> (default MODELS_DIR/retrained)")
    args = p.parse_args()

    source = args.source
    local = source == "local"
    models_dir = Path(args.models_dir) if args.models_dir else MODELS_DIR / "retrained"
    out_dir_name = "cross_retrained" if local else "cross_pretrained"
    experiment_prefix = "retrained-cross" if local else "pretrained-cross"
    title_suffix = (
        "valid, prob-sum late merge, leakage-clean" if local else "valid, prob-sum late merge"
    )

    load_dotenv()
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri is not None:
        logger.info(f"MLFLOW_TRACKING_URI: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"device: {device} | source: {source}"
                + (f" | models_dir: {models_dir}" if local else ""))

    n_models = len(ARCHES) * len(DATASETS)
    n_cells = n_models * len(DATASETS)
    logger.info(f"plan: {len(ARCHES)} archs x {len(DATASETS)} datasets = {n_models} models, "
                f"{n_cells} eval cells")
    t_start = time.perf_counter()

    # Gold dev data (already in the unified label space), loaded once.
    gold: dict[str, list[tuple[list[str], list[str]]]] = {}
    for ds in DATASETS:
        gold[ds] = read_iob2_file(
            PROCESSED_DATA_DIR / ds / "valid.unified", word_index=0, tag_index=1
        )
        logger.info(f"gold[{ds}]: {len(gold[ds])} sentences")

    out_dir = ARTIFACTS_DIR / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = out_dir / "predictions"  # per-cell CoNLL, same style as cross_dataset_model.py
    preds_dir.mkdir(parents=True, exist_ok=True)
    hub_dir = ARTIFACTS_DIR / "cross_pretrained"  # baseline for DELTA when source=local
    markdown_report: list[str] = []

    model_i = 0  # models finished
    cell_i = 0  # eval cells finished

    for arch_i, arch in enumerate(ARCHES, 1):
        logger.info(f"{'=' * 60}")
        logger.info(f"Architecture {arch_i}/{len(ARCHES)}: {arch}")
        logger.info(f"{'=' * 60}")
        # matrices[metric][train_ds][eval_ds] = value
        matrices = {m: {t: {} for t in DATASETS} for m in METRICS}

        mlflow.set_experiment(f"{experiment_prefix}-{arch}")

        for train_ds in DATASETS:
            if local:
                mid = str(local_model_path(models_dir, arch, train_ds))
                load_kwargs = {}
            else:
                mid = model_id(arch, train_ds)
                load_kwargs = {"token": True}
            logger.info(
                f"[model {model_i + 1}/{n_models}] loading {mid} "
                f"(elapsed {time.perf_counter() - t_start:.0f}s)"
            )
            t_load = time.perf_counter()
            tokenizer = AutoTokenizer.from_pretrained(mid, use_fast=True, **load_kwargs)
            model = AutoModelForTokenClassification.from_pretrained(mid, **load_kwargs)
            model = model.to(device).eval()
            id2label = {int(k): v for k, v in model.config.id2label.items()}
            unify = UNIFY[train_ds]
            # fine->coarse projection for prob-sum late merge (same for all eval sets of this model)
            proj, coarse_list = build_coarse_projection(id2label, unify)
            logger.info(f"  loaded in {time.perf_counter() - t_load:.0f}s")

            for eval_ds in DATASETS:
                sentences = [words for words, _ in gold[eval_ds]]
                gold_tags = [tags for _, tags in gold[eval_ds]]

                t_eval = time.perf_counter()
                raw, pred = predict(model, tokenizer, id2label, sentences, proj, coarse_list)
                assert all(len(g) == len(p) for g, p in zip(gold_tags, pred)), (
                    f"length mismatch for {arch} train={train_ds} eval={eval_ds}"
                )

                # Save per-cell predictions in CoNLL form for manual inspection. Tokens/order
                # match the gold file data/processed/<eval_ds>/valid.unified exactly.
                #   .txt      -> predicted UNIFIED tag (post late-merge; cross_dataset_model.py style)
                #   .raw.txt  -> predicted ORIGINAL tag (pre-merge, the model's native label scheme)
                stem = f"{arch}-train-{train_ds}-eval-{eval_ds}"
                list_to_conll(sentences, pred, preds_dir / f"{stem}.txt")
                list_to_conll(sentences, raw, preds_dir / f"{stem}.raw.txt")

                metrics = span_f1(gold_tags, pred)
                for m in METRICS:
                    matrices[m][train_ds][eval_ds] = metrics[m]

                cell_i += 1
                logger.info(
                    f"  [cell {cell_i}/{n_cells}] {arch} train={train_ds} eval={eval_ds} "
                    f"-> slot-f1={metrics['slot-f1']:.3f} "
                    f"({time.perf_counter() - t_eval:.0f}s)"
                )

                params = {
                    "arch": arch,
                    "model_id": mid,
                    "train_ds": train_ds,
                    "eval_ds": eval_ds,
                    "split": "valid",
                    "merge": "late",
                    "source": source,
                }
                _safe_mlflow(
                    lambda: _log_run(f"train-{train_ds}-eval-{eval_ds}", params, metrics),
                    f"{arch}/{train_ds}->{eval_ds}",
                )

            free(model)
            model_i += 1
            logger.info(
                f"[model {model_i}/{n_models}] done ({arch}-{train_ds}); "
                f"total elapsed {time.perf_counter() - t_start:.0f}s"
            )

        # persist per-architecture matrices (one CSV per metric) locally first,
        # then best-effort upload to mlflow.
        markdown_report.append(f"\n## {arch}\n")
        csv_paths = []
        for m in METRICS:
            csv_path = out_dir / f"{arch}__{m}.csv"
            save_matrix_csv(csv_path, matrices[m])
            csv_paths.append(csv_path)
            md = matrix_to_markdown(f"{arch} - {m} ({title_suffix})", matrices[m])
            markdown_report.append(md + "\n")
            print("\n" + md)

        # When scoring our leakage-clean retrains, also emit delta vs the hub baseline.
        if local:
            for m in METRICS:
                hub_m = read_matrix_csv(hub_dir / f"{arch}__{m}.csv")
                if hub_m is not None:
                    delta_path = out_dir / f"DELTA_{arch}__{m}.csv"
                    write_delta_csv(delta_path, matrices[m], hub_m)
                    csv_paths.append(delta_path)
                else:
                    logger.warning(f"no hub baseline for {arch}/{m}; skipping DELTA")

        # This arch's per-cell prediction files (both unified .txt and raw .raw.txt).
        pred_paths = sorted(preds_dir.glob(f"{arch}-train-*-eval-*.txt"))

        def _log_artifacts(csvs=csv_paths, preds=pred_paths):
            with mlflow.start_run(run_name=f"{arch}-summary"):
                for p in csvs:
                    mlflow.log_artifact(str(p), artifact_path=out_dir_name)
                for p in preds:
                    mlflow.log_artifact(str(p), artifact_path=f"{out_dir_name}/predictions")

        _safe_mlflow(_log_artifacts, f"{arch}-summary artifacts")
        logger.success(f"architecture {arch} complete ({arch_i}/{len(ARCHES)})")

    md_path = out_dir / "REPORT.md"
    md_path.write_text(
        f"# Cross-dataset evaluation - {source} models ({title_suffix})\n"
        + "\n".join(markdown_report),
        encoding="utf-8",
    )
    logger.success(
        f"all done in {time.perf_counter() - t_start:.0f}s; "
        f"report: {md_path}; CSV matrices in {out_dir}"
    )


if __name__ == "__main__":
    main()
