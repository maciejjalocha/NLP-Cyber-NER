"""CLI entrypoint: fine-tune RoBERTa on the combined Cyber-NER dataset.

Run from the repo root:

    python nlp_cyber_ner/train_combined_roberta.py --bf16 --output-root /scratch/.../retrained

Parses CLI flags and forwards them to nlp_cyber_ner.modeling.combined_hf_ner.run(). The recipe
defaults match train_hf_ner.py (batch 2 / lr 2e-5 / 10 epochs / max-length 512 / seed 42).
"""

import argparse

from nlp_cyber_ner.modeling.combined_hf_ner import run
from nlp_cyber_ner.modeling.train_hf_ner import BASE_MODELS


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--base-model",
        type=str,
        default=BASE_MODELS["RoBERTa"],
        help="HF model id/path, or a key of train_hf_ner.BASE_MODELS (e.g. RoBERTa)",
    )
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=float, default=10.0)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true", help="enable bf16 (set on LUMI/MI250X)")
    p.add_argument("--max-steps", type=int, default=-1, help="cap steps (dry-run); -1 disables")
    p.add_argument("--output-dir", type=str, default=None, help="transient Trainer output dir")
    p.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="parent dir; checkpoint written to <root>/combined-roberta (scratch on LUMI)",
    )
    args = p.parse_args()

    base_model = BASE_MODELS.get(args.base_model, args.base_model)

    run(
        base_model=base_model,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        max_length=args.max_length,
        seed=args.seed,
        bf16=args.bf16,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
