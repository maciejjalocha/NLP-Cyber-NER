"""Generate a self-contained HTML to inspect cross-dataset NER prediction cases.

For one prediction cell ``{arch}-train-{TRAIN}-eval-{EVAL}`` this bakes four
token-aligned label columns into a single static HTML page (no runtime file
loading, unlike ``nlp_cyber_ner/lookup2.html``):

    Token | Pred unified | Pred fine (train space) | GT unified | GT fine (eval space)

Sources (all token-aligned on the *valid* split; the sweep asserts this):
    - Pred unified (coarse)        -> ``STEM.txt``           (what was actually scored)
    - Pred fine (train ds space)   -> ``STEM.raw.txt``       (model's native argmax)
    - GT unified (coarse)          -> ``data/processed/<EVAL>/valid.unified``  (eval gold)
    - GT fine (eval ds space)      -> eval dataset's original fine valid file
                                      (interim/.cleaned, or raw/cyner/valid.txt)

Only the two *unified* columns are directly comparable (same label space);
mismatches between them are highlighted. The fine columns live in different
label spaces (train vs eval) and are shown for inspection only. The GT-fine
column was never part of the evaluation (scoring was unified-only).

Usage:
    python -m nlp_cyber_ner.modeling.viz_predictions \
        --pred artifacts/cross_retrained/cross_retrained/predictions/DeBERTa-train-cyner-eval-dnrti.txt
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from loguru import logger

from nlp_cyber_ner.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from nlp_cyber_ner.dataset import read_iob2_file

# eval-dataset -> fine (original label space) valid file. cyner has no interim
# .cleaned; its fine labels (incl. Indicator) live only in the raw CoNLL file.
GT_FINE_VALID = {
    "dnrti": INTERIM_DATA_DIR / "DNRTI" / "valid.cleaned",
    "attacker": INTERIM_DATA_DIR / "attacker" / "valid.cleaned",
    "APTNer": INTERIM_DATA_DIR / "APTNer" / "APTNERdev.cleaned",
    "cyner": RAW_DATA_DIR / "cyner" / "valid.txt",
}

STEM_RE = re.compile(r"^(?P<arch>.+)-train-(?P<train>.+)-eval-(?P<eval>.+)$")

# Entity type -> color. The unified space has 4 types; fine spaces add more, so
# unknown types fall back to a neutral tint via _type_color().
_TYPE_PALETTE = {
    "Malware": "#e57373",
    "Organization": "#64b5f6",
    "System": "#81c784",
    "Vulnerability": "#ba68c8",
    "Indicator": "#ffb74d",
}
_FALLBACK_COLORS = [
    "#4db6ac", "#ff8a65", "#a1887f", "#90a4ae", "#7986cb", "#f06292",
    "#9ccc65", "#4fc3f7", "#dce775", "#b0bec5", "#ce93d8", "#80cbc4",
]


def _entity_type(label: str) -> str:
    """'B-Malware' -> 'Malware'; 'O' -> ''."""
    if not label or label == "O":
        return ""
    return label.split("-", 1)[1] if "-" in label else label


def _build_type_colors(types: set[str]) -> dict[str, str]:
    colors = dict(_TYPE_PALETTE)
    extras = sorted(t for t in types if t and t not in colors)
    for i, t in enumerate(extras):
        colors[t] = _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)]
    return colors


def _labels_present(sents: list[tuple[list[str], list[str]]]) -> list[str]:
    seen: set[str] = set()
    for _, tags in sents:
        seen.update(tags)
    return sorted(seen)


def parse_stem(stem: str) -> tuple[str, str, str]:
    m = STEM_RE.match(stem)
    if not m:
        raise ValueError(
            f"could not parse '{stem}'; expected '<arch>-train-<TRAIN>-eval-<EVAL>'"
        )
    return m["arch"], m["train"], m["eval"]


def load_aligned(
    pred_path: Path, raw_path: Path, gt_unified_path: Path, gt_fine_path: Path
):
    """Load the four sources and assert token alignment across all of them."""
    sources = {
        "pred (.txt)": read_iob2_file(pred_path),
        "pred fine (.raw.txt)": read_iob2_file(raw_path),
        "gt unified": read_iob2_file(gt_unified_path),
        "gt fine": read_iob2_file(gt_fine_path),
    }
    names = list(sources)
    ref = sources[names[0]]
    for name in names[1:]:
        other = sources[name]
        if len(other) != len(ref):
            raise ValueError(
                f"sentence-count mismatch: '{names[0]}' has {len(ref)} but "
                f"'{name}' has {len(other)} (wrong file passed?)"
            )
    for i in range(len(ref)):
        n = len(ref[i][0])
        for name in names[1:]:
            words_i = sources[name][i][0]
            if len(words_i) != n:
                raise ValueError(
                    f"token-count mismatch at sentence {i}: '{names[0]}' has {n} "
                    f"tokens, '{name}' has {len(words_i)}"
                )
    return sources


def render_html(
    *,
    arch: str,
    train_ds: str,
    eval_ds: str,
    tokens: list[list[str]],
    pred_uni: list[list[str]],
    pred_fine: list[list[str]],
    gt_uni: list[list[str]],
    gt_fine: list[list[str]],
) -> str:
    # Collect entity types across all label columns for a shared color scheme.
    all_labels = set()
    for cols in (pred_uni, pred_fine, gt_uni, gt_fine):
        for row in cols:
            all_labels.update(row)
    type_colors = _build_type_colors({_entity_type(lbl) for lbl in all_labels})

    uni_set = sorted(set(l for row in pred_uni for l in row) | set(l for row in gt_uni for l in row))
    train_fine_set = sorted(set(l for row in pred_fine for l in row))
    eval_fine_set = sorted(set(l for row in gt_fine for l in row))

    def chip(label: str) -> str:
        t = _entity_type(label)
        if not t:
            return f'<span class="chip chip-o">{html.escape(label)}</span>'
        c = type_colors.get(t, "#bbb")
        return f'<span class="chip" style="background:{c}">{html.escape(label)}</span>'

    def cell(label: str) -> str:
        t = _entity_type(label)
        style = f' style="background:{type_colors[t]}22"' if t else ""
        return f'<td class="lbl"{style}>{html.escape(label)}</td>'

    n_sent = len(tokens)
    n_tok = sum(len(t) for t in tokens)

    blocks = []
    n_mismatch_sent = 0
    for i in range(n_sent):
        rows = []
        sent_mismatch = False
        for j in range(len(tokens[i])):
            pu, gf = pred_uni[i][j], gt_fine[i][j]
            gu, pf = gt_uni[i][j], pred_fine[i][j]
            row_mismatch = pu != gu
            if row_mismatch:
                sent_mismatch = True
            rows.append(
                f'<tr class="{"mm" if row_mismatch else ""}">'
                f"<td class=tok>{html.escape(tokens[i][j])}</td>"
                f"{cell(pu)}{cell(pf)}{cell(gu)}{cell(gf)}</tr>"
            )
        if sent_mismatch:
            n_mismatch_sent += 1
        sent_text = html.escape(" ".join(tokens[i]))
        blocks.append(
            f'<div class="sent {"has-mm" if sent_mismatch else ""}" '
            f'data-text="{sent_text.lower()}">'
            f'<div class="shead">#{i + 1}'
            f'{" <span class=badge>mismatch</span>" if sent_mismatch else ""}'
            f'<div class="stext">{sent_text}</div></div>'
            "<table><tr>"
            "<th>Token</th><th>Pred unified</th>"
            f"<th>Pred fine<br><small>train={html.escape(train_ds)}</small></th>"
            "<th>GT unified</th>"
            f"<th>GT fine<br><small>eval={html.escape(eval_ds)}</small></th></tr>"
            + "".join(rows)
            + "</table></div>"
        )

    overview = (
        '<div class="ov">'
        f'<div class="ovcol"><h3>Unified (coarse) — comparable</h3><div>{"".join(chip(l) for l in uni_set)}</div></div>'
        f'<div class="ovcol"><h3>Train fine — {html.escape(train_ds)} (pred .raw.txt)</h3><div>{"".join(chip(l) for l in train_fine_set)}</div></div>'
        f'<div class="ovcol"><h3>Eval fine — {html.escape(eval_ds)} (GT original)</h3><div>{"".join(chip(l) for l in eval_fine_set)}</div></div>'
        "</div>"
    )

    title = f"{arch} · train={train_ds} → eval={eval_ds} · valid split"
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 20px; }}
  h1 {{ font-size: 1.3em; margin: 0 0 4px; }}
  .meta {{ color: #888; margin-bottom: 12px; font-size: .9em; }}
  .note {{ background: #fff8e1; color: #5d4037; border: 1px solid #ffe082;
           padding: 8px 12px; border-radius: 6px; margin-bottom: 16px; font-size: .88em; }}
  @media (prefers-color-scheme: dark) {{ .note {{ background:#3a3320; color:#e0d7b8; border-color:#6b5d2e; }} }}
  .ov {{ display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 16px;
         padding: 12px; border: 1px solid #ccc4; border-radius: 8px; }}
  .ovcol h3 {{ font-size: .82em; margin: 0 0 6px; color: #888; font-weight: 600; }}
  .chip {{ display: inline-block; padding: 2px 8px; margin: 2px; border-radius: 10px;
           font-size: .78em; color: #111; font-family: monospace; }}
  .chip-o {{ background: #ccc4; color: inherit; }}
  .controls {{ position: sticky; top: 0; background: Canvas; padding: 8px 0;
               border-bottom: 1px solid #ccc4; margin-bottom: 12px; z-index: 5; }}
  .controls input[type=text] {{ padding: 4px 8px; font-size: .9em; width: 260px; }}
  .controls label {{ margin-right: 16px; font-size: .9em; }}
  .count {{ color: #888; font-size: .85em; }}
  .sent {{ border: 1px solid #ccc4; border-radius: 6px; padding: 10px; margin-bottom: 18px; }}
  .sent.has-mm {{ border-color: #e5737388; }}
  .shead {{ font-weight: 600; margin-bottom: 6px; font-size: .9em; }}
  .stext {{ font-weight: 400; color: #888; font-size: .9em; margin-top: 2px; }}
  .badge {{ background: #e57373; color: #fff; font-size: .7em; padding: 1px 6px;
            border-radius: 8px; margin-left: 6px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .88em; }}
  th, td {{ border: 1px solid #ccc4; padding: 4px 8px; text-align: left; }}
  th {{ font-weight: 600; font-size: .85em; }}
  td.tok {{ font-weight: 600; }}
  td.lbl {{ font-family: monospace; font-size: .85em; }}
  tr.mm td.tok {{ background: #e5737322; }}
  tr.mm {{ outline: 1px solid #e5737355; }}
</style></head>
<body>
<h1>{html.escape(title)}</h1>
<div class="meta">{n_sent} sentences · {n_tok} tokens · {n_mismatch_sent} sentences with a unified pred≠GT mismatch</div>
<div class="note"><b>Only the two unified columns are comparable.</b> Highlighting marks tokens where
<b>Pred unified ≠ GT unified</b> (this is the space that was scored). The fine columns are in
different label spaces (train vs eval) — shown for inspection only; GT-fine was not used in scoring.</div>
{overview}
<div class="controls">
  <label><input type="checkbox" id="onlyMM"> Show only mismatched sentences</label>
  <input type="text" id="filter" placeholder="filter: substring in sentence…">
  <span class="count" id="count"></span>
</div>
<div id="sents">
{"".join(blocks)}
</div>
<script>
  const sents = Array.from(document.querySelectorAll('.sent'));
  const onlyMM = document.getElementById('onlyMM');
  const filter = document.getElementById('filter');
  const count = document.getElementById('count');
  function apply() {{
    const q = filter.value.trim().toLowerCase();
    const mm = onlyMM.checked;
    let shown = 0;
    for (const s of sents) {{
      const okMM = !mm || s.classList.contains('has-mm');
      const okQ = !q || s.dataset.text.includes(q);
      const vis = okMM && okQ;
      s.style.display = vis ? '' : 'none';
      if (vis) shown++;
    }}
    count.textContent = shown + ' / ' + sents.length + ' shown';
  }}
  onlyMM.addEventListener('change', apply);
  filter.addEventListener('input', apply);
  apply();
</script>
</body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a static HTML to inspect NER prediction cases."
    )
    ap.add_argument("--pred", required=True, type=Path,
                    help="prediction .txt (unified) file; STEM = <arch>-train-<T>-eval-<E>")
    ap.add_argument("--out", type=Path, default=None,
                    help="output HTML path (default: alongside --pred, STEM.html)")
    ap.add_argument("--raw", type=Path, default=None, help="override .raw.txt path")
    ap.add_argument("--gt-unified", type=Path, default=None,
                    help="override GT unified path (default processed/<EVAL>/valid.unified)")
    ap.add_argument("--gt-fine", type=Path, default=None,
                    help="override GT fine (original) path")
    args = ap.parse_args()

    pred_path: Path = args.pred
    if pred_path.suffix != ".txt" or pred_path.name.endswith(".raw.txt"):
        raise SystemExit(f"--pred must be the unified .txt file, got {pred_path.name}")
    stem = pred_path.name[: -len(".txt")]
    arch, train_ds, eval_ds = parse_stem(stem)

    raw_path = args.raw or pred_path.with_name(f"{stem}.raw.txt")
    gt_unified_path = args.gt_unified or (PROCESSED_DATA_DIR / eval_ds / "valid.unified")
    if args.gt_fine:
        gt_fine_path = args.gt_fine
    else:
        if eval_ds not in GT_FINE_VALID:
            raise SystemExit(
                f"no default GT-fine file for eval='{eval_ds}'; pass --gt-fine explicitly"
            )
        gt_fine_path = GT_FINE_VALID[eval_ds]

    for p in (pred_path, raw_path, gt_unified_path, gt_fine_path):
        if not p.exists():
            raise SystemExit(f"missing input file: {p}")

    logger.info(f"cell: {arch} train={train_ds} eval={eval_ds}")
    logger.info(f"  pred unified : {pred_path}")
    logger.info(f"  pred fine    : {raw_path}")
    logger.info(f"  gt unified   : {gt_unified_path}")
    logger.info(f"  gt fine      : {gt_fine_path}")

    sources = load_aligned(pred_path, raw_path, gt_unified_path, gt_fine_path)
    tokens = [w for w, _ in sources["pred (.txt)"]]
    pred_uni = [t for _, t in sources["pred (.txt)"]]
    pred_fine = [t for _, t in sources["pred fine (.raw.txt)"]]
    gt_uni = [t for _, t in sources["gt unified"]]
    gt_fine = [t for _, t in sources["gt fine"]]

    html_str = render_html(
        arch=arch, train_ds=train_ds, eval_ds=eval_ds, tokens=tokens,
        pred_uni=pred_uni, pred_fine=pred_fine, gt_uni=gt_uni, gt_fine=gt_fine,
    )

    out_path = args.out or pred_path.with_suffix(".html")
    out_path.write_text(html_str, encoding="utf-8")
    logger.success(f"wrote {out_path} ({len(tokens)} sentences)")


if __name__ == "__main__":
    main()
