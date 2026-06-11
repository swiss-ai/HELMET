#!/usr/bin/env python3
"""
HELMET Multi-Context-Length Evaluation Report Generator
========================================================
Combines a long-context outputs folder (one sequence length per model,
e.g. 128 K) and an optional short-context outputs folder (multiple
sequence lengths per model, e.g. 8 K-64 K) into a single self-contained
HTML report that shows performance broken down by context length.

The HTML contains two tables:
  1. Main table  - full breakdown: task group → task → context length x model
  2. Summary table - context-length averages per task group x model
     Cells where not every task in the group was evaluated at that context
     length are flagged as incomplete (amber highlight + trailing asterisk *).

Usage:
    python make_helmet_report_ctx.py \\
        --long  /path/to/HELMET/outputs       \\
        --short /path/to/HELMET/outputs_short \\
        --out   report.html [--sort] \\
        [--models "ModelA,ModelB" | --models @model_list.txt]

    If --models is omitted the ALLOWED_MODELS list defined in this file is used.
    Pass --models ALL to include every model found on disk.
"""

import json
import re
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# Model allow-list
# Set to None (or pass --models ALL) to include every model found on disk.
ALLOWED_MODELS: Optional[List[str]] = [
    # "ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_3600",
    "ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200",
    "ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200-mixed-lr5e-6-beta0.1-bs256-lenNormfalse-maxPL2048-rollout8-images-2453510-2453543",
    "ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200-online-lr5e-6-beta0.1-bs256-lenNormfalse-maxPL2048-rollout8-images-2453762-2453767",
    "Llama-3.1-Nemotron-Nano-VL-8B-V1",
    "Olmo-3-7B-Instruct",
    "Olmo-3-7B-Instruct-SFT",
    # "Olmo-3-7B-Think-SFT",
    # "Qwen3-8B",
    "Qwen3-VL-8B-Instruct",
    "gemma-3-12b-it",
]

# Per-model maximum accepted context length
# Models with a limit below the global maximum will have those evals excluded
# and their cross-context averages flagged as incomplete (*).
DEFAULT_MAX_CTX: int = 131_072  # 128 K - used for any model not listed below

MODEL_MAX_CTX: Dict[str, int] = {
    "Olmo-3-7B-Instruct":     65_536,  # 64 K
    "Olmo-3-7B-Instruct-SFT": 65_536,  # 64 K
}

# Task catalogue
TASKS: Dict[str, tuple] = {
    "kilt_nq":             ("Retrieval-augmented generation", "substring_exact_match", "Natural Questions"),
    "kilt_triviaqa":       ("Retrieval-augmented generation", "substring_exact_match", "TriviaQA"),
    "kilt_popqa":          ("Retrieval-augmented generation", "substring_exact_match", "PopQA"),
    "kilt_hotpotqa":       ("Retrieval-augmented generation", "substring_exact_match", "HotpotQA"),
    "msmarco_rerank_psg":  ("Passage re-ranking",            "NDCG@10",               "MS MARCO"),
    "alce_asqa":           ("Generation with citations",     "str_em",                "ALCE ASQA"),
    "alce_qampari":        ("Generation with citations",     "qampari_rec_top5",      "ALCE Qampari"),
    "infbench_qa_eng":     ("Long-document QA",              "rougeL_f1",             "\u221eBENCH QA"),
    "infbench_choice_eng": ("Long-document QA",              "exact_match",           "\u221eBENCH MC"),
    "narrativeqa":         ("Long-document QA",              "rougeL_f1",             "NarrativeQA"),
    "infbench_sum_eng":    ("Summarization",                 "rougeLsum_f1",          "\u221eBENCH Sum"),
    "multi_lexsum":        ("Summarization",                 "rougeLsum_f1",          "Multi-LexSum"),
    "icl_trec_coarse":     ("Many-shot in-context learning", "exact_match",           "TREC Coarse"),
    "icl_trec_fine":       ("Many-shot in-context learning", "exact_match",           "TREC Fine"),
    "icl_nlu":             ("Many-shot in-context learning", "exact_match",           "NLU"),
    "icl_banking77":       ("Many-shot in-context learning", "exact_match",           "BANKING77"),
    "icl_clinic150":       ("Many-shot in-context learning", "exact_match",           "CLINC150"),
    "json_kv":             ("Synthetic recall",              "substring_exact_match", "JSON KV"),
    "ruler_niah_mk_2":     ("Synthetic recall",              "ruler_recall",          "RULER MK Needle"),
    "ruler_niah_mk_3":     ("Synthetic recall",              "ruler_recall",          "RULER MK UUID"),
    "ruler_niah_mv":       ("Synthetic recall",              "ruler_recall",          "RULER MV"),
}

GROUP_ORDER: List[str] = [
    "Retrieval-augmented generation",
    "Passage re-ranking",
    "Generation with citations",
    "Long-document QA",
    "Summarization",
    "Many-shot in-context learning",
    "Synthetic recall",
    "Other",
]

FALLBACK_METRICS: List[str] = [
    "exact_match", "substring_exact_match", "f1",
    "rougeL_f1", "rougeLsum_f1", "NDCG@10",
    "ruler_recall", "MRR", "rougeL_recall",
]

CTX_LABELS: Dict[int, str] = {
    4_096:    "4K",
    8_192:    "8K",
    16_384:   "16K",
    32_768:   "32K",
    65_536:   "64K",
    131_072:  "128K",
    262_144:  "256K",
    524_288:  "512K",
    1_048_576:"1M",
}

# Context lengths shown as dedicated average rows at the bottom of the main table
CTX_AVG_ROWS: List[int] = [4_096, 8_192, 16_384, 32_768, 65_536, 131_072]


def ctx_label(n: int) -> str:
    return CTX_LABELS.get(n, f"{n // 1024}K" if n >= 1024 else str(n))


# Filename utilities
def to_task_key(filename: str) -> Optional[str]:
    stem = filename
    for ext in (".json.score", ".score"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    prefix     = stem.split("_eval_")[0]
    keys_by_len = sorted(TASKS.keys(), key=len, reverse=True)
    if prefix in TASKS:
        return prefix
    for key in keys_by_len:
        if prefix.startswith(key):
            return key
    prefix2 = re.sub(r"_\d+$", "", prefix)
    if prefix2 == prefix:
        return None
    if prefix2 in TASKS:
        return prefix2
    for key in keys_by_len:
        if prefix2.startswith(key):
            return key
    return None


def extract_ctx_len(filename: str) -> Optional[int]:
    m = re.search(r"_in(\d+)[_.]", filename)
    return int(m.group(1)) if m else None


# Data loading
def load_model_ctx(model_dir: Path) -> Dict[str, Dict[int, dict]]:
    result: Dict[str, Dict[int, dict]] = defaultdict(dict)
    for sf in sorted(model_dir.glob("*.score")):
        ctx = extract_ctx_len(sf.name)
        if ctx is None:
            continue
        key = to_task_key(sf.name)
        if key is None:
            stem  = sf.name.replace(".json.score", "").replace(".score", "")
            raw   = re.sub(r"_\d+$", "", stem.split("_eval_")[0])
            if raw in TASKS:
                key = raw
            else:
                try:
                    data = json.loads(sf.read_text())
                except Exception:
                    continue
                val, pmetric = None, "—"
                for m in FALLBACK_METRICS:
                    if m in data:
                        val, pmetric = data[m], m
                        break
                if val is not None:
                    print(f"Task {raw} not available")
                    # TASKS[raw] = ("Other", pmetric, raw)
                    # if ctx not in result[raw]:
                    #     result[raw][ctx] = {"value": float(val), "metric": pmetric}
                continue
        if ctx in result[key]:
            continue
        try:
            data = json.loads(sf.read_text())
        except Exception:
            continue
        _, pmetric, _ = TASKS[key]
        val = data.get(pmetric)
        if val is None:
            for m in FALLBACK_METRICS:
                if m in data:
                    val, pmetric = data[m], m
                    break
        if val is not None:
            result[key][ctx] = {"value": float(val), "metric": pmetric}
    return dict(result)


def collect_all(
    long_dir: Path,
    short_dir: Optional[Path] = None,
    allowed_models: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    Scan model sub-directories under long_dir (and short_dir) and merge their
    .score files.

    Parameters
    ----------
    allowed_models
        When not None, only directories whose name appears in this list are
        loaded.  Pass None (or an empty list built from --models ALL) to load
        every model found on disk.
    """
    # Build the set of model names present on disk
    model_names: set = set()
    for base in filter(None, [long_dir, short_dir]):
        for d in base.iterdir():
            if d.is_dir() and any(d.glob("*.score")):
                model_names.add(d.name)

    # Apply the allow-list filter
    if allowed_models is not None:
        allowed_set  = set(allowed_models)
        skipped      = model_names - allowed_set
        not_on_disk  = allowed_set - model_names
        if skipped:
            print(
                f"[filter] Skipping {len(skipped)} model(s) not in the allow-list: "
                + ", ".join(sorted(skipped))
            )
        if not_on_disk:
            print(
                f"[filter] Warning: {len(not_on_disk)} allow-listed model(s) not found on disk: "
                + ", ".join(sorted(not_on_disk))
            )
        model_names = model_names & allowed_set

    # Iterate in allow-list order when one is given; fall back to alphabetical.
    if allowed_models is not None:
        ordered_names = [n for n in allowed_models if n in model_names]
    else:
        ordered_names = sorted(model_names)

    models: Dict[str, Dict] = {}
    for name in ordered_names:
        max_ctx = MODEL_MAX_CTX.get(name, DEFAULT_MAX_CTX)
        merged: Dict[str, Dict[int, dict]] = defaultdict(dict)
        for base in filter(None, [long_dir, short_dir]):
            mdir = base / name
            if not mdir.is_dir():
                continue
            for task_key, ctx_scores in load_model_ctx(mdir).items():
                for ctx, score in ctx_scores.items():
                    if ctx <= max_ctx and ctx not in merged[task_key]:
                        merged[task_key][ctx] = score
        if merged:
            models[name] = {k: dict(v) for k, v in merged.items()}
    return models


# Aggregation helpers
def safe_avg(vals) -> Optional[float]:
    vals = [float(v) for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def task_ctx_avg(model_data: dict, task_key: str) -> Optional[float]:
    ctx_scores = model_data.get(task_key, {})
    vals = [s["value"] for s in ctx_scores.values()]
    return safe_avg(vals) if vals else None


def group_ctx_info(
    model_data: dict,
    task_keys: List[str],
    ctx_len: int,
) -> Tuple[Optional[float], int, int]:
    """Return (avg, n_present, n_total) for a task set at one context length."""
    vals = [model_data.get(k, {}).get(ctx_len, {}).get("value") for k in task_keys]
    present = [v for v in vals if v is not None]
    return (safe_avg(present) if present else None), len(present), len(task_keys)


def fmt(v, d: int = 2) -> str:
    return f"{float(v):.{d}f}"


# Cell helpers
def val_cell(v, incomplete: bool = False) -> str:
    """Main-table grid cell.  Pass incomplete=True for cross-context averages
    where the model was not evaluated at every context length."""
    if v is not None:
        inc_cls  = " inc-cell" if incomplete else ""
        inc_attr = ' data-inc="1"' if incomplete else ""
        return f'<div class="cell{inc_cls}" data-val="{fmt(v)}"{inc_attr}></div>'
    return '<div class="cell"><span class="na">&mdash;</span></div>'


def summary_val_cell(
    avg: Optional[float],
    n_present: int,
    n_total: int,
    extra_cls: str = "",
    ctx_cls: str = "",
) -> str:
    """Summary-table <td> cell with optional incomplete annotation."""
    base_cls = f"s-cell{extra_cls}{ctx_cls}"
    if avg is None:
        return f'<td class="{base_cls} s-na"><span class="na">&mdash;</span></td>'
    incomplete = n_present < n_total
    inc_cls    = " s-inc" if incomplete else ""
    tip        = f'title="{n_present}/{n_total} tasks evaluated at this context length"'
    inc_attr   = ' data-inc="1"' if incomplete else ""
    mark       = '<sup class="inc-mark">*</sup>' if incomplete else ""
    val_str    = fmt(avg)
    return (
        f'<td class="{base_cls} s-score-cell{inc_cls}" '
        f'{tip} data-val="{val_str}"{inc_attr}>'
        f'{val_str}{mark}'
        f'</td>'
    )


# Summary table builder
def build_summary_section(
    models: Dict,
    names: List[str],
    all_task_keys: List[str],
    all_ctx_lens: List[int],
    groups: Dict[str, List[str]],
) -> str:
    k = len(all_ctx_lens)

    # - Two-row header -
    hdr1 = '<th rowspan="2" class="s-th-label">Task Group</th>\n'
    for nm in names:
        nm_disp = nm if len(nm) <= 24 else nm[:22] + "\u2026"
        hdr1 += f'<th colspan="{k}" class="s-th-model" title="{nm}">{nm_disp}</th>\n'

    hdr2 = ""
    for nm in names:
        for j, c in enumerate(all_ctx_lens):
            border = " s-col-start" if j == 0 else ""
            lbl    = ctx_label(c)
            hdr2  += f'<th class="s-th-ctx{border} s-ctx-col-{lbl}">{lbl}</th>\n'

    # - Data rows (one per task group) -
    body = ""
    for grp in GROUP_ORDER:
        if grp not in groups:
            continue
        tkeys = groups[grp]
        body += (
            f'<tr class="s-group-row sum-score-row">\n'
            f'  <td class="s-td-label">{grp}'
            f'<span class="s-task-count">\u2009({len(tkeys)})</span></td>\n'
        )
        for nm in names:
            for j, c in enumerate(all_ctx_lens):
                lbl        = ctx_label(c)
                border_cls = " s-col-start" if j == 0 else ""
                ctx_cls    = f" s-ctx-col-{lbl}"
                avg, n_p, n_t = group_ctx_info(models[nm], tkeys, c)
                body += "  " + summary_val_cell(avg, n_p, n_t,
                                                extra_cls=border_cls,
                                                ctx_cls=ctx_cls) + "\n"
        body += "</tr>\n"

    # - Overall row -
    body += (
        '<tr class="s-overall-row sum-score-row">\n'
        '  <td class="s-td-label s-td-overall">Overall Average</td>\n'
    )
    for nm in names:
        for j, c in enumerate(all_ctx_lens):
            lbl        = ctx_label(c)
            border_cls = " s-col-start" if j == 0 else ""
            ctx_cls    = f" s-ctx-col-{lbl}"
            avg, n_p, n_t = group_ctx_info(models[nm], all_task_keys, c)
            body += "  " + summary_val_cell(avg, n_p, n_t,
                                            extra_cls=border_cls,
                                            ctx_cls=ctx_cls) + "\n"
    body += "</tr>\n"

    return f"""
<div class="summary-section">
  <h2 class="s-heading">Summary by Context Length</h2>
  <p class="summary-note">
    Average score per task group at each context length.&nbsp;
    <span class="inc-legend">
      <span class="inc-ex">42.00<sup>*</sup></span>&nbsp;=&nbsp;incomplete average
    </span>
    &mdash; at least one task in the group was not evaluated at that context length,
    <em>or</em> the model has a maximum context length below the global maximum
    (hover a cell for the task count).
    Toggling a context length above also hides the corresponding columns here.
  </p>
  <div class="summary-scroll">
    <div class="summary-wrap">
      <table class="summary-table">
        <thead>
          <tr class="s-hdr-1">{hdr1}</tr>
          <tr class="s-hdr-2">{hdr2}</tr>
        </thead>
        <tbody>{body}</tbody>
      </table>
    </div>
  </div>
</div>"""


# Main HTML builder
def build_html(
    models: Dict,
    long_dir: Path,
    short_dir: Optional[Path],
    sort_by_score: bool = False,
) -> str:

    if sort_by_score:
        def _ovr(nm):
            vals = [s["value"] for t in models[nm].values() for s in t.values()]
            return safe_avg(vals) or -1.0
        names = sorted(models.keys(), key=_ovr, reverse=True)
    else:
        names = list(models.keys())

    n = len(names)

    all_task_keys = sorted(
        {k for m in models.values() for k in m},
        key=lambda k: (
            GROUP_ORDER.index(TASKS[k][0]) if TASKS[k][0] in GROUP_ORDER else 99,
            list(TASKS.keys()).index(k) if k in TASKS else 999,
        ),
    )
    all_ctx_lens = sorted({
        c
        for m in models.values()
        for ctx_scores in m.values()
        for c in ctx_scores
    })

    groups: Dict[str, List[str]] = defaultdict(list)
    for k in all_task_keys:
        groups[TASKS[k][0]].append(k)

    # Which models are limited to fewer context lengths than the global maximum?
    # Their cross-context averages are marked incomplete (*).
    global_max_ctx = max(all_ctx_lens) if all_ctx_lens else DEFAULT_MAX_CTX
    model_incomplete = {
        nm: MODEL_MAX_CTX.get(nm, DEFAULT_MAX_CTX) < global_max_ctx
        for nm in names
    }

    col_template = f"230px repeat({n}, minmax(0, 1fr))"

    hdr = '<div class="cell">Task / Context Length</div>'
    for nm in names:
        n_done  = sum(len(ctxs) for ctxs in models[nm].values())
        n_total = len(all_task_keys) * len(all_ctx_lens)
        hdr += (
            f'<div class="cell" title="{nm}">'
            f'<span class="model-name">{nm}</span>'
            f'<span class="model-count">{n_done}/{n_total}</span>'
            f'</div>'
        )

    sections_html = ""
    for grp in GROUP_ORDER:
        if grp not in groups:
            continue
        tkeys = groups[grp]

        grp_cells = (
            f'<div class="cell">'
            f'<span class="arrow">&#9658;</span>'
            f'<span class="grp-label">{grp}</span>'
            f'</div>'
        )
        for nm in names:
            avgs = [task_ctx_avg(models[nm], k) for k in tkeys]
            avgs = [a for a in avgs if a is not None]
            grp_cells += val_cell(safe_avg(avgs) if avgs else None,
                                  incomplete=model_incomplete[nm])

        task_blocks_html = ""
        for k in tkeys:
            _, pmetric, dname = TASKS[k]

            task_hdr_cells = (
                f'<div class="cell task-hdr-cell">'
                f'<span class="t-name">{dname}</span>'
                f'<span class="t-meta">avg \u00b7 {pmetric}</span>'
                f'</div>'
            )
            for nm in names:
                task_hdr_cells += val_cell(task_ctx_avg(models[nm], k),
                                           incomplete=model_incomplete[nm])

            ctx_rows_html = ""
            for c in all_ctx_lens:
                if not any(c in models[nm].get(k, {}) for nm in names):
                    continue
                lbl = ctx_label(c)
                ctx_cells = f'<div class="cell ctx-cell">\u21b3 {lbl}</div>'
                for nm in names:
                    v = models[nm].get(k, {}).get(c, {}).get("value")
                    ctx_cells += val_cell(v)
                ctx_rows_html += (
                    f'<div class="grid-row ctx-row score-row ctx-{lbl}">'
                    f'{ctx_cells}</div>\n'
                )

            task_blocks_html += f"""
<div class="task-block">
  <div class="grid-row task-hdr-row score-row">{task_hdr_cells}</div>
  {ctx_rows_html}
</div>"""

        sections_html += f"""
<div class="section">
<details>
  <summary class="grid-row score-row">{grp_cells}</summary>
  <div class="group-body">{task_blocks_html}</div>
</details>
</div>"""

    # One row per context length in CTX_AVG_ROWS, showing the average of
    # ALL tasks at that specific context length.  Rows are skipped when no
    # model has any data at that context length.  Each row carries the
    # ctx-{lbl} class so it responds to the toggle buttons exactly like the
    # per-task detail rows do.
    ctx_avg_rows_html = ""
    for c in CTX_AVG_ROWS:
        # Skip if no model has any data at this context length
        if not any(
            c in models[nm].get(k, {})
            for nm in names
            for k in all_task_keys
        ):
            continue
        lbl = ctx_label(c)
        cells = f'<div class="cell ctx-avg-label">{lbl} Average</div>'
        for nm in names:
            avg, n_p, n_t = group_ctx_info(models[nm], all_task_keys, c)
            cells += val_cell(avg, incomplete=(n_p < n_t))
        ctx_avg_rows_html += (
            f'<div class="grid-row ctx-avg-row score-row ctx-{lbl}">'
            f'{cells}</div>\n'
        )

    overall_cells = '<div class="cell">Overall Average</div>'
    for nm in names:
        avgs = [task_ctx_avg(models[nm], k) for k in all_task_keys]
        avgs = [a for a in avgs if a is not None]
        overall_cells += val_cell(safe_avg(avgs) if avgs else None,
                                  incomplete=model_incomplete[nm])

    summary_html = build_summary_section(
        models, names, all_task_keys, all_ctx_lens, groups
    )

    ctx_btns = ""
    for c in all_ctx_lens:
        lbl = ctx_label(c)
        ctx_btns += (
            f'<button class="btn ctx-toggle active" data-ctx="{lbl}" '
            f'title="Toggle {lbl} rows / columns">{lbl}</button>'
        )

    ctx_list = " \u00b7 ".join(ctx_label(c) for c in all_ctx_lens)
    subtitle  = (
        f"{n} model(s) &middot; {len(all_task_keys)} task(s) &middot; "
        f"{len(all_ctx_lens)} context length(s): {ctx_list}"
    )
    src_note = str(long_dir.resolve())
    if short_dir:
        src_note += " + " + str(short_dir.resolve())
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HELMET Evaluation Results</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {{
  --bg:#f0f2f5; --card:#fff; --border:#dde1e7; --text:#161616;
  --muted:#6f6f6f; --header-bg:#f4f6f8; --row-hover:#f8f9fb;
  --group-bg:#edf0f5; --group-hover:#e2e6ed;
  --task-bg:#f7f8fa;  --task-hover:#edf0f5;
  --ctx-bg:#fff;
  --overall-bg:#f4f6f8;
  --best-bg:#d0f0c0; --best-text:#1a5c00; --best-border:rgba(26,92,0,.25);
  --inc-bg:#fff8e6;  --inc-text:#6d4c00;  --inc-border:#e8b84b;
  --radius:10px;
  --col-template:{col_template};
}}
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{font-family:'IBM Plex Sans',system-ui,sans-serif;background:var(--bg);
  color:var(--text);padding:36px 4%;max-width:2400px;margin:0 auto;line-height:1.5;}}
h1{{font-size:24px;font-weight:700;letter-spacing:-.02em;margin-bottom:4px;}}
.subtitle{{color:var(--muted);font-size:13px;margin-bottom:4px;}}
.path-note{{color:var(--muted);font-size:11px;font-family:'IBM Plex Mono',monospace;
  margin-bottom:20px;word-break:break-all;}}

/* - Controls - */
.controls{{display:flex;gap:8px;margin-bottom:20px;flex-wrap:wrap;align-items:center;}}
.divider{{width:1px;height:24px;background:var(--border);margin:0 4px;flex-shrink:0;}}
.btn{{background:var(--card);color:var(--text);border:1px solid var(--border);
  padding:6px 12px;border-radius:6px;font-size:12px;font-family:inherit;
  font-weight:500;cursor:pointer;transition:background .15s,border-color .15s;}}
.btn:hover{{background:var(--row-hover);border-color:#b0b8c4;}}
.ctx-toggle{{font-family:'IBM Plex Mono',monospace;font-size:11px;}}
.ctx-toggle.active{{background:#e8f0fe;border-color:#6b9aee;color:#1a56cc;}}
.ctx-toggle:not(.active){{color:#aaa;text-decoration:line-through;}}
.controls-label{{font-size:11px;color:var(--muted);font-weight:500;white-space:nowrap;}}
.legend{{font-size:12px;color:var(--muted);display:flex;align-items:center;
  gap:6px;margin-left:auto;flex-wrap:wrap;}}
.legend-best{{background:var(--best-bg);color:var(--best-text);padding:2px 8px;
  border-radius:4px;font-weight:600;font-family:'IBM Plex Mono',monospace;font-size:11px;
  box-shadow:inset 0 0 0 1px var(--best-border);}}

/* ═══════════════════════ MAIN TABLE ═══════════════════════ */
.table-box{{background:var(--card);border:1px solid var(--border);
  border-radius:var(--radius);box-shadow:0 2px 8px rgba(0,0,0,.06);
  overflow:hidden;overflow-x:auto;}}
.grid-row{{display:grid;grid-template-columns:var(--col-template);
  align-items:stretch;min-width:0;}}
.header-row{{background:var(--header-bg);border-bottom:2px solid var(--border);
  position:sticky;top:0;z-index:20;}}
.header-row .cell{{padding:12px 10px 10px;font-size:11px;font-weight:600;
  color:var(--muted);text-transform:uppercase;letter-spacing:.06em;
  border-left:1px solid var(--border);display:flex;flex-direction:column;
  align-items:center;justify-content:flex-end;word-break:break-word;
  hyphens:auto;gap:3px;}}
.header-row .cell:first-child{{align-items:flex-start;justify-content:center;
  border-left:none;font-size:12px;letter-spacing:.03em;}}
.model-name{{text-align:center;line-height:1.3;font-size:10px;}}
.model-count{{font-size:9px;color:#aaa;font-family:'IBM Plex Mono',monospace;font-weight:400;}}
.section{{border-bottom:1px solid var(--border);}}
.section:last-of-type{{border-bottom:none;}}
details{{width:100%;}}
summary{{list-style:none;cursor:pointer;background:var(--group-bg);
  width:100%;transition:background .15s;}}
summary::-webkit-details-marker{{display:none;}}
summary:hover{{background:var(--group-hover);}}
details[open] summary{{border-bottom:1px solid var(--border);}}
summary .cell{{padding:13px 10px;font-size:13px;font-weight:600;
  display:flex;align-items:center;justify-content:center;
  border-left:1px solid var(--border);
  font-family:'IBM Plex Mono',monospace;color:var(--text);}}
summary .cell:first-child{{justify-content:flex-start;border-left:none;
  font-family:'IBM Plex Sans',sans-serif;}}
.arrow{{display:inline-flex;width:18px;height:18px;align-items:center;
  justify-content:center;font-size:9px;color:var(--muted);
  margin-right:8px;transition:transform .2s;flex-shrink:0;}}
details[open] .arrow{{transform:rotate(90deg);}}
.grp-label{{font-weight:600;font-size:13px;}}
.group-body{{}}
.task-block{{border-bottom:1px solid #eef0f3;}}
.task-block:last-child{{border-bottom:none;}}
.task-hdr-row{{background:var(--task-bg);transition:background .12s;}}
.task-hdr-row:hover{{background:var(--task-hover);}}
.task-hdr-row .cell{{padding:9px 10px;display:flex;align-items:center;
  justify-content:center;border-left:1px solid var(--border);
  font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--muted);}}
.task-hdr-cell{{padding-left:36px !important;justify-content:flex-start !important;
  flex-direction:column;align-items:flex-start !important;gap:2px;
  font-family:'IBM Plex Sans',sans-serif !important;border-left:none !important;}}
.t-name{{font-weight:600;font-size:12.5px;color:#333;}}
.t-meta{{font-size:9.5px;color:#bbb;font-family:'IBM Plex Mono',monospace;font-weight:400;}}
.ctx-row{{background:var(--ctx-bg);border-bottom:1px solid #f3f4f6;transition:background .12s;}}
.ctx-row:last-child{{border-bottom:none;}}
.ctx-row:hover{{background:var(--row-hover);}}
.ctx-row .cell{{padding:7px 10px;font-size:12px;display:flex;
  align-items:center;justify-content:center;
  border-left:1px solid var(--border);
  font-family:'IBM Plex Mono',monospace;color:var(--muted);}}
.ctx-cell{{padding-left:54px !important;justify-content:flex-start !important;
  font-size:11.5px !important;color:#999 !important;border-left:none !important;}}
.overall{{background:var(--overall-bg);border-top:2px solid var(--border);}}
.overall .cell{{padding:15px 10px;font-size:14px;font-weight:700;
  display:flex;align-items:center;justify-content:center;
  border-left:1px solid var(--border);font-family:'IBM Plex Mono',monospace;}}
.overall .cell:first-child{{justify-content:flex-start;border-left:none;
  font-family:'IBM Plex Sans',sans-serif;text-transform:uppercase;
  font-size:12px;letter-spacing:.06em;color:var(--muted);font-weight:600;}}
.ctx-row.hidden,.ctx-avg-row.hidden{{display:none;}}

/* - Per-context-length average rows - */
.ctx-avg-row{{background:var(--group-bg);border-top:1px solid var(--border);}}
.ctx-avg-row + .ctx-avg-row{{border-top:1px solid #d8dde5;}}
.ctx-avg-row .cell{{padding:11px 10px;font-size:12px;font-weight:600;
  display:flex;align-items:center;justify-content:center;
  border-left:1px solid var(--border);
  font-family:'IBM Plex Mono',monospace;color:var(--text);}}
.ctx-avg-label{{justify-content:flex-start !important;border-left:none !important;
  font-family:'IBM Plex Sans',sans-serif !important;
  font-size:11px !important;letter-spacing:.04em;
  color:var(--muted) !important;font-weight:600 !important;
  text-transform:uppercase;padding-left:14px !important;}}
.ctx-avg-row:first-of-type{{border-top:2px solid var(--border);}}

/* ═══════════════════════ SHARED VALUE BADGES ═══════════════════════ */
.best-badge{{background:var(--best-bg);color:var(--best-text);
  padding:3px 9px;border-radius:5px;font-weight:600;display:inline-block;
  box-shadow:inset 0 0 0 1px var(--best-border);}}
.na{{color:#ccc;font-style:italic;font-size:13px;}}
.inc-cell{{background:var(--inc-bg);color:var(--inc-text);}}

/* ═══════════════════════ SUMMARY TABLE ═══════════════════════ */
.summary-section{{margin-top:52px;}}
.s-heading{{font-size:20px;font-weight:700;letter-spacing:-.015em;margin-bottom:6px;}}
.summary-note{{font-size:12px;color:var(--muted);margin-bottom:16px;line-height:1.6;}}
.inc-legend{{display:inline-flex;align-items:baseline;gap:3px;}}
.inc-ex{{background:var(--inc-bg);color:var(--inc-text);padding:1px 7px;
  border-radius:4px;font-family:'IBM Plex Mono',monospace;font-size:11px;
  font-weight:600;box-shadow:inset 0 0 0 1px var(--inc-border);}}
.summary-scroll{{overflow-x:auto;}}
.summary-wrap{{border:1px solid var(--border);border-radius:var(--radius);
  box-shadow:0 2px 8px rgba(0,0,0,.06);overflow:hidden;display:inline-block;
  min-width:100%;}}
.summary-table{{border-collapse:collapse;background:var(--card);
  white-space:nowrap;width:100%;}}
.summary-table th,.summary-table td{{border:1px solid var(--border);}}

/* Header rows */
.s-th-label{{background:var(--header-bg);color:var(--muted);font-size:11px;
  font-weight:600;text-transform:uppercase;letter-spacing:.06em;
  padding:10px 14px;text-align:left;min-width:210px;
  position:sticky;left:0;z-index:12;border-right:2px solid var(--border);}}
.s-th-model{{background:var(--group-bg);color:var(--text);
  font-size:10px;font-family:'IBM Plex Mono',monospace;font-weight:500;
  padding:8px 12px;text-align:center;max-width:180px;
  overflow:hidden;text-overflow:ellipsis;}}
.s-th-ctx{{background:var(--header-bg);color:var(--muted);
  font-size:10px;font-family:'IBM Plex Mono',monospace;font-weight:600;
  padding:6px 10px;text-align:center;letter-spacing:.04em;min-width:54px;}}
/* Left border that separates model groups */
.s-col-start,.s-th-model{{border-left:2px solid var(--border) !important;}}

/* Task-group label cells */
.s-td-label{{font-family:'IBM Plex Sans',sans-serif;font-size:12.5px;
  font-weight:500;color:var(--text);padding:10px 14px;
  background:var(--task-bg);
  position:sticky;left:0;z-index:5;border-right:2px solid var(--border);}}
.s-task-count{{font-size:10px;color:var(--muted);font-weight:400;}}

/* Value cells */
.s-cell{{font-family:'IBM Plex Mono',monospace;font-size:12px;
  color:var(--muted);text-align:center;padding:9px 10px;
  background:var(--ctx-bg);}}
.s-cell.s-inc{{background:var(--inc-bg);color:var(--inc-text);}}
.s-cell.s-na{{background:var(--ctx-bg);}}
.inc-mark{{color:var(--inc-border);font-size:9px;vertical-align:super;
  margin-left:1px;font-family:'IBM Plex Sans',sans-serif;font-weight:700;}}

/* Overall row */
.s-overall-row .s-td-label{{font-weight:700;text-transform:uppercase;
  font-size:11px;letter-spacing:.05em;color:var(--muted);
  background:var(--overall-bg);}}
.s-overall-row .s-cell{{font-weight:700;font-size:13px;background:var(--overall-bg);}}
.s-overall-row .s-cell.s-inc{{background:#fff0c5;}}

/* Hidden summary columns */
.s-cell.s-col-hidden,.s-th-ctx.s-col-hidden{{display:none;}}
</style>
</head>
<body>
<h1>HELMET Evaluation Results</h1>
<p class="subtitle">{subtitle}</p>
<p class="path-note">&#128193; {src_note} &nbsp;&middot;&nbsp; generated {ts}</p>
<div class="controls">
  <button class="btn" id="btn-expand">&#9660; Expand all</button>
  <button class="btn" id="btn-collapse">&#9654; Collapse all</button>
  <div class="divider"></div>
  <span class="controls-label">Context lengths:</span>
  {ctx_btns}
  <span class="legend">Best per row:&nbsp;<span class="legend-best">XX.XX</span></span>
</div>

<div class="table-box">
  <div class="grid-row header-row">{hdr}</div>
  {sections_html}
  {ctx_avg_rows_html}
  <div class="grid-row overall score-row">{overall_cells}</div>
</div>

{summary_html}

<script>
// - Expand / Collapse -
document.getElementById('btn-expand').addEventListener('click', function() {{
  document.querySelectorAll('details').forEach(function(d) {{ d.open = true; }});
}});
document.getElementById('btn-collapse').addEventListener('click', function() {{
  document.querySelectorAll('details').forEach(function(d) {{ d.open = false; }});
}});

// - Best-per-row: main table -
function updateBest() {{
  document.querySelectorAll('.score-row').forEach(function(row) {{
    if (row.classList.contains('hidden')) return;
    var cells = row.querySelectorAll('[data-val]');
    if (!cells.length) return;
    var best = -Infinity;
    cells.forEach(function(c) {{
      var v = parseFloat(c.getAttribute('data-val'));
      if (!isNaN(v) && v > best) best = v;
    }});
    cells.forEach(function(c) {{
      var v = parseFloat(c.getAttribute('data-val'));
      if (isNaN(v)) return;
      var mark = c.getAttribute('data-inc') === '1'
                 ? '<sup class="inc-mark">*</sup>' : '';
      var txt = v.toFixed(2);
      if (cells.length > 1 && Math.abs(v - best) < 1e-6)
        c.innerHTML = '<span class="best-badge">' + txt + '</span>' + mark;
      else
        c.innerHTML = txt + mark;
    }});
  }});
}}
updateBest();

// - Best-per-row: summary table -
function updateBestSummary() {{
  document.querySelectorAll('.sum-score-row').forEach(function(row) {{
    var cells = Array.from(row.querySelectorAll('[data-val]'))
                     .filter(function(c) {{ return !c.classList.contains('s-col-hidden'); }});
    if (!cells.length) return;
    var best = -Infinity;
    cells.forEach(function(c) {{
      var v = parseFloat(c.getAttribute('data-val'));
      if (!isNaN(v) && v > best) best = v;
    }});
    cells.forEach(function(c) {{
      var v = parseFloat(c.getAttribute('data-val'));
      if (isNaN(v)) return;
      var mark = c.getAttribute('data-inc') === '1'
                 ? '<sup class="inc-mark">*</sup>' : '';
      var txt  = v.toFixed(2);
      if (cells.length > 1 && Math.abs(v - best) < 1e-6)
        c.innerHTML = '<span class="best-badge">' + txt + '</span>' + mark;
      else
        c.innerHTML = txt + mark;
    }});
  }});
}}
updateBestSummary();

// - Context-length toggles -
document.querySelectorAll('.ctx-toggle').forEach(function(btn) {{
  btn.addEventListener('click', function() {{
    var lbl    = btn.getAttribute('data-ctx');
    var active = btn.classList.contains('active');
    btn.classList.toggle('active', !active);

    // Main table: show/hide rows
    document.querySelectorAll('.ctx-' + lbl).forEach(function(r) {{
      r.classList.toggle('hidden', active);
    }});

    // Summary table: show/hide columns
    document.querySelectorAll('.s-ctx-col-' + lbl).forEach(function(c) {{
      c.classList.toggle('s-col-hidden', active);
    }});

    updateBest();
    updateBestSummary();
  }});
}});
</script>
</body>
</html>"""


# Entry point
def _parse_models_arg(raw: Optional[str]) -> Optional[List[str]]:
    """
    Turn the --models CLI value into a list (or None = use ALLOWED_MODELS).

    Accepted forms
    --------------
    --models ALL                      → no filter (include every model on disk)
    --models "ModelA,ModelB,ModelC"   → explicit comma-separated list
    --models @/path/to/list.txt       → one model name per line in a text file
    (omitted)                         → fall back to the ALLOWED_MODELS constant
    """
    if raw is None:
        return ALLOWED_MODELS          # use the hardcoded constant

    if raw.strip().upper() == "ALL":
        return None                    # no filtering

    if raw.startswith("@"):
        path = Path(raw[1:])
        if not path.is_file():
            raise SystemExit(f"[ERROR] Model list file not found: {path}")
        names = [ln.strip() for ln in path.read_text().splitlines()
                 if ln.strip() and not ln.startswith("#")]
        return names or None

    # Comma-separated inline list
    names = [n.strip() for n in raw.split(",") if n.strip()]
    return names or None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Generate a multi-context-length HTML report from HELMET .score files. "
            "Pass --long for the primary outputs directory and optionally --short "
            "for a secondary directory with shorter context lengths."
        )
    )
    ap.add_argument("--long",  required=True,
        help="Primary outputs directory (e.g. 128 K context length).")
    ap.add_argument("--short", default=None,
        help="Optional secondary outputs directory (e.g. 8 K-64 K context lengths).")
    ap.add_argument("--out", default="helmet_results_ctx.html",
        help="Output HTML filename (default: helmet_results_ctx.html).")
    ap.add_argument("--sort", action="store_true",
        help="Sort model columns by overall average score (highest first).")
    ap.add_argument(
        "--models", default=None, metavar="LIST|ALL|@FILE",
        help=(
            'Which models to include.  Options: '
            '"ALL" to skip filtering; '
            'a comma-separated list of model names; '
            'or "@/path/to/file.txt" (one name per line).  '
            'Omit to use the ALLOWED_MODELS constant defined in this script.'
        ),
    )
    args = ap.parse_args()

    long_dir  = Path(args.long)
    short_dir = Path(args.short) if args.short else None

    if not long_dir.is_dir():
        raise SystemExit(f"[ERROR] Not a directory: {long_dir}")
    if short_dir and not short_dir.is_dir():
        raise SystemExit(f"[ERROR] Not a directory: {short_dir}")

    allowed = _parse_models_arg(args.models)

    print(f"Long-context dir:  {long_dir.resolve()}")
    if short_dir:
        print(f"Short-context dir: {short_dir.resolve()}")
    if allowed is None:
        print("Model filter:      (none — all models included)")
    else:
        print(f"Model filter:      {len(allowed)} model(s) in allow-list")
    print()

    models = collect_all(long_dir, short_dir, allowed_models=allowed)
    if not models:
        raise SystemExit(
            "[ERROR] No model sub-directories with *.score files found "
            "(after applying the model filter)."
        )

    all_task_keys = sorted(
        {k for m in models.values() for k in m},
        key=lambda k: (
            GROUP_ORDER.index(TASKS[k][0]) if TASKS[k][0] in GROUP_ORDER else 99,
            list(TASKS.keys()).index(k) if k in TASKS else 999,
        ),
    )
    all_ctx_lens = sorted({
        c for m in models.values() for ctx_s in m.values() for c in ctx_s
    })

    print(
        f"Found {len(models)} model(s), {len(all_task_keys)} task(s), "
        f"{len(all_ctx_lens)} context length(s): "
        f"{', '.join(ctx_label(c) for c in all_ctx_lens)}\n"
    )
    for nm, task_data in models.items():
        avgs = [task_ctx_avg(task_data, k) for k in all_task_keys]
        avgs = [a for a in avgs if a is not None]
        avg_str = f"{safe_avg(avgs):.2f}" if avgs else "n/a"
        print(f"  {nm}  (overall avg: {avg_str})")
        for k in all_task_keys:
            if k not in task_data:
                continue
            dname = TASKS[k][2]
            for c, info in sorted(task_data[k].items()):
                print(f"    {dname:<28} {ctx_label(c):<6}  {info['value']:7.2f}  ({info['metric']})")
        print()

    html     = build_html(models, long_dir, short_dir, sort_by_score=args.sort)
    out_path = Path(args.out)
    out_path.write_text(html, encoding="utf-8")
    print(f"Report written → {out_path.resolve()}")


if __name__ == "__main__":
    main()