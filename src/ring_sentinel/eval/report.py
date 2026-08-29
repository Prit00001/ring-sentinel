"""Write reports/*.md from computed metrics (build spec 12, 14).

Deliberately no `tabulate` / DataFrame.to_markdown() dependency — that would
be one more package outside the build prompt's pre-approved list
(pandas, numpy, scikit-learn, lightgbm, scipy, networkx, pyarrow, fastapi,
uvicorn, jinja2, pydantic, matplotlib, pytest, groq, python-dotenv). A
Markdown table is five lines of string formatting; it doesn't need one.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if v != v:  # NaN
            return "n/a"
        return f"{v:.4f}"
    return str(v)


def df_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no rows)_\n"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def write_results_md(
    path: Path,
    headline_rows: dict[str, dict],
    fpr_points: list[float],
) -> None:
    """headline_rows maps a row label (e.g. "Rules baseline", "LGBM base",
    "+ ring features") to a headline_metrics() dict."""
    path.parent.mkdir(parents=True, exist_ok=True)

    metric_names = ["pr_auc"] + [f"recall_at_fpr_{fp}" for fp in fpr_points] + [
        "precision_at_k", "brier", "ece", "roc_auc",
    ]
    display_names = ["PR-AUC"] + [f"Recall @ {fp:.1%} FPR" for fp in fpr_points] + [
        "Precision @ k", "Brier score", "ECE", "ROC-AUC*",
    ]

    lines = ["# Ring Sentinel — Results\n"]
    lines.append("| Metric | " + " | ".join(headline_rows.keys()) + " |")
    lines.append("|---|" + "|".join(["---"] * len(headline_rows)) + "|")
    for metric, display in zip(metric_names, display_names):
        vals = [_fmt(row.get(metric)) for row in headline_rows.values()]
        lines.append(f"| {display} | " + " | ".join(vals) + " |")
    lines.append("")

    any_row = next(iter(headline_rows.values()), {})
    if any_row.get("roc_auc_caveat"):
        lines.append(f"\\*{any_row['roc_auc_caveat']}\n")

    path.write_text("\n".join(lines))


def write_ablation_md(path: Path, ablation: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "# Ring Sentinel — Ablation\n\n" + df_to_markdown_table(ablation)
    path.write_text(text)


def write_slices_md(path: Path, slices: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = ["# Ring Sentinel — Slice metrics\n"]
    for name, table in slices.items():
        parts.append(f"\n## {name}\n")
        parts.append(df_to_markdown_table(table))
    path.write_text("\n".join(parts))


def write_drift_md(path: Path, drift: pd.DataFrame, flag_threshold: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_flagged = int(drift["flagged"].sum()) if not drift.empty else 0
    parts = [
        "# Ring Sentinel — Drift report (PSI)\n",
        f"Flag threshold: PSI > {flag_threshold}. {n_flagged} feature(s) flagged.\n",
        df_to_markdown_table(drift),
    ]
    path.write_text("\n".join(parts))
