"""Aggregate per-run result files (as written by meta_bridge.summary_results, tab-separated with
a header row) into one Markdown table per phase/RQ directory.

Keeps the useful part of archive/legacy_notebooks/secalign_SECALIGN_Untitled1.ipynb (tabulating
results across models/benchmarks into a summary table) while dropping its hand-rolled ASR
computation entirely — this module only aggregates numbers that evaluation/meta_eval_runner.py
and evaluation/attack_vectors_eval.py already computed via the official judging logic.
"""

from __future__ import annotations

import csv
from pathlib import Path


def _read_tsv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def build_report(results_dir: Path, pattern: str = "*.tsv") -> str:
    """Read every *.tsv file under results_dir and render one combined Markdown table."""
    rows: list[dict] = []
    for tsv_path in sorted(results_dir.glob(pattern)):
        rows.extend(_read_tsv(tsv_path))

    if not rows:
        return f"(no result files matching {pattern} found under {results_dir})\n"

    columns = list(dict.fromkeys(key for row in rows for key in row))
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines) + "\n"


def write_report(results_dir: Path, pattern: str = "*.tsv", out_name: str = "report.md") -> Path:
    report = build_report(results_dir, pattern)
    out_path = results_dir / out_name
    out_path.write_text(report, encoding="utf-8")
    return out_path
