"""CSV logging utilities for evaluation metrics."""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Any, Dict


def _ensure_directory(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)


def log_metrics_to_csv(
    save_path: str,
    dataset_name: str,
    metrics: Dict[str, float],
    extra_info: Dict[str, Any] | None = None,
) -> None:
    """Append the metrics of one evaluation run to ``save_path``."""

    _ensure_directory(save_path)
    now = datetime.now()
    row: Dict[str, Any] = {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "dataset": dataset_name,
    }
    row.update(metrics)
    if extra_info:
        row.update(extra_info)

    file_exists = os.path.isfile(save_path)
    fieldnames = list(row.keys())
    with open(save_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)