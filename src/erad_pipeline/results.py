"""Result writing helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def save_results(
    output_dir: str | Path,
    *,
    timings: list[dict[str, object]],
    metrics: list[dict[str, object]],
    backend_report: list[dict[str, object]],
    split_indices: dict[str, np.ndarray],
    cluster_assignments: pd.DataFrame,
    metadata: dict[str, Any],
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "splits").mkdir(exist_ok=True)

    pd.DataFrame(timings).to_csv(out / "timings.csv", index=False)
    pd.DataFrame(metrics).to_csv(out / "metrics_quality.csv", index=False)
    pd.DataFrame(backend_report).to_csv(out / "backend_report.csv", index=False)
    cluster_assignments.to_csv(out / "cluster_assignments.csv", index=False)

    for split, idx in split_indices.items():
        pd.DataFrame({"row_index": idx}).to_csv(out / "splits" / f"{split}_indices.csv", index=False)

    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")
    return out


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)
