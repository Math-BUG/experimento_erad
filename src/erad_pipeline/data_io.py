"""Data loading for the ERAD experiment."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

import pandas as pd


def default_data_path() -> Path:
    ic_root = Path(__file__).resolve().parents[4]
    parquet_path = ic_root / "Parte-2" / "train_test_network.parquet"
    if parquet_path.exists():
        return parquet_path
    return ic_root / "Parte-2" / "train_test_network.csv"


def load_data(
    data_path: str | Path | None,
    *,
    backend: str,
    n_samples: int | None,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(data_path) if data_path else default_data_path()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    actual_backend = "cpu"
    notes = "pandas CSV/Parquet reader"
    force_pandas_io = os.environ.get("ERAD_FORCE_PANDAS_IO", "").strip().lower() in {"1", "true", "yes"}
    if backend == "gpu" and not force_pandas_io:
        try:
            df = _read_with_cudf(path)
            actual_backend = "gpu"
            notes = "cuDF reader, converted to pandas to preserve the shared split and sklearn-compatible metadata"
        except Exception as exc:
            df = _read_with_pandas(path)
            notes = f"cuDF read unavailable; pandas fallback used for IO: {exc}"
    elif backend == "gpu" and force_pandas_io:
        df = _read_with_pandas(path)
        notes = "pandas reader forced by ERAD_FORCE_PANDAS_IO for lower host-memory pressure"
    else:
        df = _read_with_pandas(path)

    if n_samples is not None and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    required = {"label", "type"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset must contain columns {sorted(required)}; missing {sorted(missing)}")

    return df, {
        "data_path": str(path),
        "dataset_name": path.stem,
        "actual_backend": actual_backend,
        "notes": notes,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
    }


def _read_with_pandas(path: Path) -> pd.DataFrame:
    if path.is_dir():
        return _read_directory_with_pandas(path)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported data extension {path.suffix!r}. Use CSV or Parquet.")


def _read_with_cudf(path: Path) -> pd.DataFrame:
    import cudf

    if path.is_dir():
        parquet_files = sorted(path.rglob("*.parquet")) + sorted(path.rglob("*.pq"))
        if parquet_files:
            return _cudf_to_pandas(cudf.read_parquet([str(file) for file in parquet_files]))
        csv_files = sorted(path.rglob("*.csv"))
        if csv_files:
            frames = [cudf.read_csv(str(file)) for file in csv_files]
            return _cudf_to_pandas(cudf.concat(frames, ignore_index=True))
        raise FileNotFoundError(f"No CSV/Parquet files found inside directory: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _cudf_to_pandas(cudf.read_csv(str(path)))
    if suffix in {".parquet", ".pq"}:
        return _cudf_to_pandas(cudf.read_parquet(str(path)))
    raise ValueError(f"Unsupported data extension {path.suffix!r}. Use CSV or Parquet.")


def _cudf_to_pandas(frame: Any) -> pd.DataFrame:
    df = frame.to_pandas()
    del frame
    gc.collect()
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    return df


def _read_directory_with_pandas(path: Path) -> pd.DataFrame:
    parquet_files = sorted(path.rglob("*.parquet")) + sorted(path.rglob("*.pq"))
    if parquet_files:
        frames = [pd.read_parquet(file) for file in parquet_files]
        return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    csv_files = sorted(path.rglob("*.csv"))
    if csv_files:
        frames = [pd.read_csv(file, low_memory=False) for file in csv_files]
        return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    raise FileNotFoundError(f"No CSV/Parquet files found inside directory: {path}")
