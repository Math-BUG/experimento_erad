"""Supervised learning stage for the ERAD experiment."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

TARGETS = ["cluster_kmeans30", "type", "label"]


def train_supervised_model(
    X_train: Any,
    y_train: np.ndarray,
    *,
    target: str,
    backend: str,
    n_estimators: int = 150,
    random_state: int = 10000,
    n_jobs: int = 1,
) -> tuple[Any, dict[str, str]]:
    if backend == "gpu":
        try:
            return _train_gpu_random_forest(
                X_train,
                y_train,
                target=target,
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=n_jobs,
            )
        except Exception as exc:
            allow_fallback = os.environ.get("ERAD_ALLOW_GPU_FALLBACK", "").strip().lower() in {"1", "true", "yes"}
            if not allow_fallback:
                raise
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=n_jobs,
                class_weight="balanced_subsample",
            )
            model.fit(X_train, y_train)
            return (
                model,
                {
                    "implementation_backend": "cpu",
                    "observacoes": (
                        f"Target {target}: cuML RandomForest failed and ERAD_ALLOW_GPU_FALLBACK=1 enabled "
                        f"sklearn fallback. Error: {exc}"
                    ),
                },
            )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)
    return (
        model,
        {
            "implementation_backend": "cpu",
            "observacoes": (
                f"Target {target}: sklearn RandomForestClassifier with class_weight='balanced_subsample'."
            ),
        },
    )


def predict(model: Any, X: Any) -> np.ndarray:
    if isinstance(model, dict) and model.get("backend") == "gpu":
        encoded = _to_numpy(model["model"].predict(_to_cupy_float32(X))).astype(np.int32)
        return model["label_encoder"].inverse_transform(encoded)
    return model.predict(X)


def _train_gpu_random_forest(
    X_train: Any,
    y_train: np.ndarray,
    *,
    target: str,
    n_estimators: int,
    random_state: int,
    n_jobs: int,
) -> tuple[Any, dict[str, str]]:
    import cupy as cp
    from cuml.ensemble import RandomForestClassifier as CuMLRandomForestClassifier

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(np.asarray(y_train)).astype(np.int32)
    X_gpu = _to_cupy_float32(X_train)
    y_gpu = cp.asarray(y_encoded, dtype=cp.int32)
    model = CuMLRandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_streams=max(1, min(int(n_jobs), 16)),
    )
    model.fit(X_gpu, y_gpu)
    return (
        {"backend": "gpu", "model": model, "label_encoder": encoder},
        {
            "implementation_backend": "gpu",
            "observacoes": (
                f"Target {target}: cuML RandomForestClassifier accelerated backend. "
                "This does not use sklearn class_weight='balanced_subsample' and is not a strict "
                "methodological equivalent of the CPU RandomForest."
            ),
        },
    )


def _to_cupy_float32(values: Any) -> Any:
    import cupy as cp

    if isinstance(values, cp.ndarray):
        return values.astype(cp.float32, copy=False)
    if hasattr(values, "toarray"):
        values = values.toarray()
    return cp.asarray(values, dtype=cp.float32)


def _to_numpy(values: Any) -> np.ndarray:
    try:
        import cupy as cp

        if isinstance(values, cp.ndarray):
            return cp.asnumpy(values)
    except Exception:
        pass
    if hasattr(values, "to_numpy"):
        return values.to_numpy()
    if hasattr(values, "get"):
        return values.get()
    return np.asarray(values)


def target_arrays(
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    cluster_labels: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    return {
        "cluster_kmeans30": {split: cluster_labels[split].astype(int) for split in splits},
        "type": {split: df.iloc[idx]["type"].astype(str).to_numpy() for split, idx in splits.items()},
        "label": {split: df.iloc[idx]["label"].astype(str).to_numpy() for split, idx in splits.items()},
    }
