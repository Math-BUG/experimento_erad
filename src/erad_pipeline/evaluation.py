"""Evaluation metrics for clustering and supervised stages."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    v_measure_score,
)


def evaluate(
    *,
    backend: str,
    run_id: str,
    n_samples: int | None,
    dataset_name: str,
    target: str,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_group: str,
    observacoes: str = "",
) -> dict[str, object]:
    return {
        "backend": backend,
        "run_id": run_id,
        "n_samples": n_samples,
        "nome_do_dataset": dataset_name,
        "metric_group": metric_group,
        "target": target,
        "split": split,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "ari": None,
        "ami": None,
        "v_measure": None,
        "observacoes": observacoes,
    }


def evaluate_clustering(
    *,
    backend: str,
    run_id: str,
    n_samples: int | None,
    dataset_name: str,
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    cluster_labels: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split, idx in splits.items():
        for target in ["type", "label"]:
            y_true = df.iloc[idx][target].astype(str).to_numpy()
            y_cluster = cluster_labels[split]
            rows.append(
                {
                    "backend": backend,
                    "run_id": run_id,
                    "n_samples": n_samples,
                    "nome_do_dataset": dataset_name,
                    "metric_group": "clustering",
                    "target": target,
                    "split": split,
                    "accuracy": None,
                    "balanced_accuracy": None,
                    "macro_f1": None,
                    "weighted_f1": None,
                    "mcc": None,
                    "ari": adjusted_rand_score(y_true, y_cluster),
                    "ami": adjusted_mutual_info_score(y_true, y_cluster, average_method="arithmetic"),
                    "v_measure": v_measure_score(y_true, y_cluster),
                    "observacoes": "External clustering metrics comparing cluster_kmeans30 with target labels.",
                }
            )
    return rows
