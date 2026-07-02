"""Clustering stage for the ERAD experiment."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def run_clustering(
    X_cluster: dict[str, Any],
    *,
    backend: str,
    n_clusters: int = 30,
    random_state: int = 42,
    batch_size: int = 4096,
    n_init: int = 10,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if backend == "gpu":
        try:
            return _run_gpu_kmeans(
                X_cluster,
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=n_init,
            )
        except Exception as exc:
            result, metadata = _run_cpu_minibatch_kmeans(
                X_cluster,
                n_clusters=n_clusters,
                random_state=random_state,
                batch_size=batch_size,
                n_init=n_init,
            )
            metadata["observacoes"] = f"GPU KMeans unavailable; CPU MiniBatchKMeans fallback used: {exc}"
            return result, metadata

    return _run_cpu_minibatch_kmeans(
        X_cluster,
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=batch_size,
        n_init=n_init,
    )


def _run_cpu_minibatch_kmeans(
    X_cluster: dict[str, Any],
    *,
    n_clusters: int,
    random_state: int,
    batch_size: int,
    n_init: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=batch_size,
        n_init=n_init,
    )
    labels = {"train": model.fit_predict(X_cluster["train"])}
    labels["valid"] = model.predict(X_cluster["valid"])
    labels["test"] = model.predict(X_cluster["test"])
    distances = {split: np.min(model.transform(values), axis=1) for split, values in X_cluster.items()}
    return (
        {"model": model, "labels": labels, "distances": distances},
        {
            "implementation_backend": "cpu",
            "algorithm": "sklearn MiniBatchKMeans",
            "observacoes": "CPU reference clustering from Parte-2/pipeline_meshi_local.py",
        },
    )


def _run_gpu_kmeans(
    X_cluster: dict[str, Any],
    *,
    n_clusters: int,
    random_state: int,
    n_init: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import cupy as cp
    from cuml.cluster import KMeans

    x_gpu = {split: cp.asarray(values, dtype=cp.float32) for split, values in X_cluster.items()}
    try:
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    except TypeError:
        model = KMeans(n_clusters=n_clusters, random_state=random_state)

    labels = {"train": _to_numpy(model.fit_predict(x_gpu["train"])).astype(int)}
    labels["valid"] = _to_numpy(model.predict(x_gpu["valid"])).astype(int)
    labels["test"] = _to_numpy(model.predict(x_gpu["test"])).astype(int)

    centers = cp.asarray(model.cluster_centers_)
    distances = {}
    for split, values in x_gpu.items():
        assigned = centers[cp.asarray(labels[split])]
        distances[split] = _to_numpy(cp.linalg.norm(values - assigned, axis=1))

    return (
        {"model": model, "labels": labels, "distances": distances},
        {
            "implementation_backend": "gpu",
            "algorithm": "cuML KMeans",
            "observacoes": (
                "cuML KMeans is the GPU equivalent used for the KMeans stage. "
                "The CPU reference estimator is sklearn MiniBatchKMeans, so this difference must be reported."
            ),
        },
    )


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
