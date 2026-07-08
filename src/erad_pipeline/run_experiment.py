"""Command-line entry point for the ERAD CPU/GPU experiment."""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .clustering import run_clustering
from .data_io import default_data_path, load_data
from .evaluation import evaluate, evaluate_clustering
from .preprocessing import preprocess_data
from .results import save_results
from .splitting import reference_feature_columns, split_data
from .supervised import TARGETS, predict, target_arrays, train_supervised_model
from .timer import StageTimer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ERAD-SE standardized CPU/GPU experiment.")
    parser.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--data-path", default=None, help=f"Dataset path. Default: {default_data_path()}")
    parser.add_argument("--n-samples", type=int, default=None, help="Optional sample size for validation/scaling runs.")
    parser.add_argument("--run-id", required=True, help="Identifier for this execution.")
    parser.add_argument("--output-dir", required=True, help="Directory where CSV outputs will be written.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--rf-random-state", type=int, default=10000)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--n-clusters", type=int, default=30)
    parser.add_argument("--n-trees", type=int, default=150)
    parser.add_argument("--svd-components", type=int, default=50)
    parser.add_argument(
        "--gpu-mode",
        choices=["accelerated", "strict"],
        default=os.environ.get("ERAD_GPU_MODE", "accelerated"),
        help=(
            "GPU mode. accelerated uses cuDF/cuML/CuPy where practical; strict keeps sklearn preprocessing/RF "
            "for closer CPU methodological parity."
        ),
    )
    parser.add_argument(
        "--gpu-max-categories-per-col",
        type=int,
        default=int(os.environ.get("ERAD_GPU_MAX_CATEGORIES_PER_COL", "64")),
        help="Maximum frequent categorical values kept per column in accelerated GPU ordinal encoding.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_path = run_experiment(args)
    print(f"Results written to: {output_path}")


def run_experiment(args: argparse.Namespace) -> Path:
    if args.n_samples is not None and args.n_samples <= 0:
        raise ValueError("--n-samples must be positive when provided")

    np.random.seed(args.random_state)
    total_start = time.perf_counter()

    # The timer is initialized after load_data, once dataset_name is known. The
    # reading time is measured manually to keep the output schema uniform.
    read_start = time.perf_counter()
    df, data_info = load_data(
        args.data_path,
        backend=args.backend,
        n_samples=args.n_samples,
        random_state=args.random_state,
    )
    read_seconds = time.perf_counter() - read_start

    timer = StageTimer(
        backend=args.backend,
        run_id=str(args.run_id),
        n_samples=args.n_samples,
        dataset_name=str(data_info["dataset_name"]),
    )
    backend_report: list[dict[str, object]] = []
    metrics_rows: list[dict[str, object]] = []

    timer.add_manual(
        "leitura",
        read_seconds,
        implementation_backend=str(data_info["actual_backend"]),
        observacoes=str(data_info["notes"]),
    )
    backend_report.append(_backend_row(args, "leitura", data_info["actual_backend"], data_info["notes"]))

    with timer.track(
        "split",
        implementation_backend="cpu",
        observacoes="Shared StratifiedGroupKFold split from the CPU reference pipeline.",
    ):
        splits, groups = split_data(df, random_state=args.random_state)
    backend_report.append(
        _backend_row(
            args,
            "split",
            "cpu",
            "Split remains on CPU to guarantee identical train/validation/test indices for both backends.",
        )
    )

    feature_cols = reference_feature_columns(df)

    with timer.track(
        "preprocessamento",
        implementation_backend="cpu",
        observacoes="Shared sklearn preprocessing for methodological parity.",
    ):
        preprocessed, preprocessing_backend = preprocess_data(
            df,
            splits,
            feature_cols,
            backend=args.backend,
            svd_components=args.svd_components,
            random_state=args.random_state,
            gpu_mode=args.gpu_mode,
            gpu_max_categories_per_col=args.gpu_max_categories_per_col,
        )
    _replace_last_backend(timer.rows, preprocessing_backend["implementation_backend"], preprocessing_backend["observacoes"])
    backend_report.append(
        _backend_row(
            args,
            "preprocessamento",
            preprocessing_backend["implementation_backend"],
            preprocessing_backend["observacoes"],
        )
    )

    with timer.track("clusterizacao", implementation_backend=args.backend, observacoes="See backend_report.csv."):
        clustering, clustering_meta = run_clustering(
            preprocessed.X_cluster,
            backend=args.backend,
            n_clusters=args.n_clusters,
            random_state=args.random_state,
        )
    _replace_last_backend(timer.rows, clustering_meta["implementation_backend"], clustering_meta["observacoes"])
    backend_report.append(
        _backend_row(
            args,
            "clusterizacao",
            clustering_meta["implementation_backend"],
            clustering_meta["observacoes"],
        )
    )

    metrics_rows.extend(
        evaluate_clustering(
            backend=args.backend,
            run_id=str(args.run_id),
            n_samples=args.n_samples,
            dataset_name=str(data_info["dataset_name"]),
            df=df,
            splits=splits,
            cluster_labels=clustering["labels"],
        )
    )

    targets = target_arrays(df, splits, clustering["labels"])
    preprocessed.X_cluster.clear()
    gc.collect()

    for target in TARGETS:
        with timer.track(
            "treinamento_supervisionado",
            implementation_backend="cpu",
            observacoes=f"Training target={target}.",
        ):
            model, train_meta = train_supervised_model(
                preprocessed.X_supervised["train"],
                targets[target]["train"],
                target=target,
                backend=args.backend,
                n_estimators=args.n_trees,
                random_state=args.rf_random_state,
                n_jobs=args.n_jobs,
            )
        _replace_last_backend(timer.rows, train_meta["implementation_backend"], train_meta["observacoes"])
        backend_report.append(
            _backend_row(
                args,
                f"treinamento_supervisionado:{target}",
                train_meta["implementation_backend"],
                train_meta["observacoes"],
            )
        )

        for split in ["valid", "test"]:
            with timer.track(
                "predicao",
                implementation_backend=train_meta["implementation_backend"],
                observacoes=f"Prediction target={target}, split={split}.",
            ):
                y_pred = predict(model, preprocessed.X_supervised[split])

            with timer.track(
                "avaliacao",
                implementation_backend="cpu",
                observacoes=f"Evaluation target={target}, split={split}.",
            ):
                metrics_rows.append(
                    evaluate(
                        backend=args.backend,
                        run_id=str(args.run_id),
                        n_samples=args.n_samples,
                        dataset_name=str(data_info["dataset_name"]),
                        target=target,
                        split=split,
                        y_true=targets[target][split],
                        y_pred=y_pred,
                        metric_group="supervised",
                        observacoes="Supervised metrics computed with sklearn for both backends.",
                    )
                )
            del y_pred

        del model
        gc.collect()

    cluster_assignments = _cluster_assignment_frame(splits, clustering)

    timer.add_manual(
        "tempo_total",
        time.perf_counter() - total_start,
        implementation_backend=_effective_backend(timer.rows),
        observacoes="End-to-end wall time.",
    )

    output_dir = Path(args.output_dir)
    output_dir = output_dir / args.backend / str(args.run_id)
    return save_results(
        output_dir,
        timings=timer.rows,
        metrics=metrics_rows,
        backend_report=backend_report,
        split_indices=splits,
        cluster_assignments=cluster_assignments,
        metadata={
            "args": vars(args),
            "data_info": data_info,
            "feature_cols_clustering_reference": feature_cols,
            "feature_cols_supervised": preprocessed.supervised_feature_cols,
            "preprocessing": preprocessed.metadata,
            "source_cpu": r"C:\Users\AlphaXD\Desktop\UFV\IC\Parte-2",
            "source_gpu": r"C:\Users\AlphaXD\Desktop\UFV\IC\Estendido\ids_generalization_pipeline",
            "methodology_note": (
                "The experiment follows the CPU reference methodology. GPU is used only when an equivalent "
                "stage is available without changing split, features, targets, hyperparameters or metrics."
            ),
        },
    )


def _cluster_assignment_frame(splits: dict[str, np.ndarray], clustering: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for split, idx in splits.items():
        frames.append(
            pd.DataFrame(
                {
                    "split": split,
                    "row_index": idx,
                    "cluster_kmeans30": clustering["labels"][split],
                    "cluster_distance": clustering["distances"][split],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _backend_row(
    args: argparse.Namespace,
    etapa: str,
    implementation_backend: object,
    observacoes: object,
) -> dict[str, object]:
    return {
        "backend": args.backend,
        "run_id": args.run_id,
        "n_samples": args.n_samples,
        "etapa": etapa,
        "implementation_backend": implementation_backend,
        "observacoes": observacoes,
    }


def _replace_last_backend(rows: list[dict[str, object]], implementation_backend: object, observacoes: object) -> None:
    rows[-1]["implementation_backend"] = implementation_backend
    rows[-1]["observacoes"] = observacoes


def _effective_backend(rows: list[dict[str, object]]) -> str:
    used = {row["implementation_backend"] for row in rows if row.get("implementation_backend") in {"cpu", "gpu"}}
    if used == {"gpu"}:
        return "gpu"
    if "gpu" in used:
        return "mixed"
    return "cpu"


if __name__ == "__main__":
    main(sys.argv[1:])
