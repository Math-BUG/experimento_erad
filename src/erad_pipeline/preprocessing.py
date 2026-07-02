"""Preprocessing for clustering and supervised learning."""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SUPERVISED_LEAK_COLUMNS = [
    "label",
    "type",
    "cluster_kmeans30",
    "cluster_name",
    "purity_test",
    "split",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
]


class PortsConnectivityFeaturizer(BaseEstimator, TransformerMixin):
    """CPU reference featurizer from Parte-2/pipeline_meshi_local.py."""

    def __init__(self, topN: int = 20, log_counts: bool = True, drop_raw_ips: bool = True):
        self.topN = topN
        self.log_counts = log_counts
        self.drop_raw_ips = drop_raw_ips

    def fit(self, X: pd.DataFrame, y: Any = None) -> "PortsConnectivityFeaturizer":
        _require_columns(X, ["src_port", "dst_port", "src_ip", "dst_ip"])
        frame = X.copy()
        frame["src_port"] = pd.to_numeric(frame["src_port"], errors="coerce")
        frame["dst_port"] = pd.to_numeric(frame["dst_port"], errors="coerce")
        self.top_src_ = set(frame["src_port"].value_counts().head(self.topN).index)
        self.top_dst_ = set(frame["dst_port"].value_counts().head(self.topN).index)

        frame["src_ip"] = frame["src_ip"].astype(str).fillna("NA")
        frame["dst_ip"] = frame["dst_ip"].astype(str).fillna("NA")
        self.vc_pair_ = frame.groupby(["src_ip", "dst_ip"], sort=False, observed=True).size()
        self.vc_src_ = frame["src_ip"].value_counts()
        self.vc_dst_ = frame["dst_ip"].value_counts()
        self.src_unique_dst_ = frame.groupby("src_ip")["dst_ip"].nunique()
        self.dst_unique_src_ = frame.groupby("dst_ip")["src_ip"].nunique()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        _require_columns(X, ["src_port", "dst_port", "src_ip", "dst_ip"])
        frame = X.copy()
        frame["src_port"] = pd.to_numeric(frame["src_port"], errors="coerce")
        frame["dst_port"] = pd.to_numeric(frame["dst_port"], errors="coerce")
        frame["src_port_wk"] = (frame["src_port"] < 1024).fillna(False).astype("int8")
        frame["dst_port_wk"] = (frame["dst_port"] < 1024).fillna(False).astype("int8")
        frame["src_port_top"] = frame["src_port"].astype("object").where(
            frame["src_port"].isin(self.top_src_),
            other="other",
        ).astype(str)
        frame["dst_port_top"] = frame["dst_port"].astype("object").where(
            frame["dst_port"].isin(self.top_dst_),
            other="other",
        ).astype(str)
        frame = frame.drop(columns=["src_port", "dst_port"])

        frame["src_ip"] = frame["src_ip"].astype(str).fillna("NA")
        frame["dst_ip"] = frame["dst_ip"].astype(str).fillna("NA")
        pair_index = pd.MultiIndex.from_frame(frame[["src_ip", "dst_ip"]])
        frame["pair_count"] = pair_index.map(self.vc_pair_).fillna(0).astype(float)
        frame["src_out_count"] = frame["src_ip"].map(self.vc_src_).fillna(0).astype(float)
        frame["dst_in_count"] = frame["dst_ip"].map(self.vc_dst_).fillna(0).astype(float)
        frame["src_unique_dst"] = frame["src_ip"].map(self.src_unique_dst_).fillna(0).astype(float)
        frame["dst_unique_src"] = frame["dst_ip"].map(self.dst_unique_src_).fillna(0).astype(float)

        if self.log_counts:
            for col in ["pair_count", "src_out_count", "dst_in_count", "src_unique_dst", "dst_unique_src"]:
                frame[col] = np.log1p(frame[col])

        if self.drop_raw_ips:
            frame = frame.drop(columns=["src_ip", "dst_ip"], errors="ignore")
        return frame


@dataclass
class PreprocessResult:
    X_cluster: dict[str, Any]
    X_supervised: dict[str, Any]
    supervised_feature_cols: list[str]
    metadata: dict[str, Any]


def preprocess_data(
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    feature_cols: list[str],
    *,
    backend: str,
    svd_components: int = 50,
    onehot_min_frequency: int = 10,
    random_state: int = 42,
    gpu_mode: str = "accelerated",
    gpu_max_categories_per_col: int = 64,
) -> tuple[PreprocessResult, dict[str, str]]:
    if backend == "gpu" and gpu_mode == "accelerated":
        result = _preprocess_gpu_accelerated(
            df,
            splits,
            feature_cols,
            svd_components=svd_components,
            onehot_min_frequency=onehot_min_frequency,
            random_state=random_state,
            max_categories_per_col=gpu_max_categories_per_col,
        )
        notes = {
            "implementation_backend": "gpu",
            "observacoes": (
                "Accelerated GPU preprocessing: CPU PortsConnectivityFeaturizer for derived counts, "
                "then CuPy/cuML dense preprocessing with frequency-limited ordinal categorical encoding "
                "and cuML dimensionality reduction. This is faster but not representation-identical to "
                "the CPU one-hot sklearn path."
            ),
        }
        return result, notes

    cluster_X, cluster_metadata = _preprocess_for_clustering(
        df,
        splits,
        feature_cols,
        svd_components=svd_components,
        onehot_min_frequency=onehot_min_frequency,
        random_state=random_state,
    )
    supervised_X, supervised_features, supervised_metadata = _preprocess_for_supervised(
        df,
        splits,
        onehot_min_frequency=onehot_min_frequency,
    )

    notes = {
        "implementation_backend": "cpu",
        "observacoes": (
            "Reference sklearn preprocessing is kept on CPU for both backends. "
            "It preserves PortsConnectivityFeaturizer, OneHotEncoder(min_frequency=10), "
            "TruncatedSVD and the exact supervised feature representation."
        ),
    }
    return (
        PreprocessResult(
            X_cluster=cluster_X,
            X_supervised=supervised_X,
            supervised_feature_cols=supervised_features,
            metadata={"cluster": cluster_metadata, "supervised": supervised_metadata},
        ),
        notes,
    )


def _preprocess_gpu_accelerated(
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    feature_cols: list[str],
    *,
    svd_components: int,
    onehot_min_frequency: int,
    random_state: int,
    max_categories_per_col: int,
) -> PreprocessResult:
    cluster_X, cluster_metadata = _preprocess_gpu_for_clustering(
        df,
        splits,
        feature_cols,
        svd_components=svd_components,
        onehot_min_frequency=onehot_min_frequency,
        random_state=random_state,
        max_categories_per_col=max_categories_per_col,
    )
    supervised_X, supervised_features, supervised_metadata = _preprocess_gpu_for_supervised(
        df,
        splits,
        onehot_min_frequency=onehot_min_frequency,
        max_categories_per_col=max_categories_per_col,
    )
    return PreprocessResult(
        X_cluster=cluster_X,
        X_supervised=supervised_X,
        supervised_feature_cols=supervised_features,
        metadata={"cluster": cluster_metadata, "supervised": supervised_metadata},
    )


def _preprocess_gpu_for_clustering(
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    feature_cols: list[str],
    *,
    svd_components: int,
    onehot_min_frequency: int,
    random_state: int,
    max_categories_per_col: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    featurizer = PortsConnectivityFeaturizer(topN=20, log_counts=True, drop_raw_ips=True)
    raw_train = df.iloc[splits["train"]][feature_cols]
    x_train = featurizer.fit_transform(raw_train)
    del raw_train
    gc.collect()

    raw_valid = df.iloc[splits["valid"]][feature_cols]
    x_valid = featurizer.transform(raw_valid)
    del raw_valid
    gc.collect()

    raw_test = df.iloc[splits["test"]][feature_cols]
    x_test = featurizer.transform(raw_test)
    del raw_test
    gc.collect()

    num_cols = x_train.select_dtypes(exclude=["object", "string"]).columns.tolist()
    cat_cols = x_train.select_dtypes(include=["object", "string"]).columns.tolist()
    already_logged = {"pair_count", "src_out_count", "dst_in_count", "src_unique_dst", "dst_unique_src"}
    skew_cols = [c for c in num_cols if c not in ["src_port_wk", "dst_port_wk"] and c not in already_logged]
    flag_cols = [c for c in num_cols if c not in skew_cols]

    x_gpu, encoding_metadata = _gpu_dense_matrices(
        x_train,
        x_valid,
        x_test,
        skew_cols=skew_cols,
        flag_cols=flag_cols,
        cat_cols=cat_cols,
        onehot_min_frequency=onehot_min_frequency,
        max_categories_per_col=max_categories_per_col,
        supervised_numeric=False,
    )
    del x_train, x_valid, x_test
    gc.collect()

    reduced, reducer_metadata = _gpu_reduce_and_scale(
        x_gpu,
        svd_components=svd_components,
        random_state=random_state,
    )
    return (
        reduced,
        {
            "backend": "gpu",
            "numeric_cols": num_cols,
            "categorical_cols": cat_cols,
            "skew_cols": skew_cols,
            "flag_cols": flag_cols,
            "categorical_encoding": "frequency_limited_ordinal",
            "onehot_min_frequency_reference": onehot_min_frequency,
            "gpu_max_categories_per_col": max_categories_per_col,
            **encoding_metadata,
            **reducer_metadata,
            "shapes": {split: list(values.shape) for split, values in reduced.items()},
        },
    )


def _preprocess_gpu_for_supervised(
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    *,
    onehot_min_frequency: int,
    max_categories_per_col: int,
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    feature_cols = [col for col in df.columns if col not in SUPERVISED_LEAK_COLUMNS]
    feature_dtypes = df[feature_cols].dtypes
    cat_cols = feature_dtypes[
        feature_dtypes.apply(lambda dtype: pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype))
    ].index.tolist()
    num_cols = [col for col in feature_cols if col not in cat_cols]

    x_train = df.iloc[splits["train"]][feature_cols]
    x_valid = df.iloc[splits["valid"]][feature_cols]
    x_test = df.iloc[splits["test"]][feature_cols]
    x_gpu, encoding_metadata = _gpu_dense_matrices(
        x_train,
        x_valid,
        x_test,
        skew_cols=num_cols,
        flag_cols=[],
        cat_cols=cat_cols,
        onehot_min_frequency=onehot_min_frequency,
        max_categories_per_col=max_categories_per_col,
        supervised_numeric=True,
    )
    del x_train, x_valid, x_test
    gc.collect()
    return (
        x_gpu,
        feature_cols,
        {
            "backend": "gpu",
            "numeric_cols": num_cols,
            "categorical_cols": cat_cols,
            "feature_cols": feature_cols,
            "categorical_encoding": "frequency_limited_ordinal",
            "onehot_min_frequency_reference": onehot_min_frequency,
            "gpu_max_categories_per_col": max_categories_per_col,
            **encoding_metadata,
            "shapes": {split: list(values.shape) for split, values in x_gpu.items()},
        },
    )


def _preprocess_for_clustering(
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    feature_cols: list[str],
    *,
    svd_components: int,
    onehot_min_frequency: int,
    random_state: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    featurizer = PortsConnectivityFeaturizer(topN=20, log_counts=True, drop_raw_ips=True)
    raw_train = df.iloc[splits["train"]][feature_cols]
    x_train = featurizer.fit_transform(raw_train)
    del raw_train
    gc.collect()

    raw_valid = df.iloc[splits["valid"]][feature_cols]
    x_valid = featurizer.transform(raw_valid)
    del raw_valid
    gc.collect()

    raw_test = df.iloc[splits["test"]][feature_cols]
    x_test = featurizer.transform(raw_test)
    del raw_test
    gc.collect()

    num_cols = x_train.select_dtypes(exclude=["object", "string"]).columns.tolist()
    cat_cols = x_train.select_dtypes(include=["object", "string"]).columns.tolist()
    already_logged = {"pair_count", "src_out_count", "dst_in_count", "src_unique_dst", "dst_unique_src"}
    skew_cols = [c for c in num_cols if c not in ["src_port_wk", "dst_port_wk"] and c not in already_logged]
    flag_cols = [c for c in num_cols if c not in skew_cols]

    skew_scaler = _fit_standard_scaler_chunks(x_train, skew_cols, log1p=True)
    flag_scaler = _fit_standard_scaler_chunks(x_train, flag_cols, log1p=False)
    cat_fill_values, cat_encoder = _fit_categorical_encoder(
        x_train,
        cat_cols,
        onehot_min_frequency=onehot_min_frequency,
    )
    x_train_pre = _transform_cluster_frame(
        x_train,
        skew_cols=skew_cols,
        flag_cols=flag_cols,
        cat_cols=cat_cols,
        skew_scaler=skew_scaler,
        flag_scaler=flag_scaler,
        cat_fill_values=cat_fill_values,
        cat_encoder=cat_encoder,
    )
    del x_train
    gc.collect()

    x_valid_pre = _transform_cluster_frame(
        x_valid,
        skew_cols=skew_cols,
        flag_cols=flag_cols,
        cat_cols=cat_cols,
        skew_scaler=skew_scaler,
        flag_scaler=flag_scaler,
        cat_fill_values=cat_fill_values,
        cat_encoder=cat_encoder,
    )
    del x_valid
    gc.collect()

    x_test_pre = _transform_cluster_frame(
        x_test,
        skew_cols=skew_cols,
        flag_cols=flag_cols,
        cat_cols=cat_cols,
        skew_scaler=skew_scaler,
        flag_scaler=flag_scaler,
        cat_fill_values=cat_fill_values,
        cat_encoder=cat_encoder,
    )
    del x_test
    gc.collect()

    max_components = max(0, min(x_train_pre.shape[0] - 1, x_train_pre.shape[1] - 1))
    effective_components = min(svd_components, max_components)
    if effective_components >= 1:
        svd = TruncatedSVD(n_components=effective_components, random_state=random_state)
        scaler = StandardScaler()
        z_train = scaler.fit_transform(svd.fit_transform(x_train_pre))
        z_valid = scaler.transform(svd.transform(x_valid_pre))
        z_test = scaler.transform(svd.transform(x_test_pre))
    else:
        z_train = _to_dense(x_train_pre)
        z_valid = _to_dense(x_valid_pre)
        z_test = _to_dense(x_test_pre)

    return (
        {"train": z_train, "valid": z_valid, "test": z_test},
        {
            "numeric_cols": num_cols,
            "categorical_cols": cat_cols,
            "skew_cols": skew_cols,
            "flag_cols": flag_cols,
            "chunk_rows": _chunk_rows(),
            "svd_components_requested": svd_components,
            "svd_components_effective": effective_components if effective_components >= 1 else None,
            "shapes": {"train": list(z_train.shape), "valid": list(z_valid.shape), "test": list(z_test.shape)},
        },
    )


def _preprocess_for_supervised(
    df: pd.DataFrame,
    splits: dict[str, np.ndarray],
    *,
    onehot_min_frequency: int,
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    feature_cols = [col for col in df.columns if col not in SUPERVISED_LEAK_COLUMNS]
    feature_dtypes = df[feature_cols].dtypes
    cat_cols = feature_dtypes[
        feature_dtypes.apply(lambda dtype: pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype))
    ].index.tolist()
    num_cols = [col for col in feature_cols if col not in cat_cols]

    train_idx = splits["train"]
    num_imputer = (
        SimpleImputer(strategy="median").fit(df.iloc[train_idx][num_cols])
        if num_cols
        else None
    )
    cat_fill_values, cat_encoder = _fit_categorical_encoder(
        df.iloc[train_idx],
        cat_cols,
        onehot_min_frequency=onehot_min_frequency,
    )

    x = {
        split: _transform_supervised_split(
            df,
            idx,
            num_cols=num_cols,
            cat_cols=cat_cols,
            num_imputer=num_imputer,
            cat_fill_values=cat_fill_values,
            cat_encoder=cat_encoder,
        )
        for split, idx in splits.items()
    }
    return (
        x,
        feature_cols,
        {
            "numeric_cols": num_cols,
            "categorical_cols": cat_cols,
            "feature_cols": feature_cols,
            "chunk_rows": _chunk_rows(),
            "shapes": {split: list(values.shape) for split, values in x.items()},
        },
    )


def make_onehot(min_frequency: int | None) -> OneHotEncoder:
    kwargs: dict[str, Any] = {"handle_unknown": "ignore"}
    if min_frequency:
        kwargs["min_frequency"] = min_frequency
    try:
        return OneHotEncoder(**kwargs, sparse_output=True)
    except TypeError:
        return OneHotEncoder(**kwargs, sparse=True)


def _gpu_dense_matrices(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    x_test: pd.DataFrame,
    *,
    skew_cols: list[str],
    flag_cols: list[str],
    cat_cols: list[str],
    onehot_min_frequency: int,
    max_categories_per_col: int,
    supervised_numeric: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import cupy as cp

    parts = {"train": [], "valid": [], "test": []}
    metadata: dict[str, Any] = {"numeric_stats": {}, "categorical_maps": {}}

    if skew_cols:
        train_num, valid_num, test_num, stats = _gpu_numeric_arrays(
            x_train,
            x_valid,
            x_test,
            skew_cols,
            log1p=not supervised_numeric,
            center_and_scale=supervised_numeric,
        )
        parts["train"].append(train_num)
        parts["valid"].append(valid_num)
        parts["test"].append(test_num)
        metadata["numeric_stats"]["skew_or_numeric"] = stats

    if flag_cols:
        train_flags, valid_flags, test_flags, stats = _gpu_numeric_arrays(
            x_train,
            x_valid,
            x_test,
            flag_cols,
            log1p=False,
            center_and_scale=False,
        )
        parts["train"].append(train_flags)
        parts["valid"].append(valid_flags)
        parts["test"].append(test_flags)
        metadata["numeric_stats"]["flags"] = stats

    if cat_cols:
        train_cat, valid_cat, test_cat, cat_maps = _gpu_categorical_code_arrays(
            x_train,
            x_valid,
            x_test,
            cat_cols,
            onehot_min_frequency=onehot_min_frequency,
            max_categories_per_col=max_categories_per_col,
        )
        parts["train"].append(train_cat)
        parts["valid"].append(valid_cat)
        parts["test"].append(test_cat)
        metadata["categorical_maps"] = cat_maps

    if not parts["train"]:
        raise ValueError("No feature columns available for GPU preprocessing.")

    matrices = {}
    for split in ["train", "valid", "test"]:
        matrices[split] = (
            cp.concatenate(parts[split], axis=1).astype(cp.float32)
            if len(parts[split]) > 1
            else parts[split][0].astype(cp.float32)
        )
    return matrices, metadata


def _gpu_numeric_arrays(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    x_test: pd.DataFrame,
    cols: list[str],
    *,
    log1p: bool,
    center_and_scale: bool,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    import cupy as cp

    train = _numeric_frame(x_train, cols, log1p=log1p)
    valid = _numeric_frame(x_valid, cols, log1p=log1p)
    test = _numeric_frame(x_test, cols, log1p=log1p)

    medians = train.median(numeric_only=True).fillna(0.0)
    train = train.fillna(medians)
    valid = valid.fillna(medians)
    test = test.fillna(medians)

    if center_and_scale:
        means = train.mean(numeric_only=True).fillna(0.0)
    else:
        means = pd.Series(0.0, index=cols)
    stds = train.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)

    train_arr = cp.asarray(((train - means) / stds).to_numpy(dtype=np.float32), dtype=cp.float32)
    valid_arr = cp.asarray(((valid - means) / stds).to_numpy(dtype=np.float32), dtype=cp.float32)
    test_arr = cp.asarray(((test - means) / stds).to_numpy(dtype=np.float32), dtype=cp.float32)
    stats = {
        "cols": cols,
        "log1p": log1p,
        "center_and_scale": center_and_scale,
        "median": {k: float(v) for k, v in medians.items()},
        "mean": {k: float(v) for k, v in means.items()},
        "std": {k: float(v) for k, v in stds.items()},
    }
    return train_arr, valid_arr, test_arr, stats


def _numeric_frame(frame: pd.DataFrame, cols: list[str], *, log1p: bool) -> pd.DataFrame:
    values = frame.loc[:, cols].apply(pd.to_numeric, errors="coerce").astype("float32")
    if log1p:
        with np.errstate(invalid="ignore", divide="ignore"):
            values = np.log1p(values)
    return values


def _gpu_categorical_code_arrays(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    x_test: pd.DataFrame,
    cat_cols: list[str],
    *,
    onehot_min_frequency: int,
    max_categories_per_col: int,
) -> tuple[Any, Any, Any, dict[str, Any]]:
    import cupy as cp

    min_frequency = max(1, int(onehot_min_frequency or 1))
    max_categories = max(1, int(max_categories_per_col or 64))
    train_arrays = []
    valid_arrays = []
    test_arrays = []
    maps: dict[str, Any] = {}

    for col in cat_cols:
        train_s = _normalize_category_series(x_train[col])
        mode = _category_mode(train_s)
        filled_train = train_s.fillna(mode)
        counts = filled_train.value_counts(dropna=False)
        kept_index = counts[counts >= min_frequency].head(max_categories).index
        keep_values = [str(value) for value in kept_index]
        if not keep_values:
            keep_values = [str(mode)]
        code_map = {value: code + 1 for code, value in enumerate(keep_values)}

        train_codes = _category_codes(train_s, mode, code_map)
        valid_codes = _category_codes(_normalize_category_series(x_valid[col]), mode, code_map)
        test_codes = _category_codes(_normalize_category_series(x_test[col]), mode, code_map)

        mean = float(train_codes.mean())
        std = float(train_codes.std(ddof=0))
        if std == 0.0 or np.isnan(std):
            std = 1.0

        train_arrays.append(cp.asarray(((train_codes - mean) / std).to_numpy(dtype=np.float32))[:, None])
        valid_arrays.append(cp.asarray(((valid_codes - mean) / std).to_numpy(dtype=np.float32))[:, None])
        test_arrays.append(cp.asarray(((test_codes - mean) / std).to_numpy(dtype=np.float32))[:, None])
        maps[col] = {
            "mode": str(mode),
            "n_kept": int(len(code_map)),
            "min_frequency": int(min_frequency),
            "max_categories": int(max_categories),
            "unknown_code": 0,
            "mean": mean,
            "std": std,
            "kept_values": keep_values,
        }

    return (
        cp.concatenate(train_arrays, axis=1).astype(cp.float32),
        cp.concatenate(valid_arrays, axis=1).astype(cp.float32),
        cp.concatenate(test_arrays, axis=1).astype(cp.float32),
        maps,
    )


def _gpu_reduce_and_scale(
    x_gpu: dict[str, Any],
    *,
    svd_components: int,
    random_state: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import cupy as cp

    requested = int(svd_components or 0)
    max_components = max(0, min(x_gpu["train"].shape[0] - 1, x_gpu["train"].shape[1] - 1))
    n_components = min(requested, max_components)
    reducer_name = None
    if n_components >= 1:
        reducer, reducer_name = _make_gpu_reducer(n_components, random_state)
        z_train = cp.asarray(reducer.fit_transform(x_gpu["train"]), dtype=cp.float32)
        z_valid = cp.asarray(reducer.transform(x_gpu["valid"]), dtype=cp.float32)
        z_test = cp.asarray(reducer.transform(x_gpu["test"]), dtype=cp.float32)
    else:
        z_train = x_gpu["train"]
        z_valid = x_gpu["valid"]
        z_test = x_gpu["test"]

    mean = cp.nanmean(z_train, axis=0)
    std = cp.nanstd(z_train, axis=0)
    std = cp.where((std == 0) | cp.isnan(std), 1.0, std)
    reduced = {
        "train": cp.nan_to_num((z_train - mean) / std).astype(cp.float32),
        "valid": cp.nan_to_num((z_valid - mean) / std).astype(cp.float32),
        "test": cp.nan_to_num((z_test - mean) / std).astype(cp.float32),
    }
    return (
        reduced,
        {
            "reducer": reducer_name,
            "svd_components_requested": svd_components,
            "svd_components_effective": int(n_components) if n_components >= 1 else None,
        },
    )


def _make_gpu_reducer(n_components: int, random_state: int) -> tuple[Any, str]:
    try:
        from cuml.decomposition import TruncatedSVD as CuMLTruncatedSVD

        return CuMLTruncatedSVD(n_components=n_components, random_state=random_state), "cuML TruncatedSVD"
    except Exception:
        from cuml.decomposition import PCA

        return PCA(n_components=n_components, random_state=random_state), "cuML PCA"


def _normalize_category_series(series: pd.Series) -> pd.Series:
    out = series.astype("object").copy()
    missing = out.isna()
    out.loc[~missing] = out.loc[~missing].map(str)
    out.loc[missing] = np.nan
    return out


def _category_mode(series: pd.Series) -> str:
    modes = series.dropna().mode()
    if modes.empty:
        return "__MISSING__"
    return str(modes.iloc[0])


def _category_codes(series: pd.Series, mode: str, code_map: dict[str, int]) -> pd.Series:
    filled = series.fillna(mode).map(str)
    return filled.map(code_map).fillna(0).astype("float32")


def _chunk_rows() -> int:
    raw = os.environ.get("ERAD_PREPROCESS_CHUNK_ROWS", "500000")
    try:
        return max(10000, int(raw))
    except ValueError:
        return 500000


def _fit_standard_scaler_chunks(
    frame: pd.DataFrame,
    cols: list[str],
    *,
    log1p: bool,
) -> StandardScaler | None:
    if not cols:
        return None
    scaler = StandardScaler(with_mean=False)
    chunk_rows = _chunk_rows()
    for start in range(0, len(frame), chunk_rows):
        values = _numeric_values(frame.iloc[start : start + chunk_rows], cols, log1p=log1p)
        scaler.partial_fit(values)
    return scaler


def _fit_categorical_encoder(
    frame: pd.DataFrame,
    cat_cols: list[str],
    *,
    onehot_min_frequency: int,
) -> tuple[dict[str, object], OneHotEncoder | None]:
    if not cat_cols:
        return {}, None

    fill_values = _categorical_fill_values(frame, cat_cols)
    train_cat = _normalize_categorical_frame(frame, cat_cols, fill_values)
    encoder = make_onehot(onehot_min_frequency)
    encoder.fit(train_cat)
    del train_cat
    gc.collect()
    return fill_values, encoder


def _transform_cluster_frame(
    frame: pd.DataFrame,
    *,
    skew_cols: list[str],
    flag_cols: list[str],
    cat_cols: list[str],
    skew_scaler: StandardScaler | None,
    flag_scaler: StandardScaler | None,
    cat_fill_values: dict[str, object],
    cat_encoder: OneHotEncoder | None,
) -> sparse.csr_matrix:
    chunks: list[sparse.csr_matrix] = []
    chunk_rows = _chunk_rows()
    for start in range(0, len(frame), chunk_rows):
        chunk = frame.iloc[start : start + chunk_rows]
        parts: list[sparse.csr_matrix] = []
        if skew_cols and skew_scaler is not None:
            values = skew_scaler.transform(_numeric_values(chunk, skew_cols, log1p=True))
            parts.append(sparse.csr_matrix(values))
        if flag_cols and flag_scaler is not None:
            values = flag_scaler.transform(_numeric_values(chunk, flag_cols, log1p=False))
            parts.append(sparse.csr_matrix(values))
        if cat_cols and cat_encoder is not None:
            parts.append(_transform_categorical_chunk(chunk, cat_cols, cat_fill_values, cat_encoder))
        chunks.append(_hstack_parts(parts, len(chunk)))
    return sparse.vstack(chunks, format="csr")


def _transform_supervised_split(
    df: pd.DataFrame,
    idx: np.ndarray,
    *,
    num_cols: list[str],
    cat_cols: list[str],
    num_imputer: SimpleImputer | None,
    cat_fill_values: dict[str, object],
    cat_encoder: OneHotEncoder | None,
) -> sparse.csr_matrix:
    chunks: list[sparse.csr_matrix] = []
    chunk_rows = _chunk_rows()
    for start in range(0, len(idx), chunk_rows):
        chunk_idx = idx[start : start + chunk_rows]
        parts: list[sparse.csr_matrix] = []
        if num_cols and num_imputer is not None:
            values = num_imputer.transform(df.iloc[chunk_idx][num_cols])
            parts.append(sparse.csr_matrix(values))
        if cat_cols and cat_encoder is not None:
            parts.append(_transform_categorical_chunk(df.iloc[chunk_idx], cat_cols, cat_fill_values, cat_encoder))
        chunks.append(_hstack_parts(parts, len(chunk_idx)))
    return sparse.vstack(chunks, format="csr")


def _numeric_values(frame: pd.DataFrame, cols: list[str], *, log1p: bool) -> np.ndarray:
    values = frame.loc[:, cols].to_numpy(dtype=np.float64, na_value=np.nan)
    if log1p:
        values = np.log1p(values)
    return values


def _categorical_fill_values(frame: pd.DataFrame, cat_cols: list[str]) -> dict[str, object]:
    fill_values: dict[str, object] = {}
    for col in cat_cols:
        series = frame[col]
        non_missing = series[~series.isna()]
        if len(non_missing) == 0:
            fill_values[col] = "NA"
            continue
        mode = non_missing.astype(str).mode(dropna=True)
        fill_values[col] = mode.iloc[0] if len(mode) else "NA"
    return fill_values


def _normalize_categorical_frame(
    frame: pd.DataFrame,
    cat_cols: list[str],
    fill_values: dict[str, object],
) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for col in cat_cols:
        series = frame[col]
        missing = series.isna()
        normalized = series.astype("object")
        if (~missing).any():
            normalized.loc[~missing] = normalized.loc[~missing].map(str)
        normalized.loc[missing] = fill_values[col]
        out[col] = normalized
    return out


def _transform_categorical_chunk(
    frame: pd.DataFrame,
    cat_cols: list[str],
    fill_values: dict[str, object],
    cat_encoder: OneHotEncoder,
) -> sparse.csr_matrix:
    normalized = _normalize_categorical_frame(frame, cat_cols, fill_values)
    encoded = cat_encoder.transform(normalized)
    del normalized
    return encoded.tocsr()


def _hstack_parts(parts: list[sparse.csr_matrix], n_rows: int) -> sparse.csr_matrix:
    if not parts:
        return sparse.csr_matrix((n_rows, 0))
    if len(parts) == 1:
        return parts[0].tocsr()
    return sparse.hstack(parts, format="csr")


def categoricals_to_string(X: Any) -> Any:
    if isinstance(X, pd.DataFrame):
        for col in X.columns:
            series = X[col]
            missing = series.isna()
            if missing.any() or pd.api.types.is_string_dtype(series.dtype):
                normalized = series.astype("object")
                normalized.loc[missing] = np.nan
                X[col] = normalized
        return X

    arr = np.asarray(X, dtype=object)
    missing = pd.isna(arr)
    if missing.any():
        arr = arr.copy()
        arr[missing] = np.nan
    return arr


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required by PortsConnectivityFeaturizer: {missing}")


def _to_dense(values: Any) -> np.ndarray:
    return values.toarray() if hasattr(values, "toarray") else np.asarray(values)
