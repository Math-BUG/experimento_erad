"""Reference train/validation/test split."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DROP_COLUMNS = ["label", "type", "split", "cluster_kmeans30", "cluster_name", "purity_test"]
GROUP_COLUMNS = ["src_ip", "dst_ip", "service", "proto"]


def split_data(
    df: pd.DataFrame,
    *,
    random_state: int = 42,
    validation_random_state: int = 123,
) -> tuple[dict[str, np.ndarray], pd.Series]:
    y_type = df["type"].copy()
    group_cols = [col for col in GROUP_COLUMNS if col in df.columns]
    if group_cols:
        groups = pd.util.hash_pandas_object(df[group_cols].fillna("NA"), index=False).astype("uint64")
    else:
        groups = pd.Series(np.arange(len(df)), index=df.index)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    split_placeholders = np.empty((len(df), 0), dtype=np.int8)
    train_valid_idx, test_idx = next(sgkf.split(split_placeholders, y_type, groups=groups))

    sgkf_valid = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=validation_random_state)
    train_rel, valid_rel = next(
        sgkf_valid.split(
            np.empty((len(train_valid_idx), 0), dtype=np.int8),
            y_type.iloc[train_valid_idx],
            groups=groups.iloc[train_valid_idx],
        )
    )
    train_idx = np.asarray(train_valid_idx)[train_rel]
    valid_idx = np.asarray(train_valid_idx)[valid_rel]

    return {"train": train_idx, "valid": valid_idx, "test": test_idx}, groups


def reference_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in DROP_COLUMNS]
