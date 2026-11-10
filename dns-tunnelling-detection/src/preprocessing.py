from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def build_scaler() -> Pipeline:
    """Median imputation and standardization, fitted only on benign training observations."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def validate_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        raise ValueError("Feature dataset is empty.")
    numeric = features.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([float("inf"), float("-inf")], pd.NA).fillna(0)
    return numeric
