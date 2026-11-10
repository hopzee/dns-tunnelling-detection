from __future__ import annotations

import pandas as pd

from src.preprocessing import validate_feature_frame


def score_records(bundle, features: pd.DataFrame) -> pd.DataFrame:
    X = validate_feature_frame(features).reindex(columns=bundle["feature_columns"], fill_value=0)
    scaled = bundle["scaler"].transform(X)
    estimator = bundle["estimator"]
    raw_prediction = estimator.predict(scaled)
    decision = estimator.decision_function(scaled)
    return pd.DataFrame({
        "prediction": ["Anomaly" if value == -1 else "Normal" for value in raw_prediction],
        "anomaly_score": (-decision).astype(float),
    })
