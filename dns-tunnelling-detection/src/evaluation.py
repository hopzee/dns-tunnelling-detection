from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from src.preprocessing import validate_feature_frame


def evaluate_model(bundle, features, labels):
    """Evaluate predictions against held-out labels; labels are never passed to model fitting."""
    X = validate_feature_frame(features)
    expected = bundle["feature_columns"]
    X = X.reindex(columns=expected, fill_value=0)
    X_scaled = bundle["scaler"].transform(X)
    raw = bundle["estimator"].predict(X_scaled)
    predictions = (raw == -1).astype(int)  # anomaly=malicious
    y = np.asarray(labels, dtype=int)
    if len(y) != len(predictions):
        raise ValueError("Evaluation labels and feature rows must have the same length.")
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "predictions": predictions.tolist(),
    }
