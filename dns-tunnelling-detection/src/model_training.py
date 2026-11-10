from __future__ import annotations

from pathlib import Path
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from src.preprocessing import build_scaler, validate_feature_frame


def train_models(benign_features, contamination: float = 0.05, nu: float = 0.05):
    """Fit both anomaly detectors exclusively on the supplied benign feature matrix."""
    X = validate_feature_frame(benign_features)
    if len(X) < 10:
        raise ValueError("At least 10 benign observations are required.")
    scaler = build_scaler()
    X_scaled = scaler.fit_transform(X)
    return {
        "Isolation Forest": {
            "estimator": IsolationForest(
                n_estimators=200,
                contamination=contamination,
                random_state=42,
                n_jobs=-1,
            ).fit(X_scaled),
            "scaler": scaler,
            "feature_columns": list(X.columns),
        },
        "One-Class SVM": {
            "estimator": OneClassSVM(kernel="rbf", gamma="scale", nu=nu).fit(X_scaled),
            "scaler": scaler,
            "feature_columns": list(X.columns),
        },
    }


def save_models(models, directory: str | Path):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, bundle in models.items():
        filename = "isolation_forest.joblib" if name == "Isolation Forest" else "one_class_svm.joblib"
        joblib.dump(bundle, directory / filename)


def load_model(path: str | Path):
    return joblib.load(path)
