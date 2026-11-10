import pandas as pd
from src.model_training import train_models
from src.evaluation import evaluate_model


def test_models_train_on_benign_and_evaluate_held_out():
    benign = pd.DataFrame({
        "domain_length": [12, 13, 14, 12, 15, 13, 14, 12, 13, 15, 14, 13],
        "entropy": [2.0, 2.1, 2.0, 2.2, 2.1, 2.0, 2.2, 2.1, 2.0, 2.2, 2.1, 2.0],
    })
    models = train_models(benign)
    held_out = pd.DataFrame({
        "domain_length": [13, 14, 50, 55],
        "entropy": [2.0, 2.1, 4.2, 4.5],
    })
    labels = [0, 0, 1, 1]
    for model in models.values():
        result = evaluate_model(model, held_out, labels)
        assert set(result) >= {"recall", "precision", "f1", "fpr", "confusion_matrix"}
        assert 0 <= result["fpr"] <= 1
