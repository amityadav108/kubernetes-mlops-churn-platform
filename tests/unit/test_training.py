import json

import numpy as np

from churn_mlops.training.train import (
    calculate_metrics,
    load_config,
)


def test_calculate_metrics():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.7, 0.9])

    metrics = calculate_metrics(
        y_true,
        probabilities,
        threshold=0.5,
    )

    assert metrics["accuracy"] == 0.75
    assert abs(metrics["precision"] - (2 / 3)) < 1e-9
    assert metrics["recall"] == 1.0
    assert abs(metrics["f1_score"] - 0.8) < 1e-9
    assert metrics["roc_auc"] == 0.75
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]


def test_load_config(tmp_path):
    config = {
        "dataset_path": "data/raw/Telco-Customer-Churn.csv",
        "model_path": "models/churn_model.joblib",
        "metrics_path": "artifacts/metrics.json",
        "random_state": 42,
        "test_size": 0.2,
        "classification_threshold": 0.5,
        "model_type": "logistic_regression",
        "max_iter": 1000,
    }

    config_path = tmp_path / "training.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = load_config(config_path)

    assert loaded == config
