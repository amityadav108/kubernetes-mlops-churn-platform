import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_mlops.data.validation import load_and_validate_data


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate the JSON training configuration."""
    path = Path(config_path)

    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open(encoding="utf-8") as file:
        config = json.load(file)

    required = {
        "dataset_path",
        "model_path",
        "metrics_path",
        "random_state",
        "test_size",
        "classification_threshold",
        "model_type",
        "max_iter",
    }

    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing configuration keys: {sorted(missing)}")

    if config["model_type"] != "logistic_regression":
        raise ValueError("Only logistic_regression is currently supported.")

    if not 0 < config["test_size"] < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if not 0 <= config["classification_threshold"] <= 1:
        raise ValueError("classification_threshold must be between 0 and 1.")

    if config["max_iter"] < 1:
        raise ValueError("max_iter must be a positive integer.")

    return config


def build_pipeline(
    features: pd.DataFrame,
    max_iter: int,
    random_state: int,
) -> Pipeline:
    """Build preprocessing and classification as one pipeline."""
    numeric_columns = features.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = [
        column
        for column in features.columns
        if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessing = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )


def calculate_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Calculate binary classification metrics."""
    predictions = (probabilities >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).tolist(),
        "classification_threshold": threshold,
    }


def run_training(config_path: str | Path) -> dict[str, Any]:
    """Train, evaluate, and save the churn classification model."""
    config = load_config(config_path)

    features, target = load_and_validate_data(config["dataset_path"])

    if target.nunique() != 2:
        raise ValueError("Training requires both churn target classes.")

    train_features, test_features, train_target, test_target = (
        train_test_split(
            features,
            target,
            test_size=config["test_size"],
            random_state=config["random_state"],
            stratify=target,
        )
    )

    # Convert pandas nullable strings to ordinary objects with np.nan
    # so scikit-learn's imputers can handle missing categorical values.
    categorical_columns = train_features.select_dtypes(
        include=["object", "string"]
    ).columns

    train_features = train_features.copy()
    test_features = test_features.copy()

    for column in categorical_columns:
        train_features[column] = (
            train_features[column].astype(object).where(
                train_features[column].notna(), np.nan
            )
        )
        test_features[column] = (
            test_features[column].astype(object).where(
                test_features[column].notna(), np.nan
            )
        )

    pipeline = build_pipeline(
        train_features,
        max_iter=config["max_iter"],
        random_state=config["random_state"],
    )

    pipeline.fit(train_features, train_target)

    probabilities = pipeline.predict_proba(test_features)[:, 1]

    metrics = calculate_metrics(
        test_target,
        probabilities,
        threshold=config["classification_threshold"],
    )

    metrics["dataset"] = {
        "total_rows": int(len(features)),
        "training_rows": int(len(train_features)),
        "test_rows": int(len(test_features)),
        "input_features": int(features.shape[1]),
        "churn_rate": float(target.mean()),
    }

    metrics["training"] = {
        "model_type": config["model_type"],
        "random_state": config["random_state"],
        "test_size": config["test_size"],
        "max_iter": config["max_iter"],
    }

    model_path = Path(config["model_path"])
    metrics_path = Path(config["metrics_path"])

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "pipeline": pipeline,
        "metadata": {
            "model_type": config["model_type"],
            "feature_columns": features.columns.tolist(),
            "metrics": metrics,
        },
    }

    joblib.dump(model_bundle, model_path)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, allow_nan=False)

    print("Training completed successfully.")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(json.dumps(metrics, indent=2))

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the customer churn model."
    )
    parser.add_argument(
        "--config",
        default="configs/training.json",
        help="Path to the JSON training configuration.",
    )
    args = parser.parse_args()

    run_training(args.config)


if __name__ == "__main__":
    main()
