
from pathlib import Path

import pandas as pd


class DataValidationError(ValueError):
    """Raised when the input dataset fails validation."""


REQUIRED_COLUMNS = {
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
}


def load_and_validate_data(
    dataset_path: str | Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load, validate, and split the customer churn dataset."""
    path = Path(dataset_path)

    if not path.exists() or not path.is_file():
        raise DataValidationError(f"Dataset file not found: {path}")

    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataValidationError(
            f"Could not read dataset file: {path}"
        ) from exc

    if df.empty:
        raise DataValidationError("Dataset is empty.")

    df.columns = df.columns.str.strip()

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise DataValidationError(f"Missing required columns: {missing}")

    df = df.copy()
    df["customerID"] = df["customerID"].astype("string").str.strip()

    if df["customerID"].isna().any() or df["customerID"].eq("").any():
        raise DataValidationError("Customer IDs cannot be blank.")

    if df["customerID"].duplicated().any():
        raise DataValidationError("Duplicate customer IDs were found.")

    numeric_columns = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]

    df["TotalCharges"] = df["TotalCharges"].replace(
        r"^\s*$", pd.NA, regex=True
    )

    for col in numeric_columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
        except (ValueError, TypeError) as exc:
            raise DataValidationError(
                f"Column {col} contains non-numeric values."
            ) from exc

        if col == "TotalCharges":
            zero_tenure = df["tenure"].eq(0)
            blank_charges = df[col].isna()
            invalid_blanks = blank_charges & ~zero_tenure

            if invalid_blanks.any():
                raise DataValidationError(
                    "Missing TotalCharges for customers with positive tenure."
                )

            df.loc[blank_charges & zero_tenure, col] = 0

        if df[col].isna().any():
            raise DataValidationError(
                f"Column {col} contains missing or invalid values."
            )

    if (df["tenure"] < 0).any():
        raise DataValidationError("Tenure cannot be negative.")

    if (df["MonthlyCharges"] < 0).any():
        raise DataValidationError("MonthlyCharges cannot be negative.")

    if (df["TotalCharges"] < 0).any():
        raise DataValidationError("TotalCharges cannot be negative.")

    for col in ("SeniorCitizen", "tenure"):
        if not df[col].mod(1).eq(0).all():
            raise DataValidationError(
                f"Column {col} must contain whole numbers."
            )

    df["Churn"] = df["Churn"].astype("string").str.strip()

    if df["Churn"].isna().any():
        raise DataValidationError("Churn target contains missing values.")

    invalid_targets = set(df["Churn"].unique()) - {"Yes", "No"}
    if invalid_targets:
        invalid = ", ".join(sorted(invalid_targets))
        raise DataValidationError(f"Invalid Churn values: {invalid}")

    y = df["Churn"].map({"No": 0, "Yes": 1}).astype("int64")
    X = df.drop(columns=["customerID", "Churn"])

    if X.isna().any().any():
        raise DataValidationError("Features contain missing values.")

    return X, y
