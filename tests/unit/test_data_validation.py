import pandas as pd
import pytest

from churn_mlops.data.validation import (
    DataValidationError,
    load_and_validate_data,
)


def valid_customer() -> dict:
    """Return one valid customer record."""
    return {
        "customerID": "TEST-0001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 65.5,
        "TotalCharges": 786.0,
        "Churn": "Yes",
    }


def write_csv(tmp_path, records: list[dict]) -> str:
    path = tmp_path / "customers.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return str(path)


def test_valid_data_returns_features_and_encoded_target(tmp_path):
    path = write_csv(tmp_path, [valid_customer()])

    features, target = load_and_validate_data(path)

    assert features.shape == (1, 19)
    assert "customerID" not in features.columns
    assert "Churn" not in features.columns
    assert target.tolist() == [1]


def test_duplicate_customer_ids_are_rejected(tmp_path):
    customer = valid_customer()
    duplicate = customer.copy()
    duplicate["Churn"] = "No"
    path = write_csv(tmp_path, [customer, duplicate])

    with pytest.raises(DataValidationError, match="Duplicate customer IDs"):
        load_and_validate_data(path)


def test_missing_required_column_is_rejected(tmp_path):
    customer = valid_customer()
    del customer["Contract"]
    path = write_csv(tmp_path, [customer])

    with pytest.raises(DataValidationError, match="Missing required columns"):
        load_and_validate_data(path)


def test_blank_total_charges_for_zero_tenure_becomes_zero(tmp_path):
    customer = valid_customer()
    customer["tenure"] = 0
    customer["TotalCharges"] = ""
    path = write_csv(tmp_path, [customer])

    features, _ = load_and_validate_data(path)

    assert features.loc[0, "TotalCharges"] == 0


def test_missing_total_charges_for_positive_tenure_is_rejected(tmp_path):
    customer = valid_customer()
    customer["TotalCharges"] = ""
    path = write_csv(tmp_path, [customer])

    with pytest.raises(DataValidationError, match="Missing TotalCharges"):
        load_and_validate_data(path)
