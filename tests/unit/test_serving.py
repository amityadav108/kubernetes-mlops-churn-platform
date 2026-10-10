
from fastapi.testclient import TestClient

from churn_mlops.serving.main import app

client = TestClient(app)

VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 75.5,
    "TotalCharges": 906.0,
}


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
    }


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["service"] == "Customer Churn Prediction API"


def test_predict_endpoint():
    response = client.post("/predict", json=VALID_CUSTOMER)

    assert response.status_code == 200
    result = response.json()

    assert result["churn_prediction"] in {"Yes", "No"}
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["classification_threshold"] == 0.5


def test_predict_rejects_invalid_gender():
    customer = VALID_CUSTOMER.copy()
    customer["gender"] = "Unknown"

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


def test_predict_rejects_unexpected_fields():
    customer = VALID_CUSTOMER.copy()
    customer["unexpected_field"] = "invalid"

    response = client.post("/predict", json=customer)

    assert response.status_code == 422
