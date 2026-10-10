
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
InternetOption = Literal["DSL", "Fiber optic", "No"]
PhoneLinesOption = Literal["Yes", "No", "No phone service"]
InternetServiceOption = Literal["Yes", "No", "No internet service"]
ContractOption = Literal["Month-to-month", "One year", "Two year"]
PaymentMethodOption = Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


class ChurnPredictionRequest(BaseModel):
    """Customer attributes required for a churn prediction."""

    model_config = ConfigDict(extra="forbid")

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=100)
    PhoneService: YesNo
    MultipleLines: PhoneLinesOption
    InternetService: InternetOption
    OnlineSecurity: InternetServiceOption
    OnlineBackup: InternetServiceOption
    DeviceProtection: InternetServiceOption
    TechSupport: InternetServiceOption
    StreamingTV: InternetServiceOption
    StreamingMovies: InternetServiceOption
    Contract: ContractOption
    PaperlessBilling: YesNo
    PaymentMethod: PaymentMethodOption
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class ChurnPredictionResponse(BaseModel):
    """Prediction returned by the model-serving API."""

    churn_prediction: Literal["Yes", "No"]
    churn_probability: float = Field(ge=0, le=1)
    classification_threshold: float = Field(ge=0, le=1)


class HealthResponse(BaseModel):
    """Basic model-serving health information."""

    status: Literal["ok"]
    model_loaded: bool
