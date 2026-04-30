# api/schema.py
from pydantic import BaseModel, Field, model_validator
#from typing import Self


class CustomerData(BaseModel):
    model_config = {"strict": True}

    # Binary flags — must be 0 or 1
    gender: int = Field(ge=0, le=1)
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: int = Field(ge=0, le=1)
    Dependents: int = Field(ge=0, le=1)
    PhoneService: int = Field(ge=0, le=1)
    PaperlessBilling: int = Field(ge=0, le=1)

    # Numeric — bounded
    tenure: int = Field(ge=0, le=72)
    MonthlyCharges: float = Field(ge=0.0, le=200.0)
    TotalCharges: float = Field(ge=0.0, le=10000.0)

    # One-hot encoded features
    MultipleLines_No_phone_service: int = Field(default=0, ge=0, le=1)
    MultipleLines_Yes: int = Field(default=0, ge=0, le=1)

    InternetService_Fiber_optic: int = Field(default=0, ge=0, le=1)
    InternetService_No: int = Field(default=0, ge=0, le=1)

    OnlineSecurity_Yes: int = Field(default=0, ge=0, le=1)
    OnlineBackup_Yes: int = Field(default=0, ge=0, le=1)
    DeviceProtection_Yes: int = Field(default=0, ge=0, le=1)
    TechSupport_Yes: int = Field(default=0, ge=0, le=1)

    StreamingTV_Yes: int = Field(default=0, ge=0, le=1)
    StreamingMovies_Yes: int = Field(default=0, ge=0, le=1)

    Contract_One_year: int = Field(default=0, ge=0, le=1)
    Contract_Two_year: int = Field(default=0, ge=0, le=1)

    PaymentMethod_Credit_card_automatic: int = Field(default=0, ge=0, le=1)
    PaymentMethod_Electronic_check: int = Field(default=0, ge=0, le=1)
    PaymentMethod_Mailed_check: int = Field(default=0, ge=0, le=1)

    @model_validator(mode="after")
    def check_total_charges_floor(self) -> "CustomerData":
        if self.tenure > 0:
            floor = self.tenure * self.MonthlyCharges * 0.1
            if self.TotalCharges < floor:
                raise ValueError(
                    f"TotalCharges {self.TotalCharges} too low. "
                    f"Expected >= {floor:.2f} given tenure={self.tenure}, "
                    f"MonthlyCharges={self.MonthlyCharges}"
                )
        return self