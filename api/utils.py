# api/utils.py

FEATURE_COLUMNS = [
    'gender',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'tenure',
    'PhoneService',
    'PaperlessBilling',
    'MonthlyCharges',
    'TotalCharges',
    'MultipleLines_No_phone_service',
    'MultipleLines_Yes',
    'InternetService_Fiber_optic',
    'InternetService_No',
    'OnlineSecurity_Yes',
    'OnlineBackup_Yes',
    'DeviceProtection_Yes',
    'TechSupport_Yes',
    'StreamingTV_Yes',
    'StreamingMovies_Yes',
    'Contract_One_year',
    'Contract_Two_year',
    'PaymentMethod_Credit_card_automatic',
    'PaymentMethod_Electronic_check',
    'PaymentMethod_Mailed_check',
]

NUMERIC_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]