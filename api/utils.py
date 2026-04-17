#api\utils.py

FEATURE_COLUMNS = ['gender',
            'SeniorCitizen',
            'Partner',
            'Dependents',
            'tenure',
            'PhoneService',
            'PaperlessBilling',
            'MonthlyCharges',
            'TotalCharges',
            'MultipleLines_No phone service',
            'MultipleLines_Yes',
            'InternetService_Fiber optic',
            'InternetService_No',
            'OnlineSecurity_Yes',
            'OnlineBackup_Yes',
            'DeviceProtection_Yes',
            'TechSupport_Yes',
            'StreamingTV_Yes',
            'StreamingMovies_Yes',
            'Contract_One year',
            'Contract_Two year',
            'PaymentMethod_Credit card (automatic)',
            'PaymentMethod_Electronic check',
            'PaymentMethod_Mailed check'
    ]

NUMERIC_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]