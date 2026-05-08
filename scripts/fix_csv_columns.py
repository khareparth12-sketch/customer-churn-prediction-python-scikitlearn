# run once from project root — fixes the CSV in place
import pandas as pd

df = pd.read_csv("data/processed/telco_final_processed.csv")

rename_map = {
    "MultipleLines_No phone service": "MultipleLines_No_phone_service",
    "InternetService_Fiber optic":    "InternetService_Fiber_optic",
    "Contract_One year":              "Contract_One_year",
    "Contract_Two year":              "Contract_Two_year",
    "PaymentMethod_Credit card (automatic)": "PaymentMethod_Credit_card_automatic",
    "PaymentMethod_Electronic check": "PaymentMethod_Electronic_check",
    "PaymentMethod_Mailed check":     "PaymentMethod_Mailed_check",
}

df.rename(columns=rename_map, inplace=True)
df.to_csv("data/processed/telco_final_processed.csv", index=False)
print("Done. Columns:", df.columns.tolist())