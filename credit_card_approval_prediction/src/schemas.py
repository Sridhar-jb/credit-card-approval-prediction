FEATURE_COLUMNS = [
    "gender",
    "own_car",
    "own_property",
    "income",
    "income_type",
    "education",
    "family_status",
    "housing_type",
    "employment_years",
    "age",
    "existing_loan_balance",
    "credit_inquiries",
    "past_due_count",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "own_car",
    "own_property",
    "income_type",
    "education",
    "family_status",
    "housing_type",
]

NUMERIC_COLUMNS = [
    "income",
    "employment_years",
    "age",
    "existing_loan_balance",
    "credit_inquiries",
    "past_due_count",
]

MODEL_PATH = "models/best_credit_card_model.joblib"

