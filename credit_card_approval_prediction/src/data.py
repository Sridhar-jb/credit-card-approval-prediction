from __future__ import annotations

import numpy as np
import pandas as pd

from .schemas import FEATURE_COLUMNS, NUMERIC_COLUMNS


POSITIVE_VALUES = {"approved", "approve", "yes", "y", "true", "1", 1, True}
NEGATIVE_VALUES = {"rejected", "reject", "no", "n", "false", "0", 0, False}
RISK_STATUS_VALUES = {"1", "2", "3", "4", "5", "c", "x", "bad", "late", "past_due", "default"}


def load_training_data(data_path: str | None, target_column: str | None) -> tuple[pd.DataFrame, pd.Series]:
    if data_path:
        frame = pd.read_csv(data_path)
    else:
        frame = make_synthetic_credit_data()

    frame = frame.drop_duplicates().reset_index(drop=True)
    target = derive_target(frame, target_column)
    features = prepare_features(frame)
    return features, target


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.copy()
    for column in FEATURE_COLUMNS:
        if column not in features:
            features[column] = np.nan

    features = features[FEATURE_COLUMNS]
    for column in NUMERIC_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    return features


def derive_target(frame: pd.DataFrame, target_column: str | None) -> pd.Series:
    if target_column and target_column in frame:
        return normalize_binary_target(frame[target_column])

    common_targets = ["approval_status", "approved", "target", "label", "class"]
    for column in common_targets:
        if column in frame:
            return normalize_binary_target(frame[column])

    status_columns = [
        column
        for column in frame.columns
        if column.lower() in {"status", "payment_status", "past_due", "loan_status", "overdue_count"}
    ]
    if status_columns:
        risky = frame[status_columns].astype(str).apply(
            lambda row: any(value.strip().lower() in RISK_STATUS_VALUES for value in row),
            axis=1,
        )
        return (~risky).astype(int)

    if "past_due_count" in frame:
        return (pd.to_numeric(frame["past_due_count"], errors="coerce").fillna(0) == 0).astype(int)

    raise ValueError("No target column or payment status field was found.")


def normalize_binary_target(series: pd.Series) -> pd.Series:
    mapped = series.map(lambda value: _map_binary(value))
    if mapped.isna().any():
        bad_values = sorted(series[mapped.isna()].astype(str).unique())
        raise ValueError(f"Target contains unsupported values: {bad_values}")
    return mapped.astype(int)


def _map_binary(value: object) -> int | float:
    normalized = str(value).strip().lower()
    if value in POSITIVE_VALUES or normalized in POSITIVE_VALUES:
        return 1
    if value in NEGATIVE_VALUES or normalized in NEGATIVE_VALUES:
        return 0
    return np.nan


def make_synthetic_credit_data(rows: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(
        {
            "gender": rng.choice(["Male", "Female"], rows),
            "own_car": rng.choice(["Yes", "No"], rows, p=[0.42, 0.58]),
            "own_property": rng.choice(["Yes", "No"], rows, p=[0.65, 0.35]),
            "income": rng.normal(72000, 28000, rows).clip(15000, 240000).round(0),
            "income_type": rng.choice(["Working", "Commercial associate", "Pensioner", "State servant"], rows),
            "education": rng.choice(["Secondary", "Higher education", "Incomplete higher", "Academic degree"], rows),
            "family_status": rng.choice(["Married", "Single", "Civil marriage", "Separated", "Widow"], rows),
            "housing_type": rng.choice(["House", "Apartment", "Rented apartment", "With parents"], rows),
            "employment_years": rng.gamma(4, 2, rows).clip(0, 35).round(1),
            "age": rng.normal(41, 11, rows).clip(18, 72).round(0),
            "existing_loan_balance": rng.gamma(2.4, 9000, rows).round(0),
            "credit_inquiries": rng.poisson(1.4, rows).clip(0, 10),
            "past_due_count": rng.poisson(0.35, rows).clip(0, 6),
        }
    )

    debt_ratio = data["existing_loan_balance"] / data["income"]
    score = (
        1.8
        - 2.4 * debt_ratio
        - 0.45 * data["credit_inquiries"]
        - 1.25 * data["past_due_count"]
        + 0.025 * data["employment_years"]
        + 0.35 * (data["own_property"] == "Yes").astype(int)
        + rng.normal(0, 0.45, rows)
    )
    probability = 1 / (1 + np.exp(-score))
    data["approval_status"] = np.where(rng.random(rows) < probability, "approved", "rejected")
    return data
