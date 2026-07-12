from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .schemas import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


@dataclass
class TrainingResult:
    best_name: str
    best_pipeline: Pipeline
    metrics: pd.DataFrame
    reports: dict[str, str]
    confusion_matrices: dict[str, list[list[int]]]


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )


def get_classifiers() -> dict[str, object]:
    classifiers: dict[str, object] = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=180,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    try:
        from xgboost import XGBClassifier

        classifiers["XGBoost"] = XGBClassifier(
            n_estimators=160,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )
    except Exception:
        pass
    return classifiers


def train_and_select_model(features: pd.DataFrame, target: pd.Series) -> TrainingResult:
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    rows = []
    best_name = ""
    best_pipeline: Pipeline | None = None
    best_score = -1.0
    reports: dict[str, str] = {}
    confusion_matrices: dict[str, list[list[int]]] = {}

    for name, classifier in get_classifiers().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        probabilities = _positive_probabilities(pipeline, x_test)

        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probabilities),
        }
        rows.append(metrics)
        reports[name] = classification_report(y_test, predictions, target_names=["Rejected", "Approved"])
        confusion_matrices[name] = confusion_matrix(y_test, predictions).tolist()

        if metrics["f1"] > best_score:
            best_score = metrics["f1"]
            best_name = name
            best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError("No model could be trained.")

    metrics_frame = pd.DataFrame(rows).sort_values(["f1", "roc_auc"], ascending=False)
    return TrainingResult(
        best_name=best_name,
        best_pipeline=best_pipeline,
        metrics=metrics_frame,
        reports=reports,
        confusion_matrices=confusion_matrices,
    )


def _positive_probabilities(pipeline: Pipeline, features: pd.DataFrame):
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        return pipeline.predict_proba(features)[:, 1]
    return pipeline.decision_function(features)
