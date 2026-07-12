from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from src.data import load_training_data
from src.model import train_and_select_model
from src.schemas import MODEL_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train credit card approval classifiers.")
    parser.add_argument("--data", help="Optional CSV training dataset path.")
    parser.add_argument("--target", help="Optional target column name.")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Output path for the best model pipeline.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features, target = load_training_data(args.data, args.target)
    result = train_and_select_model(features, target)

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.best_pipeline, model_path)

    metrics_path = model_path.parent / "model_metrics.csv"
    result.metrics.to_csv(metrics_path, index=False)

    reports_path = model_path.parent / "classification_reports.txt"
    reports_path.write_text(
        "\n\n".join(f"{name}\n{'=' * len(name)}\n{report}" for name, report in result.reports.items()),
        encoding="utf-8",
    )

    confusion_path = model_path.parent / "confusion_matrices.json"
    confusion_path.write_text(json.dumps(result.confusion_matrices, indent=2), encoding="utf-8")

    print("Model comparison:")
    print(result.metrics.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nBest model: {result.best_name}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved classification reports: {reports_path}")
    print(f"Saved confusion matrices: {confusion_path}")


if __name__ == "__main__":
    main()
