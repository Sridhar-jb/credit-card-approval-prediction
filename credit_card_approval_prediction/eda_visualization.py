from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data import load_training_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EDA charts for credit card approval data.")
    parser.add_argument("--data", help="Optional CSV dataset path.")
    parser.add_argument("--target", help="Optional target column name.")
    parser.add_argument("--output-dir", default="reports/figures", help="Directory for generated charts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features, target = load_training_data(args.data, args.target)
    frame = features.copy()
    frame["approval_status"] = target.map({1: "Approved", 0: "Rejected"})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    save_count_plot(frame, "approval_status", output_dir / "approval_status_count.png")
    save_count_plot(frame, "income_type", output_dir / "income_type_count.png", hue="approval_status")
    save_count_plot(frame, "education", output_dir / "education_count.png", hue="approval_status")
    save_distribution_plot(frame, "income", output_dir / "annual_income_distribution.png")
    save_distribution_plot(frame, "employment_years", output_dir / "employment_duration_distribution.png")
    save_distribution_plot(frame, "existing_loan_balance", output_dir / "loan_balance_distribution.png")

    summary_path = output_dir.parent / "eda_summary.csv"
    frame.describe(include="all").transpose().to_csv(summary_path)

    print(f"Saved EDA charts to: {output_dir}")
    print(f"Saved dataset summary to: {summary_path}")


def save_count_plot(frame: pd.DataFrame, column: str, path: Path, hue: str | None = None) -> None:
    plt.figure(figsize=(9, 5))
    sns.countplot(data=frame, x=column, hue=hue, order=frame[column].value_counts().index)
    plt.title(f"{column.replace('_', ' ').title()} Count Plot")
    plt.xlabel(column.replace("_", " ").title())
    plt.ylabel("Count")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_distribution_plot(frame: pd.DataFrame, column: str, path: Path) -> None:
    plt.figure(figsize=(9, 5))
    sns.histplot(data=frame, x=column, hue="approval_status", kde=True, bins=30)
    plt.title(f"{column.replace('_', ' ').title()} Distribution")
    plt.xlabel(column.replace("_", " ").title())
    plt.ylabel("Applicants")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
