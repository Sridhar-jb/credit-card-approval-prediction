from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.data import prepare_features
from src.schemas import FEATURE_COLUMNS, MODEL_PATH

app = Flask(__name__)


def load_model():
    path = Path(MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError("Model not found. Run `python train_model.py` first.")
    return joblib.load(path)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    model = load_model()
    applicant = {column: request.form.get(column) for column in FEATURE_COLUMNS}
    features = prepare_features(pd.DataFrame([applicant]))
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1]) if hasattr(model, "predict_proba") else None
    label = "Approved" if prediction == 1 else "Rejected"
    return render_template("result.html", label=label, probability=probability, applicant=applicant)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    model = load_model()
    payload = request.get_json(force=True)
    records = payload if isinstance(payload, list) else [payload]
    features = prepare_features(pd.DataFrame(records))
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1] if hasattr(model, "predict_proba") else [None] * len(records)
    results = [
        {
            "prediction": "approved" if int(prediction) == 1 else "rejected",
            "approval_probability": None if probability is None else round(float(probability), 4),
        }
        for prediction, probability in zip(predictions, probabilities)
    ]
    return jsonify(results if isinstance(payload, list) else results[0])


if __name__ == "__main__":
    app.run(debug=True)

