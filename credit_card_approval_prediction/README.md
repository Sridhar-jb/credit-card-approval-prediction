# Credit Card Approval Prediction

This hands-on machine learning project builds an intelligent web application that predicts whether a credit card application should be approved or rejected. It covers environment setup, dataset understanding, exploratory data analysis, preprocessing, feature engineering, model training, model evaluation, Flask integration, and IBM Cloud deployment support.

## Features

- Logistic Regression, Decision Tree, Random Forest, and XGBoost/Gradient Boosting training
- Matplotlib and Seaborn exploratory data analysis
- Automatic preprocessing for categorical and numeric applicant data
- Binary risk label handling from payment status or approval columns
- Accuracy score, confusion matrix, and classification report outputs
- Saved best model pipeline with `joblib`
- Flask web app for single applicant eligibility checks
- Batch prediction endpoint for compliance review workflows
- IBM Watson Machine Learning deployment helper

## Project Structure

```text
credit_card_approval_prediction/
  app.py
  train_model.py
  eda_visualization.py
  ibm_watson_deploy.py
  requirements.txt
  src/
    data.py
    model.py
    schemas.py
  templates/
    index.html
    result.html
  static/
    styles.css
  models/
    .gitkeep
  reports/
    generated after EDA/training
  data/
    .gitkeep
```

## Quick Start

Use Python 3.11 or 3.12 for this project. Some data science packages may not have ready-to-install Windows wheels for newer Python versions, which can cause compiler errors during installation.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python eda_visualization.py
python train_model.py
python app.py
```

Open `http://127.0.0.1:5000` in a browser.

## 1. Environment Setup & Package Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Core packages include NumPy, Pandas, Scikit-learn, XGBoost, Flask, and Joblib.

Install optional EDA and cloud packages only after the core app is working:

```bash
pip install -r requirements-eda.txt
pip install -r requirements-cloud.txt
```

The EDA package file includes Matplotlib and Seaborn. The cloud package file includes IBM Watson Machine Learning.

If package installation fails with a compiler error on Windows, check your Python version:

```bash
python --version
```

Install Python 3.11 or 3.12 from python.org, then create a fresh virtual environment before installing requirements.

On Windows, if Matplotlib tries to build from source and complains about missing compilers, run the core workflow first with `requirements.txt`, then install `requirements-eda.txt` later.

## 2. Dataset Collection & Understanding

Place a credit card approval dataset CSV in `data/`. Open-source sources commonly include Kaggle credit card approval datasets or UCI-style application records. The project expects applicant profile fields such as gender, income type, annual income, employment duration, education level, loan balance, credit inquiries, and payment history.

If no CSV is supplied, the scripts generate a synthetic banking-style dataset so learners can run the full workflow immediately.

## 3. Data Visualization & Analysis

Generate EDA outputs:

```bash
python eda_visualization.py --data data/credit_card_applications.csv --target approval_status
```

This creates count plots and distribution plots in `reports/figures/`, including approval status, income type, education level, annual income, employment duration, and existing loan balance.

## 4. Data Preprocessing & Feature Engineering

The preprocessing pipeline handles:

- Missing values with median and most-frequent imputers
- Duplicate-safe CSV loading workflows
- Categorical encoding with one-hot encoding
- Numeric feature scaling
- Payment status conversion into binary approval/risk labels

## 5. Machine Learning Model Building

Train and compare models:

```bash
python train_model.py --data data/credit_card_applications.csv --target approval_status
```

The script trains Logistic Regression, Decision Tree, Random Forest, and XGBoost when available. It saves:

- `models/best_credit_card_model.joblib`
- `models/model_metrics.csv`
- `models/classification_reports.txt`
- `models/confusion_matrices.json`

Accepted target values include `approved/rejected`, `yes/no`, `1/0`, and similar binary labels. If your data has payment status fields instead of a direct target, the pipeline can derive a binary high-risk label from columns such as `status`, `payment_status`, `past_due`, or `overdue_count`.

## 6. Building the Flask Web Application

Run the app:

```bash
python app.py
```

The Flask UI includes a home introduction, applicant input form, and prediction result display. The app also exposes `/api/predict` for JSON-based real-time predictions.

## Example Input Fields

- `gender`
- `own_car`
- `own_property`
- `income`
- `income_type`
- `education`
- `family_status`
- `housing_type`
- `employment_years`
- `age`
- `existing_loan_balance`
- `credit_inquiries`
- `past_due_count`

## IBM Watson Machine Learning

Set these environment variables before running the deployment helper:

```bash
set IBM_CLOUD_API_KEY=your_api_key
set IBM_WML_SPACE_ID=your_space_id
set IBM_WML_URL=https://us-south.ml.cloud.ibm.com
python ibm_watson_deploy.py
```

The helper uploads the saved model pipeline and creates an online deployment.
