# ChurnGuard — Customer Churn Prediction

A machine learning web application that predicts customer churn using a Random Forest classifier. Built with FastAPI and scikit-learn, featuring a dark-themed UI for entering customer data and receiving real-time churn risk analysis.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Model](#model)
- [Features (Input Variables)](#features-input-variables)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [How It Works](#how-it-works)
- [Notebook Walkthrough](#notebook-walkthrough)
- [Screenshots](#screenshots)

## Overview

Customer churn — when a customer stops using a service — is a critical business problem. ChurnGuard uses a trained Random Forest model to predict whether a customer will churn based on 19 attributes spanning demographics, account details, service subscriptions, and billing information.

The app serves a single-page web form where users input customer attributes and receive:

- A binary prediction: **Will Churn** or **Will Stay**
- Churn probability percentage
- Retention likelihood percentage
- Recommended retention actions (when churn risk is high)

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   Browser (UI)                   │
│  index.html + style.css (Jinja2 templates)       │
└──────────────────────┬──────────────────────────┘
                       │ POST /predict (form data)
                       ▼
┌─────────────────────────────────────────────────┐
│              FastAPI Application                  │
│  main.py — Uvicorn ASGI server on :8001          │
│                                                   │
│  ┌─────────────┐   ┌──────────────────────────┐ │
│  │  / (GET)     │   │  /predict (POST)         │ │
│  │  Render form │   │  Encode features →       │ │
│  │              │   │  Predict via model →     │ │
│  │              │   │  Return result + form    │ │
│  └─────────────┘   └──────────┬───────────────┘ │
│                                │                  │
│                    ┌───────────▼───────────────┐ │
│                    │  final_model.joblib        │ │
│                    │  RandomForestClassifier    │ │
│                    │  (300 trees, depth=10)     │ │
│                    └───────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

## Dataset

The model is trained on the [scikit-learn/churn-prediction](https://huggingface.co/datasets/scikit-learn/churn-prediction) dataset from HuggingFace. It contains telecom customer records with the following characteristics:

- **Size:** 7,043 customer records
- **Target:** `Churn` (Yes/No — whether the customer left)
- **Class distribution:** ~73% No Churn, ~27% Churn
- **Features:** 19 input features + 1 target + customerID (dropped during training)

## Model

**Algorithm:** Random Forest Classifier (scikit-learn)

**Hyperparameters** (tuned via GridSearchCV):

| Parameter           | Value |
|---------------------|-------|
| `n_estimators`      | 300   |
| `max_depth`         | 10    |
| `min_samples_split` | 5     |
| `min_samples_leaf`  | 2     |
| `random_state`      | 42    |

**Training approach:**

1. Data cleaning — convert `TotalCharges` to numeric, impute missing values with mean, drop `customerID`
2. Label encoding — binary columns (gender, Partner, Dependents, PhoneService, PaperlessBilling, Churn)
3. One-hot encoding — multi-class columns (MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaymentMethod)
4. 80/20 train/test split (`random_state=42`)
5. GridSearchCV hyperparameter tuning on Decision Tree, Random Forest, and Voting Classifier
6. Best model (Random Forest) saved as `final_model.joblib`

**Models evaluated:**

| Model               | Description                                    |
|---------------------|------------------------------------------------|
| Decision Tree       | Single tree baseline                           |
| Random Forest       | Ensemble of 300 trees (selected)               |
| Voting Classifier   | Soft-voting ensemble of DT + RF                |

## Features (Input Variables)

| Feature             | Type     | Values                                         |
|---------------------|----------|------------------------------------------------|
| gender              | Binary   | Female (0), Male (1)                           |
| SeniorCitizen       | Binary   | 0 (No), 1 (Yes)                               |
| Partner             | Binary   | No (0), Yes (1)                               |
| Dependents          | Binary   | No (0), Yes (1)                               |
| tenure              | Numeric  | 0–72 months                                   |
| PhoneService        | Binary   | No (0), Yes (1)                               |
| MultipleLines       | Categorical | No phone service, No, Yes                  |
| InternetService     | Categorical | DSL, Fiber optic, No                        |
| OnlineSecurity      | Categorical | No internet service, No, Yes               |
| OnlineBackup        | Categorical | No internet service, No, Yes               |
| DeviceProtection    | Categorical | No internet service, No, Yes               |
| TechSupport         | Categorical | No internet service, No, Yes               |
| StreamingTV         | Categorical | No internet service, No, Yes               |
| StreamingMovies     | Categorical | No internet service, No, Yes               |
| Contract            | Categorical | Month-to-month, One year, Two year         |
| PaperlessBilling    | Binary   | No (0), Yes (1)                               |
| PaymentMethod       | Categorical | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) |
| MonthlyCharges      | Numeric  | 0–200 ($)                                     |
| TotalCharges        | Numeric  | 0–10000 ($)                                   |

**Total model features after encoding:** 29 (9 label-encoded + 20 one-hot encoded)

## Tech Stack

| Layer        | Technology                           |
|-------------|--------------------------------------|
| ML Framework | scikit-learn 1.5.2                  |
| Backend      | FastAPI 0.115.5 + Uvicorn 0.32.1   |
| Templating   | Jinja2 3.1.4                        |
| Data         | pandas 2.2.3, numpy 2.1.3           |
| Serialization| joblib 1.4.2                        |
| Frontend     | Vanilla HTML/CSS (no JS framework)  |
| Fonts        | Google Fonts (Outfit, Space Mono)    |

## Project Structure

```
Customer_Churn_Prediction/
├── main.py                              # FastAPI application
├── Customer_Churn_Prediction.ipynb      # Jupyter notebook (training & EDA)
├── final_model.joblib                   # Trained Random Forest model (~14MB)
├── finalized_random_forest_model.joblib # Alternate model artifact (~14MB)
├── requirements.txt                     # Python dependencies
├── templates/
│   └── index.html                       # Jinja2 template (form + results)
└── static/
    └── style.css                        # Dark-themed CSS
```

## Setup & Installation

**Prerequisites:** Python 3.9+

```bash
# Clone the repository
git clone https://github.com/balu-16/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

## Running the Application

```bash
python main.py
```

The server starts on `http://localhost:8001`. Open this URL in your browser to access the prediction form.

To run on a different host/port:

```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

For development with auto-reload:

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

## API Endpoints

| Method | Path       | Description                        |
|--------|-----------|------------------------------------|
| GET    | `/`       | Renders the prediction form        |
| POST   | `/predict`| Accepts form data, returns prediction with probabilities |

FastAPI also provides automatic interactive API docs at:

- Swagger UI: `http://localhost:8001/docs`
- ReDoc: `http://localhost:8001/redoc`

### POST `/predict`

**Request:** Form-encoded fields (all 19 features listed above)

**Response:** Rendered HTML with:
- `prediction`: "Will Churn" or "Will Stay"
- `churn_probability`: float (0–100)
- `stay_probability`: float (0–100)
- `is_churn`: boolean

## How It Works

1. User fills out the web form with customer attributes
2. Form submits a POST request to `/predict`
3. Backend encodes inputs:
   - Binary fields: label-encoded (Yes→1, No→0)
   - Multi-class fields: one-hot encoded (matching training columns)
4. Encoded features are assembled into a pandas DataFrame (29 columns)
5. `model.predict()` returns the class (0=Stay, 1=Churn)
6. `model.predict_proba()` returns class probabilities
7. Result is rendered in the UI with probability bars and (if churn) retention recommendations

## Notebook Walkthrough

`Customer_Churn_Prediction.ipynb` contains the full ML pipeline:

| Cell(s) | Step                                    |
|---------|-----------------------------------------|
| 0–3     | Install deps & import libraries         |
| 4–7     | Load dataset from HuggingFace           |
| 8–19    | EDA: head, info, describe, unique values, missing values |
| 20–23   | Visualizations: tenure/charges histograms, gender/senior pie charts |
| 24–25   | Feature encoding (label + one-hot)      |
| 26–27   | Churn distribution & gender vs churn    |
| 28–29   | Correlation heatmap                     |
| 30–33   | Train/test split + data cleaning        |
| 34–35   | Train DT, RF, Voting Classifier         |
| 36–39   | GridSearchCV hyperparameter tuning      |
| 40–42   | Prediction on new data & model saving   |
