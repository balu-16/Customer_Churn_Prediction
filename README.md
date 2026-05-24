# 🛡️ ChurnGuard — Customer Churn Prediction

<p align="center">
  <strong>AI-Powered Customer Retention Intelligence</strong><br>
  Predict telecom customer churn using a trained Random Forest model served via a FastAPI web application.
</p>

---

## 📖 About

ChurnGuard is a machine learning web application that predicts whether a telecom customer will churn based on their demographics, account information, service subscriptions, and billing details.

The model is trained on the [scikit-learn churn prediction dataset](https://huggingface.co/datasets/scikit-learn/churn-prediction) using a **Random Forest Classifier** optimized via `GridSearchCV`. The web interface is a clean, dark-themed HTML form powered by **FastAPI** and **Jinja2**.

---

## ✨ Features

- 🔮 **Real-time predictions** — Submit customer data and get instant churn risk assessment
- 📊 **Probability bars** — Visual churn vs. retention probability breakdown
- 🎨 **Modern dark UI** — Responsive design with smooth animations
- ⚡ **Recommended actions** — When churn is predicted, actionable retention steps are displayed
- 📱 **Mobile-friendly** — Works across all screen sizes

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) |
| **Frontend** | Jinja2 HTML templates + custom CSS |
| **ML Model** | scikit-learn `RandomForestClassifier` (optimized with GridSearchCV) |
| **Serialization** | [joblib](https://joblib.readthedocs.io/) |
| **Data Processing** | pandas, numpy |
| **Notebook** | Jupyter Notebook (exploratory analysis + model training) |

---

## 📦 Installation

### Prerequisites

- Python 3.9+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/balu-16/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

The server starts at **http://localhost:8001**. Open it in your browser.

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| **Algorithm** | Random Forest Classifier |
| **Hyperparameter Tuning** | GridSearchCV (`n_estimators`, `max_depth`, `min_samples_split`) |
| **Train/Test Split** | 80/20 (`random_state=42`) |
| **Models Compared** | Decision Tree, Random Forest, Voting Classifier (DT + RF) |
| **Encoding** | Label encoding (binary cols) + One-hot encoding (multi-class cols) |

### Input Features (19)

- **Demographics**: gender, senior citizen, partner, dependents
- **Account**: tenure, contract type, paperless billing, payment method
- **Phone**: phone service, multiple lines
- **Internet**: internet service, online security, online backup, device protection, tech support, streaming TV, streaming movies
- **Billing**: monthly charges, total charges

### Output

- **Prediction**: `Will Churn` or `Will Stay`
- **Churn probability**: percentage
- **Retention likelihood**: percentage

---

## 📁 Project Structure

```
Customer_Churn_Prediction/
├── main.py                                  # FastAPI application (routes, model loading, prediction logic)
├── requirements.txt                         # Python dependencies
├── final_model.joblib                       # Serialized trained Random Forest model
├── finalized_random_forest_model.joblib     # Alternate model artifact
├── Customer_Churn_Prediction.ipynb          # Jupyter notebook (EDA, training, evaluation)
├── static/
│   └── style.css                            # Dark-themed responsive CSS
└── templates/
    └── index.html                           # Jinja2 HTML form + result display
```

---

## 🚀 Usage

1. Fill in the customer form fields (demographics, services, billing)
2. Click **Analyze Churn Risk**
3. View the prediction result with churn/retention probability bars
4. If churn is predicted, review the recommended retention actions

---

## 📄 License

This project is open source. See the repository for license details.
