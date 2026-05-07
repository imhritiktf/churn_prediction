# 📉 Customer Churn Prediction — End-to-End ML Project

A production-ready machine learning pipeline that predicts whether a telecom customer will churn, built with a modular project structure, Flask web app, and full ML lifecycle from data ingestion to deployment.

---

## 🔍 Problem Statement

Customer churn is one of the biggest challenges in the telecom industry. Retaining an existing customer is significantly cheaper than acquiring a new one. This project builds a classification model to identify customers likely to churn, enabling proactive retention strategies.

**Dataset:** IBM Telco Customer Churn (7,043 customers, 20 features)

---

## ⚙️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| ML Libraries | Scikit-learn, XGBoost, LightGBM, Imbalanced-learn |
| Web Framework | Flask |
| Data | Pandas, NumPy |
| Serialization | Joblib |
| Environment | Virtualenv |
| Version Control | Git & GitHub |

---

## 📁 Project Structure

```
churn_prediction/
├── artifacts/                  # Saved model & preprocessor (.pkl files)
├── notebooks/
│   └── data/                   # Raw dataset
├── src/
│   ├── components/
│   │   ├── data_ingestion.py       # Load, clean, train-test split
│   │   ├── data_transformation.py  # Encoding, scaling, SMOTE
│   │   └── model_trainer.py        # Train, tune, evaluate, save best model
│   ├── pipeline/
│   │   ├── train_pipeline.py       # Orchestrates full training flow
│   │   └── predict_pipeline.py     # Loads artifacts, serves predictions
│   ├── exception.py            # Custom exception handler
│   ├── logger.py               # Logging setup
│   └── utils.py                # save_object, load_object, evaluate_models
├── templates/
│   ├── index.html              # Landing page
│   └── home.html               # Prediction form + result
├── app.py                      # Flask application
├── setup.py                    # Package setup
└── requirements.txt
```

---

## 🧠 ML Pipeline

### 1. Data Ingestion
- Loads raw CSV, fixes `TotalCharges` (spaces → NaN via `pd.to_numeric`)
- Drops `customerID` (non-feature)
- Stratified train-test split (80/20) to maintain class ratio

### 2. Data Transformation
Four column types handled separately via `ColumnTransformer`:

| Column Type | Columns | Treatment |
|---|---|---|
| Numerical | tenure, MonthlyCharges, TotalCharges | SimpleImputer(median) + StandardScaler |
| Binary Yes/No | Partner, Dependents, PhoneService, PaperlessBilling | OrdinalEncoder + StandardScaler |
| Gender | gender | OrdinalEncoder + StandardScaler |
| Multi-category | Contract, PaymentMethod, InternetService, etc. | OneHotEncoder(drop='first') |
| Passthrough | SeniorCitizen | As-is (already 0/1) |

**Imbalance Handling:**
- Dataset: No Churn = 5174 (73%) vs Churn = 1869 (27%)
- SMOTE applied on training set only (never on test — prevents data leakage)
- `class_weight='balanced'` on applicable models
- `scale_pos_weight = neg/pos` for XGBoost

### 3. Model Training & Hyperparameter Tuning
- 8 models evaluated: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNN, XGBoost, LightGBM, AdaBoost
- `RandomizedSearchCV` with `StratifiedKFold(n_splits=5)` for tuning
- Scoring metric: **ROC-AUC** (not accuracy — data is imbalanced)
- Best model auto-saved to `artifacts/model.pkl`

### 4. Evaluation Metrics
| Metric | Why Used |
|---|---|
| ROC-AUC | Threshold-independent, best for model comparison |
| Recall (Churn=Yes) | Missing a churner is costly for business |
| F1-Score | Balance of precision & recall |
| Confusion Matrix | Visualize False Negatives vs False Positives |

---

## 📊 Results

| Model | ROC-AUC |
|---|---|
| XGBoost (tuned) | **0.8492** ✅ |
| Logistic Regression (tuned) | 0.8454 |
| Random Forest (tuned) | 0.8446 |
| LightGBM (tuned) | 0.8445 |

**Best Model: XGBoost**
- Recall (Churn): 80% — 300 out of 374 actual churners caught
- Only 74 churners missed on test set

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/churn_prediction.git
cd churn_prediction
```

### 2. Create & activate virtual environment
```bash
python -m venv venv
source venv/Scripts/activate    # Windows
source venv/bin/activate        # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Add dataset
Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` inside `notebooks/data/`

### 5. Train the model
```bash
python src/pipeline/train_pipeline.py
```

### 6. Run the Flask app
```bash
python app.py
```

Open browser: `http://127.0.0.1:5000`

---

## 🌐 Web App

The Flask app accepts customer details via a form and returns:
- **Prediction:** Will Churn ⚠️ or Will Not Churn ✅
- **Churn Probability:** e.g., 82.4% chance of churning

---

## 📚 Key Learnings

- Modular ML project structure (ingestion → transformation → training → prediction)
- Handling imbalanced classification with SMOTE + class weights
- Data leakage prevention — SMOTE only on train, preprocessor fit only on train
- Why ROC-AUC > Accuracy for imbalanced datasets
- StratifiedKFold for reliable cross-validation on imbalanced data
- End-to-end deployment with Flask

---

## 🔗 Dataset Source

[IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 👤 Author

**Hritik** — BCA Student | Aspiring Data Analyst & ML Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/hritik-modanwal-86b89a25a)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/imhritiktf)