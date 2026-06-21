# 📊 Task 2 — End-to-End ML Pipeline with Scikit-learn (Customer Churn)

A reusable, production-ready machine learning pipeline that predicts whether a
telecom customer will churn, built with scikit-learn's `Pipeline`, `ColumnTransformer`,
and `GridSearchCV` APIs.

---

## ✅ Task Requirements Coverage

| Requirement | Status |
|---|---|
| Data preprocessing (scaling, encoding) using `Pipeline` | ✅ `ColumnTransformer` with `StandardScaler` + `OneHotEncoder` |
| Train Logistic Regression and Random Forest | ✅ Both in a single `GridSearchCV` |
| Hyperparameter tuning with `GridSearchCV` | ✅ 5-fold CV, F1-score optimized, 24 total candidates |
| Export complete pipeline using `joblib` | ✅ Saved as `churn_pipeline.joblib`, reloaded and tested |

---

## 📁 Project Structure

```
task2_churn_pipeline/
├── churn_pipeline.py        # Main script — run this in VS Code
├── churn_pipeline.ipynb     # Jupyter Notebook version (step-by-step)
├── churn_pipeline.joblib    # Exported trained pipeline (created after running)
└── README.md                # This file
```

---

## 📦 Dataset

**Telco Customer Churn** — 7,043 customers, 21 columns.

The script auto-downloads the dataset directly from IBM's public GitHub repository.
No manual download or Kaggle login required.

---

## ⚙️ Setup

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# Install dependencies
pip install pandas numpy scikit-learn joblib
```

---

## ▶️ How to Run

### Option A — VS Code (recommended)
```bash
python churn_pipeline.py
```

### Option B — Jupyter Notebook
```bash
pip install jupyter
jupyter notebook churn_pipeline.ipynb
```

Both options produce identical results. Runtime: under 1 minute on any laptop (no GPU needed).

---

## 🔑 Key Design Decisions

**Single GridSearchCV across both models:**
Both Logistic Regression and Random Forest are tuned in one unified `GridSearchCV`
call using a list-style `param_grid`. This is the correct production approach —
one search, one winner, no manual comparison code.

**`drop='first'` in OneHotEncoder:**
Drops the first category from each encoded column to avoid the dummy variable trap
(multicollinearity), which matters especially for Logistic Regression.

**Complete pipeline in one `.joblib` file:**
The exported file contains the preprocessing steps (scaling + encoding) AND the
trained classifier. You feed it raw, unprocessed data — it handles everything internally.

---

## 📊 Results

| Metric | Score |
|---|---|
| Accuracy | ~80% |
| F1-Score | ~0.60 |
| Precision | ~0.65 |
| Recall | ~0.57 |
| ROC-AUC | ~0.84 |

Note: This dataset is imbalanced (~27% churn), so F1-score and ROC-AUC are more
meaningful than raw accuracy.

---

## 🔁 Using the Exported Pipeline

```python
import joblib
import pandas as pd

pipeline = joblib.load('churn_pipeline.joblib')

# Raw unprocessed customer data (same columns as training, minus Churn)
new_customer = pd.DataFrame([{
    'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes',
    'Dependents': 'No', 'tenure': 5, 'PhoneService': 'Yes',
    'MultipleLines': 'No', 'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No', 'OnlineBackup': 'No',
    'DeviceProtection': 'No', 'TechSupport': 'No',
    'StreamingTV': 'No', 'StreamingMovies': 'No',
    'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 80.0, 'TotalCharges': 400.0
}])

pred  = pipeline.predict(new_customer)[0]
proba = pipeline.predict_proba(new_customer)[0][1]
print('Churn' if pred == 1 else 'No Churn', f'({proba:.1%} probability)')
```

---

## 🧠 Skills Gained

- ML pipeline construction with `Pipeline` and `ColumnTransformer`
- Hyperparameter tuning across multiple model types with `GridSearchCV`
- Model export and reusability via `joblib`
- Production-readiness: no data leakage, single reusable artifact

---

## 📌 References

- [Telco Customer Churn Dataset — IBM](https://github.com/IBM/telco-customer-churn-on-icp4d)
- [scikit-learn Pipeline docs](https://scikit-learn.org/stable/modules/compose.html)
- [GridSearchCV docs](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [joblib docs](https://joblib.readthedocs.io/)
