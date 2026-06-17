**Author:** Kukatla Ajay  
**Tools:** Python, Pandas, Scikit-Learn, SMOTE, Matplotlib, Seaborn  
**Model:** Random Forest Classifier  

---

## 🎯 Project Overview

Built an end-to-end machine learning pipeline to predict customer churn for a telecom company using a dataset of **7,043 customer records across 21 features**. The goal is to identify customers likely to churn so the business can take proactive retention actions.

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | **96.38%** |
| ROC-AUC | **0.99** |
| 5-Fold CV ROC-AUC | **0.99 ± 0.00** |
| Top Churn Driver | **Contract Type (57% importance)** |

---

## 🗂️ Project Structure

```
customer-churn-prediction/
│
├── data/
│   └── telco_churn.csv          # Dataset (7,043 records × 21 features)
│
├── outputs/
│   ├── 01_eda_analysis.png      # EDA visualizations
│   ├── 02_model_results.png     # Confusion matrix, ROC curve, feature importance
│   └── summary_report.txt       # Final model summary
│
├── generate_data.py             # Script to generate/load dataset
├── churn_prediction.py          # Main ML pipeline
├── requirements.txt             # Dependencies
└── README.md
```

---

## 🔧 Steps Performed

### 1. Exploratory Data Analysis (EDA)
- Churn rate distribution
- Churn by Contract Type, Tenure, Monthly Charges, Payment Method, Senior Citizen

### 2. Data Preprocessing
- Label encoding of 14 categorical features
- Feature-target separation (19 input features)

### 3. SMOTE — Handling Class Imbalance
- Applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance training data
- Improved minority class recall by **18%**

### 4. Model Training — Random Forest
- 200 decision trees, max depth 12
- `class_weight='balanced'` for robust minority handling

### 5. Evaluation
- Confusion Matrix
- ROC Curve & AUC Score
- Classification Report (Precision, Recall, F1)
- 5-Fold Cross Validation

### 6. Feature Importance
- Contract Type → **57% importance** (top predictor)
- Tenure → 18.3%
- Monthly Charges → 15.0%

---

## 📈 Key Insights

- **Month-to-month contract** customers churn the most
- Customers with **tenure < 12 months** are at highest risk
- **High monthly charges (> ₹75)** correlate strongly with churn
- Electronic check users have the highest churn rate among payment methods

---

## ▶️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/KukatlaAjay/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate dataset
python generate_data.py

# 4. Run the ML pipeline
python churn_prediction.py
```

Output plots and summary will be saved in the `outputs/` folder.

---

## 📦 Dependencies

```
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
```

---

## 📬 Contact

**Kukatla Ajay**  
📧 ajaykukatla07@mail.com  
📍 Hyderabad, India  
🔗 [GitHub](https://github.com/KukatlaAjay)
