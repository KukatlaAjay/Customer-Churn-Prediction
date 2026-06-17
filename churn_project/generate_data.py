import pandas as pd
import numpy as np

np.random.seed(42)
n = 7043

contract_type = np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.25, 0.20])
tenure = np.where(contract_type == 'Month-to-month',
                  np.random.randint(1, 36, n),
                  np.where(contract_type == 'One year',
                           np.random.randint(12, 60, n),
                           np.random.randint(24, 72, n)))

monthly_charges = np.where(contract_type == 'Month-to-month',
                           np.random.uniform(45, 110, n),
                           np.where(contract_type == 'One year',
                                    np.random.uniform(30, 85, n),
                                    np.random.uniform(20, 70, n)))

total_charges = (tenure * monthly_charges + np.random.normal(0, 50, n)).clip(0)

# Churn probability based on key drivers
churn_prob = (
    0.40 * (contract_type == 'Month-to-month').astype(float) +
    0.15 * (monthly_charges > 75).astype(float) +
    0.20 * (tenure < 12).astype(float) +
    np.random.uniform(0, 0.25, n)
)
churn = (churn_prob > 0.45).astype(int)

df = pd.DataFrame({
    'customerID': [f'CUST-{i:05d}' for i in range(1, n+1)],
    'gender': np.random.choice(['Male', 'Female'], n),
    'SeniorCitizen': np.random.choice([0, 1], n, p=[0.84, 0.16]),
    'Partner': np.random.choice(['Yes', 'No'], n, p=[0.48, 0.52]),
    'Dependents': np.random.choice(['Yes', 'No'], n, p=[0.30, 0.70]),
    'tenure': tenure,
    'PhoneService': np.random.choice(['Yes', 'No'], n, p=[0.90, 0.10]),
    'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n, p=[0.42, 0.48, 0.10]),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.34, 0.44, 0.22]),
    'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.29, 0.50, 0.21]),
    'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.34, 0.44, 0.22]),
    'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.34, 0.44, 0.22]),
    'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.29, 0.49, 0.22]),
    'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.38, 0.40, 0.22]),
    'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.39, 0.39, 0.22]),
    'Contract': contract_type,
    'PaperlessBilling': np.random.choice(['Yes', 'No'], n, p=[0.59, 0.41]),
    'PaymentMethod': np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        n, p=[0.34, 0.23, 0.22, 0.21]),
    'MonthlyCharges': monthly_charges.round(2),
    'TotalCharges': total_charges.round(2),
    'Churn': np.where(churn == 1, 'Yes', 'No')
})

df.to_csv('/home/claude/churn_project/data/telco_churn.csv', index=False)
print(f"Dataset saved: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Churn rate: {df['Churn'].value_counts(normalize=True)['Yes']:.1%}")
