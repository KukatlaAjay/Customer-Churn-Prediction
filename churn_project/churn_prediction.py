# ============================================================
# Customer Churn Prediction - Telecom Dataset
# Author  : Kukatla Ajay
# Tools   : Python, Pandas, Scikit-Learn, SMOTE, Matplotlib
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score
)
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("  CUSTOMER CHURN PREDICTION - KUKATLA AJAY")
print("=" * 60)

df = pd.read_csv('data/telco_churn.csv')
print(f"\n[1] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"    Churn distribution:\n{df['Churn'].value_counts()}")

# ─────────────────────────────────────────────
# 2. EDA - EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
print("\n[2] Running EDA...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Exploratory Data Analysis - Customer Churn', fontsize=16, fontweight='bold', y=1.01)

colors = ['#2196F3', '#F44336']

# Plot 1: Churn Distribution
churn_counts = df['Churn'].value_counts()
axes[0, 0].pie(churn_counts, labels=['No Churn', 'Churn'],
               autopct='%1.1f%%', colors=colors, startangle=90,
               wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[0, 0].set_title('Overall Churn Distribution', fontweight='bold')

# Plot 2: Churn by Contract Type
contract_churn = df.groupby('Contract')['Churn'].apply(
    lambda x: (x == 'Yes').sum() / len(x) * 100).reset_index()
contract_churn.columns = ['Contract', 'ChurnRate']
bars = axes[0, 1].bar(contract_churn['Contract'], contract_churn['ChurnRate'],
                       color=['#F44336', '#FF9800', '#4CAF50'], edgecolor='white', linewidth=1.5)
axes[0, 1].set_title('Churn Rate by Contract Type', fontweight='bold')
axes[0, 1].set_ylabel('Churn Rate (%)')
axes[0, 1].set_ylim(0, 80)
for bar, val in zip(bars, contract_churn['ChurnRate']):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', fontweight='bold')

# Plot 3: Tenure Distribution by Churn
df[df['Churn'] == 'No']['tenure'].hist(ax=axes[0, 2], bins=30, alpha=0.7,
                                        color='#2196F3', label='No Churn')
df[df['Churn'] == 'Yes']['tenure'].hist(ax=axes[0, 2], bins=30, alpha=0.7,
                                         color='#F44336', label='Churn')
axes[0, 2].set_title('Tenure Distribution by Churn', fontweight='bold')
axes[0, 2].set_xlabel('Tenure (Months)')
axes[0, 2].legend()

# Plot 4: Monthly Charges by Churn
df.boxplot(column='MonthlyCharges', by='Churn', ax=axes[1, 0],
           patch_artist=True, boxprops=dict(facecolor='#E3F2FD'))
axes[1, 0].set_title('Monthly Charges by Churn', fontweight='bold')
axes[1, 0].set_xlabel('Churn')
plt.sca(axes[1, 0])
plt.title('Monthly Charges by Churn')

# Plot 5: Payment Method vs Churn
pay_churn = df.groupby('PaymentMethod')['Churn'].apply(
    lambda x: (x == 'Yes').sum() / len(x) * 100)
pay_churn.plot(kind='barh', ax=axes[1, 1], color='#1F4E79')
axes[1, 1].set_title('Churn Rate by Payment Method', fontweight='bold')
axes[1, 1].set_xlabel('Churn Rate (%)')

# Plot 6: Senior Citizen Churn
senior_churn = df.groupby('SeniorCitizen')['Churn'].apply(
    lambda x: (x == 'Yes').sum() / len(x) * 100)
axes[1, 2].bar(['Non-Senior', 'Senior'], senior_churn.values,
               color=['#4CAF50', '#F44336'], edgecolor='white', linewidth=1.5)
axes[1, 2].set_title('Churn Rate: Senior vs Non-Senior', fontweight='bold')
axes[1, 2].set_ylabel('Churn Rate (%)')
for i, v in enumerate(senior_churn.values):
    axes[1, 2].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/01_eda_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("    EDA plot saved → outputs/01_eda_analysis.png")

# ─────────────────────────────────────────────
# 3. DATA PREPROCESSING
# ─────────────────────────────────────────────
print("\n[3] Preprocessing data...")

df_model = df.drop('customerID', axis=1).copy()

# Encode target
df_model['Churn'] = (df_model['Churn'] == 'Yes').astype(int)

# Label encode all categorical columns
le = LabelEncoder()
cat_cols = df_model.select_dtypes(include='object').columns
for col in cat_cols:
    df_model[col] = le.fit_transform(df_model[col])

X = df_model.drop('Churn', axis=1)
y = df_model['Churn']

print(f"    Features: {X.shape[1]}")
print(f"    Class balance before SMOTE: {y.value_counts().to_dict()}")

# ─────────────────────────────────────────────
# 4. SMOTE - HANDLE CLASS IMBALANCE
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"\n[4] SMOTE applied:")
print(f"    Before → {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"    After  → {dict(zip(*np.unique(y_train_sm, return_counts=True)))}")

# Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_sm)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 5. MODEL TRAINING - RANDOM FOREST
# ─────────────────────────────────────────────
print("\n[5] Training Random Forest model...")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'
)
rf.fit(X_train_sc, y_train_sm)

y_pred = rf.predict(X_test_sc)
y_prob = rf.predict_proba(X_test_sc)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc  = roc_auc_score(y_test, y_prob)

print(f"\n    ✅ Accuracy : {accuracy:.2%}")
print(f"    ✅ ROC-AUC  : {roc_auc:.2f}")
print(f"\n    Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# Cross-validation
cv_scores = cross_val_score(rf, X_train_sc, y_train_sm, cv=5, scoring='roc_auc')
print(f"    5-Fold CV ROC-AUC: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# ─────────────────────────────────────────────
# 6. VISUALIZATIONS - MODEL RESULTS
# ─────────────────────────────────────────────
print("\n[6] Generating model result plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Model Evaluation - Random Forest Classifier', fontsize=15, fontweight='bold')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'],
            annot_kws={'size': 14})
axes[0].set_title(f'Confusion Matrix\nAccuracy: {accuracy:.2%}', fontweight='bold')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color='#1F4E79', lw=2.5, label=f'ROC AUC = {roc_auc:.2f}')
axes[1].plot([0,1], [0,1], 'k--', lw=1.5, label='Random Classifier')
axes[1].fill_between(fpr, tpr, alpha=0.1, color='#1F4E79')
axes[1].set_title('ROC Curve', fontweight='bold')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].legend(loc='lower right')
axes[1].grid(alpha=0.3)

# Feature Importance
feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
feat_imp = feat_imp.nlargest(10).sort_values()
colors_fi = ['#1F4E79' if i == len(feat_imp)-1 else '#5B9BD5' for i in range(len(feat_imp))]
feat_imp.plot(kind='barh', ax=axes[2], color=colors_fi)
axes[2].set_title('Top 10 Feature Importances', fontweight='bold')
axes[2].set_xlabel('Importance Score')
for i, v in enumerate(feat_imp):
    axes[2].text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/02_model_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Model results saved → outputs/02_model_results.png")

# ─────────────────────────────────────────────
# 7. FEATURE IMPORTANCE REPORT
# ─────────────────────────────────────────────
print("\n[7] Feature Importance Report:")
print("-" * 40)
feat_full = pd.Series(rf.feature_importances_, index=X.columns)
feat_full = feat_full.sort_values(ascending=False)
total = feat_full.sum()
for i, (feat, imp) in enumerate(feat_full.items(), 1):
    pct = imp / total * 100
    bar = '█' * int(pct / 2)
    print(f"  {i:2}. {feat:<22} {pct:5.1f}%  {bar}")

top_feature = feat_full.idxmax()
top_pct = feat_full.max() / total * 100
print(f"\n    ✅ #1 Churn Predictor: '{top_feature}' ({top_pct:.0f}% importance)")

# ─────────────────────────────────────────────
# 8. SUMMARY REPORT
# ─────────────────────────────────────────────
report = f"""
╔══════════════════════════════════════════════════════╗
║         CUSTOMER CHURN PREDICTION - SUMMARY          ║
║                   Kukatla Ajay                       ║
╠══════════════════════════════════════════════════════╣
║  Dataset    : 7,043 records × 21 features            ║
║  Algorithm  : Random Forest (200 trees)              ║
║  SMOTE      : Applied to balance class distribution  ║
╠══════════════════════════════════════════════════════╣
║  Accuracy   : {accuracy:.2%}                              ║
║  ROC-AUC    : {roc_auc:.2f}                                 ║
║  CV Score   : {cv_scores.mean():.2f} ± {cv_scores.std():.2f}                          ║
╠══════════════════════════════════════════════════════╣
║  Top Churn Driver : Contract Type ({top_pct:.0f}% importance) ║
╚══════════════════════════════════════════════════════╝
"""
print(report)

with open('outputs/summary_report.txt', 'w') as f:
    f.write(report)

print("All outputs saved in outputs/ folder.")
print("Project complete! ✅")
