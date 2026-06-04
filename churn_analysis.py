import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
import pickle
import os

# ══════════════════════════════════════════════════════
# STEP 1 — LOAD & CLEAN
# ══════════════════════════════════════════════════════
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)
df.drop(columns=['customerID'], inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['Churn_Label'] = df['Churn'].map({0: 'No Churn', 1: 'Churned'})

print("✅ Data loaded and cleaned")
print(f"   Shape: {df.shape}")
print(f"   Churn rate: {df['Churn'].mean():.1%}")

# ══════════════════════════════════════════════════════
# STEP 2 — EDA CHARTS (Fixed title alignment)
# ══════════════════════════════════════════════════════
sns.set_theme(style="whitegrid", font_scale=1.0)
NAVY   = '#1a2f5a'
GOLD   = '#c8913a'
COLORS = [NAVY, '#2d5a8e', GOLD, '#e8b05a']

fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor('#f9f8f5')

# Title OUTSIDE gridspec — perfectly centered
fig.text(0.5, 0.96,
         'Customer Churn — Exploratory Data Analysis',
         ha='center', va='top',
         fontsize=17, fontweight='bold', color=NAVY)

gs = gridspec.GridSpec(2, 3, figure=fig,
                       hspace=0.45, wspace=0.35,
                       left=0.06, right=0.97,
                       top=0.89, bottom=0.08)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[1, 2])

# Chart 1 — Pie
churn_counts = df['Churn'].value_counts()
wedges, texts, autotexts = ax1.pie(
    churn_counts,
    labels=['No Churn', 'Churned'],
    autopct='%1.1f%%',
    colors=[NAVY, GOLD],
    startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2},
    textprops={'fontsize': 10})
for at in autotexts:
    at.set_color('white')
    at.set_fontweight('bold')
ax1.set_title('Churn Distribution', fontweight='bold', color=NAVY, pad=10)

# Chart 2 — Contract Type
contract_churn = df.groupby('Contract')['Churn'].mean().reset_index()
bars2 = ax2.bar(contract_churn['Contract'], contract_churn['Churn'],
                color=[NAVY, '#2d5a8e', GOLD], edgecolor='white', linewidth=1.5)
for bar, val in zip(bars2, contract_churn['Churn']):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01,
             f'{val:.0%}', ha='center', va='bottom',
             fontsize=9, fontweight='bold', color=NAVY)
ax2.set_title('Churn Rate by Contract Type', fontweight='bold', color=NAVY, pad=10)
ax2.set_ylabel('Churn Rate', fontsize=9)
ax2.set_xlabel('')
ax2.set_ylim(0, 0.55)
ax2.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax2.tick_params(axis='x', labelsize=9)

# Chart 3 — Tenure
sns.histplot(data=df, x='tenure', hue='Churn_Label', bins=30,
             palette={'No Churn': NAVY, 'Churned': GOLD},
             ax=ax3, alpha=0.85)
ax3.set_title('Tenure Distribution by Churn',
              fontweight='bold', color=NAVY, pad=10)
ax3.set_xlabel('Tenure (months)', fontsize=9)
ax3.set_ylabel('Count', fontsize=9)
legend = ax3.get_legend()
if legend:
    legend.set_title('')

# Chart 4 — Monthly Charges
sns.boxplot(data=df, x='Churn_Label', y='MonthlyCharges',
            hue='Churn_Label', legend=False,
            palette={'No Churn': NAVY, 'Churned': GOLD},
            ax=ax4, linewidth=1.5,
            flierprops={'markerfacecolor': GOLD, 'markersize': 3})
ax4.set_title('Monthly Charges vs Churn',
              fontweight='bold', color=NAVY, pad=10)
ax4.set_xlabel('')
ax4.set_ylabel('Monthly Charges (₹)', fontsize=9)

# Chart 5 — Internet Service
internet_churn = df.groupby(
    'InternetService')['Churn'].mean().reset_index()
bars5 = ax5.bar(internet_churn['InternetService'], internet_churn['Churn'],
                color=[NAVY, '#2d5a8e', GOLD],
                edgecolor='white', linewidth=1.5)
for bar, val in zip(bars5, internet_churn['Churn']):
    ax5.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01,
             f'{val:.0%}', ha='center', va='bottom',
             fontsize=9, fontweight='bold', color=NAVY)
ax5.set_title('Churn Rate by Internet Service',
              fontweight='bold', color=NAVY, pad=10)
ax5.set_ylabel('Churn Rate', fontsize=9)
ax5.set_xlabel('')
ax5.set_ylim(0, 0.55)
ax5.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

# Chart 6 — Payment Method
pay_churn = df.groupby('PaymentMethod')['Churn'].mean().reset_index()
short_labels = ['Bank Transfer', 'Credit Card', 'Elec. Check', 'Mailed Check']
bars6 = ax6.bar(short_labels, pay_churn['Churn'],
                color=COLORS, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars6, pay_churn['Churn']):
    ax6.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01,
             f'{val:.0%}', ha='center', va='bottom',
             fontsize=9, fontweight='bold', color=NAVY)
ax6.set_title('Churn Rate by Payment Method',
              fontweight='bold', color=NAVY, pad=10)
ax6.set_ylabel('Churn Rate', fontsize=9)
ax6.set_xlabel('')
ax6.set_ylim(0, 0.65)
ax6.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax6.tick_params(axis='x', labelsize=8.5)

plt.savefig('eda_charts.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("✅ EDA charts saved as eda_charts.png")

# ══════════════════════════════════════════════════════
# STEP 3 — MODEL BUILDING
# ══════════════════════════════════════════════════════
print("\n--- Building Random Forest Model ---")

df_model = df.drop(columns=['Churn_Label'])
cat_cols  = df_model.select_dtypes(include='object').columns.tolist()

le = LabelEncoder()
for col in cat_cols:
    df_model[col] = le.fit_transform(df_model[col])

X = df_model.drop(columns=['Churn'])
y = df_model['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    class_weight='balanced')
rf_model.fit(X_train, y_train)

y_pred  = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1]

acc     = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n════════════════════════════════")
print("        MODEL RESULTS")
print("════════════════════════════════")
print(f"  Accuracy  : {acc:.2%}")
print(f"  ROC-AUC   : {roc_auc:.2%}")
print("════════════════════════════════")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['No Churn', 'Churned']))

# ══════════════════════════════════════════════════════
# STEP 4 — FEATURE IMPORTANCE CHART
# ══════════════════════════════════════════════════════
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top10 = importances.sort_values(ascending=True).tail(10)

fig2, ax = plt.subplots(figsize=(10, 6))
fig2.patch.set_facecolor('#f9f8f5')
ax.set_facecolor('#f9f8f5')

bars = ax.barh(top10.index, top10.values,
               color=[NAVY if v < top10.values.max()*0.6
                      else GOLD for v in top10.values],
               edgecolor='white', linewidth=1.2)
for bar, val in zip(bars, top10.values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9,
            fontweight='bold', color=NAVY)

ax.set_title('Top 10 Features Driving Customer Churn',
             fontsize=14, fontweight='bold', color=NAVY, pad=15)
ax.set_xlabel('Feature Importance Score', fontsize=10)
ax.set_xlim(0, top10.values.max() * 1.18)
ax.tick_params(axis='y', labelsize=10)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight',
            facecolor=fig2.get_facecolor())
plt.show()
print("✅ Feature importance chart saved")

# ══════════════════════════════════════════════════════
# STEP 5 — SAVE MODEL + CLEANED DATA FOR POWER BI
# ══════════════════════════════════════════════════════

# Save model as pickle file
with open('churn_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("✅ Model saved as churn_model.pkl")

# Save cleaned data for Power BI import
df_powerbi = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df_powerbi['TotalCharges'] = pd.to_numeric(
    df_powerbi['TotalCharges'], errors='coerce')
df_powerbi.dropna(subset=['TotalCharges'], inplace=True)
df_powerbi['Churn_Binary'] = df_powerbi['Churn'].map({'Yes': 1, 'No': 0})
df_powerbi['Churn_Risk'] = df_powerbi['Churn'].map(
    {'Yes': 'High Risk', 'No': 'Low Risk'})
df_powerbi.to_csv('churn_clean_for_powerbi.csv', index=False)
print("✅ Clean data saved as churn_clean_for_powerbi.csv")

# ══════════════════════════════════════════════════════
# STEP 6 — BUSINESS INSIGHTS SUMMARY
# ══════════════════════════════════════════════════════
print("\n════════════════════════════════════════════════")
print("       KEY BUSINESS INSIGHTS")
print("════════════════════════════════════════════════")
df_orig = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df_orig['TotalCharges'] = pd.to_numeric(
    df_orig['TotalCharges'], errors='coerce')
df_orig.dropna(inplace=True)

mtm = df_orig[df_orig['Contract']=='Month-to-month']['Churn'].value_counts(normalize=True)['Yes']
fiber = df_orig[df_orig['InternetService']=='Fiber optic']['Churn'].value_counts(normalize=True)['Yes']
new_cust = df_orig[df_orig['tenure'] <= 6]['Churn'].value_counts(normalize=True)['Yes']
elec = df_orig[df_orig['PaymentMethod']=='Electronic check']['Churn'].value_counts(normalize=True)['Yes']

print(f"  Month-to-month contract churn rate : {mtm:.1%}")
print(f"  Fiber optic internet churn rate    : {fiber:.1%}")
print(f"  New customers (≤6 months) churn    : {new_cust:.1%}")
print(f"  Electronic check payment churn     : {elec:.1%}")
print("════════════════════════════════════════════════")
print("\n🎉 PROJECT 1 COMPLETE!")
print("   Files saved:")
print("   → eda_charts.png")
print("   → feature_importance.png")
print("   → churn_model.pkl")
print("   → churn_clean_for_powerbi.csv")