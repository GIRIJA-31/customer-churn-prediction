# 🔄 Customer Churn Prediction

## 📌 Project Overview
Predicted customer churn for a telecom company using machine learning.
Identified key churn drivers to help the business retain at-risk customers
and reduce revenue loss.

## 📊 Dataset
- **Source:** IBM Telco Customer Churn Dataset (Kaggle)
- **Size:** 7,032 customers | 20 features
- **Target:** Churn (Yes/No)

## 🛠️ Tech Stack
- **Language:** Python 3.10
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Model:** Random Forest Classifier

## 🔍 Key Steps
1. **Data Cleaning** — Fixed TotalCharges dtype, removed 11 null rows
2. **EDA** — Analyzed churn patterns across contract type, tenure,
   internet service, and payment method
3. **Feature Engineering** — Label encoded 15 categorical columns
4. **Modelling** — Random Forest with class_weight='balanced'
   to handle 73/27 class imbalance
5. **Evaluation** — Accuracy, ROC-AUC, Classification Report

## 📈 Results
| Metric | Score |
|---|---|
| Accuracy | 76.97% |
| ROC-AUC | 83.04% |
| Churned Recall | 72% |

## 💡 Key Business Insights
| Finding | Insight |
|---|---|
| Month-to-month contracts | 43% churn — highest risk segment |
| Fiber optic internet | 42% churn — needs service quality review |
| New customers (≤6 months) | Highest churn — need onboarding program |
| Electronic check payment | 45% churn — push auto-pay adoption |
| Contract type (0.188) | #1 most important feature driving churn |

## 📁 Project Files
churn_project/
│
├── churn_analysis.py           # Main analysis + ML code
├── eda_charts.png              # 6-panel EDA visualization
├── feature_importance.png      # Top 10 churn drivers chart
├── churn_model.pkl             # Saved Random Forest model
└── churn_clean_for_powerbi.csv # Cleaned data for dashboard
## 🚀 How to Run
```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Run analysis
python churn_analysis.py
```

## 🎯 Business Impact
The model identifies the **top 20% highest-risk customers** before they
churn. In a real telecom company with 100,000 customers, this could
enable targeted retention campaigns saving an estimated **₹2–4 Cr/year**
in lost revenue.