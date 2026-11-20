# Loan Approval Prediction (CreditWise)

A machine learning project that predicts whether a loan application will be **approved or rejected** based on the applicant's financial and personal details.

## Problem Statement
Banks receive thousands of loan applications. This project builds a classification model that helps screen applicants quickly and consistently, using features such as credit score, income, DTI ratio, savings, employment status and loan purpose.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Jupyter Notebook

## Workflow
1. **Data cleaning:** handled missing values with `SimpleImputer` (mean for numerical, most frequent for categorical columns)
2. **EDA:** class balance (pie chart), income and credit score distributions by approval status, outlier detection with box plots, correlation heatmap
3. **Encoding:** `LabelEncoder` for Education Level and target, `OneHotEncoder` for Employment Status, Marital Status, Loan Purpose, Property Area, Gender, Employer Category (42 features in total)
4. **Train-test split:** 80/20 with `random_state=42`
5. **Scaling:** `StandardScaler` (fit on train only, to avoid data leakage)
6. **Modeling:** Logistic Regression, K-Nearest Neighbors (k=13), Gaussian Naive Bayes
7. **Evaluation:** Precision and Recall, plus 5-fold cross-validation
8. **Feature engineering:** added squared `Credit_Score` and `DTI_Ratio` features and compared the results against the baseline

## Results

| Model | Precision | Recall |
|---|---|---|
| Logistic Regression | 79% | 80% |
| KNN (k=13) | 73% | 49% |
| Gaussian Naive Bayes | 78 %| 77% |


## Key Learnings
- A single train-test split can be misleading, so I used cross-validation to compare models and feature sets reliably.
- Feature engineering does not always help. Squaring features slightly lowered Naive Bayes precision, because it breaks the Gaussian and independence assumptions.
- Precision matters in lending: a false approval (bad loan) is costly, so precision was my main metric.

## Project Structure
```
loan-approval-prediction/
├── credit_wise.ipynb
├── data/
│   └── loan_approval_data.csv
├── requirements.txt
└── README.md
```

## How to Run
```bash
git clone https://github.com/<your-username>/loan-approval-prediction.git
cd loan-approval-prediction
pip install -r requirements.txt
jupyter notebook credit_wise.ipynb
```

## Future Improvements
- Try Random Forest and XGBoost
- Handle class imbalance (`stratify`, class weights, SMOTE)
- Wrap preprocessing and model in a scikit-learn `Pipeline`
- Deploy with FastAPI + Docker and a Streamlit front end

## Author
**Shreya Gupta** | [LinkedIn](|linkedin.com/in/shreyag15 ) | shreyaguptas0315@gmail.com
