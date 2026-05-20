# Credit Risk Prediction Project

## Project Overview
This project builds a machine learning model to predict serious financial delinquency within two years using customer credit information.
The objective is to identify high-risk borrowers by analyzing financial behavior, debt patterns, income, and credit history.
The dataset contains 150,000 customer records and multiple financial indicators.

## Business Problem
Financial institutions need reliable methods to detect customers who may experience financial distress.

Accurate prediction helps:
- reduce default risk
- improve lending decisions
- strengthen risk management strategies
- support early intervention programs

This project applies supervised machine learning techniques to solve this binary classification problem.

## Dataset
**Dataset:** Give Me Some Credit from Kaggle.

### Target Variable:
SeriousDlqin2yrs
- 0 → No financial distress
- 1 → Serious financial distress

### Dataset size:
- 150,000 observations
- 11 predictor variables

### Features include:
- Revolving credit utilization
- Age
- Debt ratio
- Monthly income
- Credit lines and loans
- Delinquency history
- Number of dependents

## Project Workflow
### 1. Exploratory Data Analysis
- Distribution analysis of financial, personal, and credit history variables
- Log transformation of MonthlyIncome to handle right skew
- Correlation heatmap to identify multicollinearity

### 2. Data Cleaning
- Missing values: MonthlyIncome (20k missing) and NumberOfDependents imputed with median
- Outliers: RevolvingUtilizationOfUnsecuredLines clipped at 1.0; DebtRatio capped at the 99th percentile
- Impossible values: Age of 0 replaced with median; sentinel values of 96/98 in late payment columns replaced with median

### 3. Preprocessing
- Dropped raw MonthlyIncome in favour of log_income
- 80/20 train-test split (stratified by default)
- Standard scaling applied to all features

### 4. Class Imbalance Handling
Applied SMOTE (Synthetic Minority Oversampling Technique) to the training set only, preventing data leakage

### 5. Modeling
Three classifiers trained and evaluated:

- Logistic Regression (baseline)
- Random Forest
- XGBoost

## Results

| Model | ROC-AUC | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 0.852 | 0.206 | 0.748 | 0.323 |
| Random Forest | 0.830 | 0.334 | 0.424 | 0.374 |
| XGBoost | 0.844 | 0.393 | 0.366 | 0.379 |

## Key Findings

- Logistic Regression achieved the highest ROC-AUC.
- XGBoost produced the strongest balance between precision and F1 score.
- SMOTE improved minority class detection.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- Imbalanced-Learn (SMOTE)

## Author
Merieme Ennajah
[LinkedIn](www.linkedin.com/in/merieme-ennajah-54955338b)
