# Employee Attrition Prediction

Predicting which employees are at risk of leaving, using the IBM HR Analytics Employee Attrition
dataset — an end-to-end classification project covering EDA, statistical analysis, preprocessing,
feature engineering, model comparison, and business recommendations.

## Problem

Employee attrition is expensive: every voluntary departure costs an organization in recruiting,
onboarding, and lost institutional knowledge. This project builds a machine learning model that
predicts whether an employee is likely to leave (`Attrition`: Yes/No), based on factors such as
job role, satisfaction levels, work-life balance, tenure, salary, and performance — and translates
the findings into concrete HR retention actions.

## Dataset

**IBM HR Analytics Employee Attrition Dataset** — 1,470 employees, 35 columns, no missing values,
no duplicates. [Original source on Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset).
A copy is included in [`data/`](data/) for convenience/reproducibility.

## Repository structure

```
.
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv   # dataset
├── notebooks/
│   └── Employee_Attrition_Prediction.ipynb     # full, already-executed analysis notebook
├── src/
│   └── attrition_pipeline.py                   # same analysis as a plain .py script
├── requirements.txt
└── README.md
```

## Workflow

`Data → Statistics → EDA → Preprocessing → Feature Engineering → Modeling → Evaluation → Insights → Recommendations`

1. **Data Understanding** — shape, dtypes, missing/duplicate checks, dropped 3 zero-variance
   columns (`EmployeeCount`, `Over18`, `StandardHours`) and the `EmployeeNumber` ID column.
2. **Statistical Analysis** — mean/median/mode/variance/IQR, skewness, correlation structure.
3. **EDA** — univariate, bivariate, and multivariate analysis of attrition against job role,
   department, overtime, satisfaction, tenure, and compensation.
4. **Preprocessing** — label/one-hot encoding, stratified 80/20 train-test split (to preserve the
   class imbalance ratio in both sets), feature scaling.
5. **Feature Engineering** — 4 engineered features: `TenureRatio`, `PromotionStagnation`,
   `AvgSatisfaction`, `IncomePerJobLevel`.
6. **Modeling** — 5 classifiers: Logistic Regression, Decision Tree, Random Forest, AdaBoost, KNN.
7. **Evaluation** — Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC. Because only ~16%
   of employees in the data actually left, **Recall and F1 on the "Yes" class are prioritized over
   raw accuracy**, which would be misleadingly high for a model that just predicts "No" every time.

## Key findings

- **Attrition is imbalanced**: 83.9% stayed, 16.1% left.
- **Sales Representative** has by far the highest attrition rate of any role (**39.8%**), followed
  by Laboratory Technician (23.9%) and Human Resources (23.1%).
- **Overtime, frequent business travel, low satisfaction/work-life balance, and early tenure** are
  the strongest behavioural and demographic attrition signals.

## Model comparison (test set results)

| Model | Train Acc | Test Acc | Precision | Recall | F1 | ROC-AUC | Train-Test Gap |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.781 | 0.745 | 0.341 | **0.638** | **0.444** | 0.806 | 0.036 |
| AdaBoost | 0.900 | **0.854** | **0.577** | 0.319 | 0.411 | 0.783 | 0.046 |
| Decision Tree | 0.861 | 0.741 | 0.290 | 0.426 | 0.345 | 0.593 | 0.120 |
| Random Forest | 0.991 | 0.823 | 0.381 | 0.170 | 0.235 | 0.778 | 0.168 |
| KNN | 0.862 | 0.847 | 0.667 | 0.085 | 0.151 | 0.689 | 0.015 |

**Final model: Logistic Regression.** It has the best Recall (catches 63.8% of employees who
actually left) and F1 Score, with a small, healthy train-test gap — meaning it generalizes well.
Random Forest's near-perfect training accuracy (99.1%) but weak test recall (17.0%) is a textbook
overfitting signature, not a sign of a better model. AdaBoost is a solid secondary option if the
business prefers higher precision (fewer false alarms) over higher recall.

## Business recommendations

1. **Prioritize Sales Representative retention** — attrition there is ~2.5x the company average.
2. **Audit overtime policy**, especially in Sales and Lab Technician roles.
3. **Review compensation and pay-per-level equity** — income-related features were top model signals.
4. **Build an early-tenure retention program** — younger, shorter-tenure employees leave disproportionately.
5. **Use the model as a prioritization tool, not an automated decision-maker** — pair its "at-risk"
   list with human judgment (stay interviews, manager check-ins) rather than acting on it alone.

## How to run

### Option 1 — Google Colab
1. Open [`notebooks/Employee_Attrition_Prediction.ipynb`](notebooks/Employee_Attrition_Prediction.ipynb) in Colab.
2. Upload `data/WA_Fn-UseC_-HR-Employee-Attrition.csv` to the Colab session (there's an upload cell at the top).
3. `Runtime → Run all`.

### Option 2 — Locally
```bash
git clone <this-repo-url>
cd employee-attrition-prediction
pip install -r requirements.txt
jupyter notebook notebooks/Employee_Attrition_Prediction.ipynb
```
or run the plain script version:
```bash
python src/attrition_pipeline.py
```

## Limitations

- Trained on a single company snapshot (1,470 employees) — drivers may not transfer to other
  industries, geographies, or company cultures.
- Class imbalance means even the best model has an imperfect precision/recall trade-off.
- Feature importance shows association, not proven causation.
- The dataset can't capture everything that drives someone to quit (manager quality, external
  offers, personal circumstances).

## License

MIT — see [LICENSE](LICENSE).
