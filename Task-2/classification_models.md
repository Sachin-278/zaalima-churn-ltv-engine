# Task 3: Classification Models

This task trains three churn classifiers on the engineered telco dataset:

- **Logistic Regression**: interpretable baseline with coefficient direction and magnitude.
- **Random Forest**: non-linear ensemble that captures feature interactions.
- **XGBoost**: gradient-boosted tree model designed for strong tabular-data performance.

The pipeline uses one-hot encoding for categorical variables, median imputation and scaling for numeric variables, and stratified train/test splitting. Because churn is imbalanced, the comparison reports precision, recall, F1, and ROC-AUC alongside accuracy. The models use balanced class weights where supported so the training objective does not ignore churners.

## Run

Install dependencies if needed:

```bash
pip install pandas numpy scikit-learn xgboost joblib
```

Train all models:

```bash
python Task-2/train_classification_models.py
```

The script writes trained pipelines, per-model classification reports, and a metrics comparison to `Task-2/artifacts/`.
