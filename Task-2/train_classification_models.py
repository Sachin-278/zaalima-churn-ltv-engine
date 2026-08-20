from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier


FEATURE_COLUMNS = [
    'gender',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'tenure',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod',
    'MonthlyCharges',
    'TotalCharges',
    'avg_monthly_usage_vs_charge',
    'tenure_bucket',
    'num_services_subscribed',
    'charges_per_tenure',
]
TARGET = 'Churn'


def load_feature_engineering():
    module_path = Path(__file__).resolve().with_name('feature_engineering.py')
    spec = importlib.util.spec_from_file_location('feature_engineering', module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Could not load feature engineering module: {module_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.engineer_features


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    return ColumnTransformer([
        ('numeric', numeric_pipeline, numeric_columns),
        ('categorical', categorical_pipeline, categorical_columns),
    ])


def build_models(class_weight: str = 'balanced') -> dict[str, object]:
    return {
        'logistic_regression': LogisticRegression(
            max_iter=2000,
            class_weight=class_weight,
            random_state=42,
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=300,
            class_weight=class_weight,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
        'xgboost': XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            n_jobs=-1,
            random_state=42,
        ),
    }


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions, zero_division=0),
        'recall': recall_score(y_test, predictions, zero_division=0),
        'f1': f1_score(y_test, predictions, zero_division=0),
        'roc_auc': roc_auc_score(y_test, probabilities),
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / 'Task-1' / 'Data' / 'telco_churn.csv'
    artifacts_dir = repo_root / 'Task-2' / 'artifacts'
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    engineer_features = load_feature_engineering()
    data = engineer_features(pd.read_csv(input_path))
    data[TARGET] = data[TARGET].map({'No': 0, 'Yes': 1})
    data = data.dropna(subset=[TARGET])

    X = data[FEATURE_COLUMNS]
    y = data[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    results: dict[str, dict[str, float]] = {}
    for name, classifier in build_models().items():
        pipeline = Pipeline([
            ('preprocessor', build_preprocessor(X_train)),
            ('classifier', classifier),
        ])
        pipeline.fit(X_train, y_train)
        results[name] = evaluate_model(pipeline, X_test, y_test)
        joblib.dump(pipeline, artifacts_dir / f'{name}.joblib')
        report = classification_report(
            y_test,
            pipeline.predict(X_test),
            target_names=['Stayed', 'Churned'],
            zero_division=0,
        )
        (artifacts_dir / f'{name}_classification_report.txt').write_text(report, encoding='utf-8')

    (artifacts_dir / 'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    metrics = pd.DataFrame(results).T.sort_values('f1', ascending=False)
    metrics.to_csv(artifacts_dir / 'metrics.csv', index_label='model')
    print(metrics.to_string(float_format=lambda value: f'{value:.4f}'))
    print(f'\nArtifacts saved to: {artifacts_dir}')


if __name__ == '__main__':
    main()
