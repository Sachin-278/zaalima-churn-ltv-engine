from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SERVICE_COLUMNS = [
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'MultipleLines',
]


def clean_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().replace('', np.nan)
    return pd.to_numeric(cleaned, errors='coerce')


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['TotalCharges'] = clean_numeric_series(df['TotalCharges'])
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(0)

    df['avg_monthly_usage_vs_charge'] = np.where(
        df['tenure'] > 0,
        (df['TotalCharges'] / df['tenure']) / df['MonthlyCharges'],
        1.0,
    )
    df['avg_monthly_usage_vs_charge'] = (
        df['avg_monthly_usage_vs_charge']
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
    )

    tenure_bins = [0, 6, 12, 24, 36, 48, 60, 120, np.inf]
    tenure_labels = ['0-6m', '6-12m', '12-24m', '24-36m', '36-48m', '48-60m', '60-120m', '120m+']
    df['tenure_bucket'] = pd.cut(
        df['tenure'],
        bins=tenure_bins,
        labels=tenure_labels,
        include_lowest=True,
        right=True,
    )

    usage_map = {
        'Yes': 1,
        'No': 0,
        'No internet service': 0,
    }
    df['num_services_subscribed'] = (
        df[SERVICE_COLUMNS]
        .replace(usage_map)
        .apply(pd.to_numeric, errors='coerce')
        .sum(axis=1)
    )

    df['charges_per_tenure'] = np.where(
        df['tenure'] > 0,
        df['TotalCharges'] / df['tenure'],
        df['MonthlyCharges'],
    )

    return df


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / 'Task-1' / 'Data' / 'telco_churn.csv'
    output_path = repo_root / 'Task-2' / 'engineered_telco_data.csv'

    df = pd.read_csv(input_path)
    engineered = engineer_features(df)
    engineered.to_csv(output_path, index=False)

    print('Engineered features added:')
    for feature in [
        'avg_monthly_usage_vs_charge',
        'tenure_bucket',
        'num_services_subscribed',
        'charges_per_tenure',
    ]:
        print('-', feature)

    print('\nPreview:')
    preview = engineered[['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'avg_monthly_usage_vs_charge', 'tenure_bucket', 'num_services_subscribed', 'charges_per_tenure']].head(5)
    print(preview.to_string(index=False))


if __name__ == '__main__':
    main()
