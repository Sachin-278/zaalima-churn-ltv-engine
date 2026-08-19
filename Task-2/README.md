# Task 2: Feature Engineering

This task creates derived variables to improve churn and lifetime-value modeling.

## New engineered features

- `avg_monthly_usage_vs_charge`: ratio of usage to charge, useful for spotting customers who feel the value is low.
- `tenure_bucket`: tenure grouped into lifecycle bands such as 0-6 months, 6-12 months, 12-24 months, and longer.
- `num_services_subscribed`: total number of telecom add-ons or bundled services tied to the account.
- `charges_per_tenure`: charge level normalized by customer tenure.

## Run

```bash
python Task-2/feature_engineering.py
```

This reads the telco dataset from `Task-1/Data/telco_churn.csv` and writes the transformed dataset to `Task-2/engineered_telco_data.csv`.
