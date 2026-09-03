from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from feature_engineering import engineer_features


# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = REPO_ROOT / "Task-2" / "artifacts" / "xgboost.joblib"
DATA_PATH = REPO_ROOT / "Task-1" / "Data" / "telco_churn.csv"
ARTIFACTS_DIR = REPO_ROOT / "Task-2" / "artifacts"

# Ensure the artifacts directory exists before saving outputs
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# Validate required input files before loading
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )


# Load trained model
model = joblib.load(MODEL_PATH)

# Load and engineer the data
data = pd.read_csv(DATA_PATH)
data = engineer_features(data)


# Use the same features used by the classification model
feature_columns = [
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


# Validate that all required features are available
missing_features = [
    column for column in feature_columns
    if column not in data.columns
]

if missing_features:
    raise ValueError(
        f"Missing required feature columns: {missing_features}"
    )

X = data[feature_columns]

print(f"Original input features: {len(feature_columns)}")


# Get the preprocessing and XGBoost classifier
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]


# Transform the data using the same preprocessing used during training
X_transformed = preprocessor.transform(X)

# Get feature names after encoding
feature_names = preprocessor.get_feature_names_out()

X_transformed = pd.DataFrame(
    X_transformed,
    columns=feature_names,
    index=X.index,
)


# Create SHAP explainer
explainer = shap.TreeExplainer(classifier)

# Calculate SHAP values
shap_values = explainer.shap_values(X_transformed)


# Validate that SHAP values match the transformed feature count
if shap_values.shape[1] != X_transformed.shape[1]:
    raise ValueError(
        "SHAP values do not match the number of transformed features."
    )

print("SHAP values calculated successfully.")
print("SHAP feature count validation passed.")
print(f"Number of samples: {X_transformed.shape[0]}")
print(f"Number of features: {X_transformed.shape[1]}")


# Create SHAP summary plot
plt.figure()

shap.summary_plot(
    shap_values,
    X_transformed,
    show=False,
)

plt.tight_layout()

summary_path = ARTIFACTS_DIR / "shap_summary.png"
plt.savefig(summary_path, bbox_inches="tight")
plt.close()

print(f"SHAP summary plot saved to: {summary_path}")


# Calculate mean absolute SHAP importance
importance = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": abs(shap_values).mean(axis=0),
})

importance = importance.sort_values(
    "mean_abs_shap",
    ascending=False,
)


# Round SHAP importance values for consistent reporting
importance["mean_abs_shap"] = importance["mean_abs_shap"].round(6)


# Save SHAP feature importance
importance_path = ARTIFACTS_DIR / "shap_feature_importance.csv"
importance.to_csv(importance_path, index=False)

# Validate that the feature importance output was created successfully
if not importance_path.exists():
    raise FileNotFoundError(
        f"SHAP feature importance file was not created: {importance_path}"
    )

print(f"SHAP feature importance saved to: {importance_path}")


# Display top 10 features
print("\nTop 10 features influencing churn predictions:")
print(importance.head(10).to_string(index=False))


# Display the most influential feature
top_feature = importance.iloc[0]

print("\nMost influential feature:")
print(f"Feature: {top_feature['feature']}")
print(f"Mean absolute SHAP value: {top_feature['mean_abs_shap']:.6f}")


# Display completion message
print("\nSHAP analysis completed successfully.")
