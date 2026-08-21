from pathlib import Path

import joblib
import pandas as pd
import shap


# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = REPO_ROOT / "Task-2" / "artifacts" / "xgboost.joblib"
DATA_PATH = REPO_ROOT / "Task-1" / "Data" / "telco_churn.csv"


# Load trained XGBoost pipeline
model = joblib.load(MODEL_PATH)

# Load original dataset
data = pd.read_csv(DATA_PATH)

print("Model loaded successfully.")
print("Dataset loaded successfully.")
print(f"Dataset shape: {data.shape}")
