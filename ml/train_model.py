"""
AI BASED FOOD SYSTEM — Model Training Script
=============================================
Trains a Linear Regression model to predict Expected_Students
from Day_of_Week, Meal_Type, Food_Item, and Popularity_Index.

Usage:
    python ml/train_model.py

Output:
    ml/meal_demand_model.pkl   — trained model pipeline
    ml/model_metadata.json     — label encoders + evaluation metrics
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET    = os.path.join(BASE_DIR, "meal_dataset.xlsx")  # <-- put your Excel file here
MODEL_OUT  = os.path.join(BASE_DIR, "meal_demand_model.pkl")
META_OUT   = os.path.join(BASE_DIR, "model_metadata.json")

# ── 1. Load dataset (seed + actuals combined) ────────────────────────────────
print("📂 Loading dataset...")
df_seed = pd.read_excel(DATASET, engine="openpyxl")
df_seed.columns = df_seed.columns.str.strip()
print(f"   Seed rows: {len(df_seed)}")

# Also load actuals_log.xlsx if it exists (real data submitted by users)
ACTUALS_LOG = os.path.join(BASE_DIR, "actuals_log.xlsx")
if os.path.exists(ACTUALS_LOG):
    df_actuals = pd.read_excel(ACTUALS_LOG, engine="openpyxl")
    df_actuals.columns = df_actuals.columns.str.strip()
    # Keep only the training columns from actuals
    KEEP = ["Day_of_Week", "Meal_Type", "Food_Item", "Popularity_Index", "Expected_Students"]
    df_actuals = df_actuals[[c for c in KEEP if c in df_actuals.columns]]
    df = pd.concat([df_seed, df_actuals], ignore_index=True)
    print(f"   Actuals rows: {len(df_actuals)}")
    print(f"   Combined rows: {len(df)}")
else:
    df = df_seed
    print(f"   No actuals log yet — using seed only")

print(f"   Columns: {list(df.columns)}")

# ── 2. Validate required columns ─────────────────────────────────────────────
REQUIRED = ["Day_of_Week", "Meal_Type", "Food_Item", "Popularity_Index", "Expected_Students"]
missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing columns: {missing}")

# Drop rows with nulls in key columns
df.dropna(subset=REQUIRED, inplace=True)
print(f"   Clean rows after null-drop: {len(df)}")

# ── 3. Encode categorical features ───────────────────────────────────────────
CATEGORICALS = ["Day_of_Week", "Meal_Type", "Food_Item"]
encoders = {}

for col in CATEGORICALS:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].str.strip())
    encoders[col] = {
        "classes": le.classes_.tolist()   # save for decoding / validation
    }
    print(f"   Encoded '{col}': {le.classes_.tolist()}")

# ── 4. Define features and target ────────────────────────────────────────────
FEATURE_COLS = ["Day_of_Week_enc", "Meal_Type_enc", "Food_Item_enc", "Popularity_Index"]
TARGET_COL   = "Expected_Students"

X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

# ── 5. Train / test split (only if enough data) ───────────────────────────────
MIN_ROWS_FOR_SPLIT = 30  # need at least 30 rows to meaningfully split

if len(df) >= MIN_ROWS_FOR_SPLIT:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n🔀 Train samples: {len(X_train)}  |  Test samples: {len(X_test)}")
    eval_on_test = True
else:
    print(f"\n⚠️  Only {len(df)} rows — training on full dataset (no hold-out split).")
    print(f"   Add more data to your Excel file for a proper train/test evaluation.")
    X_train, y_train = X, y
    X_test,  y_test  = X, y   # evaluate on training data as an approximation
    eval_on_test = False

# ── 6. Train model ────────────────────────────────────────────────────────────
print("\n🧠 Training Random Forest model...")
model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ── 7. Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

eval_note = "(training data — add 30+ rows for a true test split)" if not eval_on_test else "(held-out test set)"
print(f"\n📊 Evaluation Results {eval_note}:")
print(f"   MAE  (Mean Absolute Error):          {mae:.2f} students")
print(f"   RMSE (Root Mean Squared Error):      {rmse:.2f} students")
print(f"   R²   (Coefficient of Determination): {r2:.4f}")

# ── 8. Show predictions vs actuals ───────────────────────────────────────────
print(f"\n🔍 Sample Predictions vs Actuals:")
print(f"   {'Predicted':>12}  {'Actual':>8}  {'Diff':>8}")
for pred, actual in zip(y_pred[:8], y_test[:8]):
    diff = pred - actual
    print(f"   {pred:>12.1f}  {actual:>8.0f}  {diff:>+8.1f}")

# ── 9. Save model ─────────────────────────────────────────────────────────────
with open(MODEL_OUT, "wb") as f:
    pickle.dump(model, f)
print(f"\n✅ Model saved  → {MODEL_OUT}")

# ── 10. Save metadata (encoders + metrics) ───────────────────────────────────
metadata = {
    "feature_columns": FEATURE_COLS,
    "target_column": TARGET_COL,
    "encoders": encoders,
    "metrics": {
        "MAE":  round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2":   round(r2, 4)
    },
    "model_type": "RandomForestRegressor"
}

with open(META_OUT, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"✅ Metadata saved → {META_OUT}")
print("\n🎉 Training complete!")