import pandas as pd
import numpy as np
import re
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

# 1. Load Data
df = pd.read_csv("workshop_service_data.csv")
print(f"Total raw rows in CSV: {len(df)}")

# Clean and normalize dates
df["issue_date"] = pd.to_datetime(df["issue_date"], errors="coerce")
df = df.dropna(subset=["issue_date"])

# Normalize string fields
for col in ["vehicle_token", "client_token", "client_type", "oil_grade"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.upper()

# Normalize vehicle tokens to group repeat visits (strip trailing words like SMOKE/EXTRA)
df["vehicle_canonical"] = df["vehicle_token"].apply(
    lambda x: re.sub(r"_(SMOKE|EXTRA|VIC|QLD|SUPERVISOR).*", "", str(x))
)

# Sort chronologically
df = df.sort_values(by=["vehicle_canonical", "issue_date"])

print(f"Unique canonical vehicle tokens: {df['vehicle_canonical'].nunique()}")
counts = df["vehicle_canonical"].value_counts()
print(f"Vehicles with 2+ visits: {(counts >= 2).sum()}")

# 2. Extract Longitudinal Transitions
features = []
y_days = []
y_categories = []

target_subsystems = ["cat_brakes", "cat_cooling", "cat_drivetrain", "cat_electrical"]

# First attempt: Pairwise transitions per vehicle
for entity_id, group in df.groupby("vehicle_canonical"):
    if len(group) >= 2:
        rows = group.to_dict("records")
        for i in range(len(rows) - 1):
            curr = rows[i]
            nxt = rows[i + 1]

            delta_days = (nxt["issue_date"] - curr["issue_date"]).days

            # Avoid same-day duplicate invoices
            if delta_days <= 0 or np.isnan(delta_days):
                continue

            cat_vals = [int(nxt.get(c, 0)) for c in target_subsystems]

            features.append(
                {
                    "client_type": str(curr.get("client_type", "RETAIL")),
                    "odometer_km": curr.get("odometer_km"),
                    "total_aud": float(curr.get("total_aud", 0.0)),
                    "has_warning": int(curr.get("has_warning", 0)),
                    "oil_grade": str(curr.get("oil_grade", "NONE")),
                    "cat_brakes": int(curr.get("cat_brakes", 0)),
                    "cat_cooling": int(curr.get("cat_cooling", 0)),
                    "cat_drivetrain": int(curr.get("cat_drivetrain", 0)),
                    "cat_electrical": int(curr.get("cat_electrical", 0)),
                }
            )
            y_days.append(float(delta_days))
            y_categories.append(cat_vals)

# Fallback: if vehicle-level is too fragmented, pair on repeat client accounts
if len(features) < 3:
    print(
        "Warning: Few vehicle pairs found. Supplementing with repeat client account visits..."
    )
    for client_id, group in df.groupby("client_token"):
        if len(group) >= 2:
            rows = group.sort_values("issue_date").to_dict("records")
            for i in range(len(rows) - 1):
                curr = rows[i]
                nxt = rows[i + 1]
                delta_days = (nxt["issue_date"] - curr["issue_date"]).days
                if delta_days <= 0:
                    continue
                features.append(
                    {
                        "client_type": str(curr.get("client_type", "RETAIL")),
                        "odometer_km": curr.get("odometer_km"),
                        "total_aud": float(curr.get("total_aud", 0.0)),
                        "has_warning": int(curr.get("has_warning", 0)),
                        "oil_grade": str(curr.get("oil_grade", "NONE")),
                        "cat_brakes": int(curr.get("cat_brakes", 0)),
                        "cat_cooling": int(curr.get("cat_cooling", 0)),
                        "cat_drivetrain": int(curr.get("cat_drivetrain", 0)),
                        "cat_electrical": int(curr.get("cat_electrical", 0)),
                    }
                )
                y_days.append(float(delta_days))
                y_categories.append([int(nxt.get(c, 0)) for c in target_subsystems])

X = pd.DataFrame(features)
y_days = np.array(y_days, dtype=np.float64)
y_cat = np.array(y_categories, dtype=np.int32)

print(f"\nFinal feature set shape: {X.shape}")
if len(X) == 0:
    print(
        "Error: No sequential pairs could be created. Check 'workshop_service_data.csv' vehicle/client tokens."
    )
    exit(1)

# 3. Pipeline Setup
cat_cols = ["client_type", "oil_grade"]
num_cols = [c for c in X.columns if c not in cat_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        (
            "cat",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="UNKNOWN"),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            ),
            cat_cols,
        ),
    ]
)

# 4. Fit Models
regressor_pipeline = Pipeline(
    steps=[
        ("prep", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
)

regressor_pipeline.fit(X, y_days)
pred_days = regressor_pipeline.predict(X)
print(f"Regression MAE: {mean_absolute_error(y_days, pred_days):.2f} days")

classifier_pipeline = Pipeline(
    steps=[
        ("prep", preprocessor),
        (
            "model",
            MultiOutputClassifier(
                RandomForestClassifier(n_estimators=100, random_state=42)
            ),
        ),
    ]
)

classifier_pipeline.fit(X, y_cat)
print("Subsystem Multi-Label Classifier fitted successfully.")

# 5. Save Artifacts
joblib.dump(regressor_pipeline, "regressor_model.pkl")
joblib.dump(classifier_pipeline, "classifier_model.pkl")
print("Saved models: 'regressor_model.pkl' & 'classifier_model.pkl'")
