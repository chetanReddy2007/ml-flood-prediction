"""
train_model.py
--------------
Machine Learning pipeline for Flood Prediction.

Steps:
  1. Generate a synthetic dataset with realistic weather features.
  2. Pre-process data (handle missing values, scale features).
  3. Split into train/test sets (80/20).
  4. Train an XGBoost classifier.
  5. Evaluate and print accuracy.
  6. Persist the trained model and scaler to the Model/ directory.

Run this script once before starting the Flask app:
    python train_model.py
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
N_SAMPLES   = 5000          # Number of synthetic data points
MODEL_DIR   = "Model"       # Directory to save model artefacts
MODEL_PATH  = os.path.join(MODEL_DIR, "flood_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Ensure the Model/ directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 2. SYNTHETIC DATASET GENERATION
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Generating synthetic dataset ...")

# Feature distributions (realistic meteorological ranges)
annual_rainfall   = np.random.uniform(200,  3000, N_SAMPLES)   # mm/year
seasonal_rainfall = np.random.uniform(50,   1500, N_SAMPLES)   # mm/season
cloud_visibility  = np.random.uniform(0,    10,   N_SAMPLES)   # km  (lower = more cloud)
temperature       = np.random.uniform(10,   45,   N_SAMPLES)   # °C
humidity          = np.random.uniform(20,   100,  N_SAMPLES)   # %

# --- Rule-based labelling (mimics domain knowledge) --------------------------
# A location is at high flood risk when:
#   • Annual rainfall is very high (> 1800 mm) OR
#   • Seasonal rainfall is very high (> 900 mm) OR
#   • Humidity is high (> 80 %) AND rainfall is moderate-to-high (> 1000 mm)
#   • Low cloud visibility (< 3 km) combined with high seasonal rain (> 600 mm)

flood_risk = (
    (annual_rainfall   > 1800) |
    (seasonal_rainfall > 900)  |
    ((humidity         > 80)   & (annual_rainfall   > 1000)) |
    ((cloud_visibility < 3)    & (seasonal_rainfall > 600))
).astype(int)

# Add small random noise (5 % label flip) to make the task non-trivial
noise_idx = np.random.choice(N_SAMPLES, size=int(0.05 * N_SAMPLES), replace=False)
flood_risk[noise_idx] = 1 - flood_risk[noise_idx]

# Build the DataFrame
df = pd.DataFrame({
    "Annual_Rainfall":   annual_rainfall,
    "Seasonal_Rainfall": seasonal_rainfall,
    "Cloud_Visibility":  cloud_visibility,
    "Temperature":       temperature,
    "Humidity":          humidity,
    "Flood_Risk":        flood_risk,
})

print(f"  Dataset shape  : {df.shape}")
print(f"  Class balance  : {df['Flood_Risk'].value_counts().to_dict()}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. PRE-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[INFO] Pre-processing ...")

# 3a. Check for missing values and fill with column median (robust to outliers)
missing = df.isnull().sum().sum()
if missing > 0:
    print(f"  Found {missing} missing values — imputing with column medians.")
    df.fillna(df.median(numeric_only=True), inplace=True)
else:
    print("  No missing values found.")

# 3b. Separate features and target
FEATURE_COLS = ["Annual_Rainfall", "Seasonal_Rainfall",
                "Cloud_Visibility", "Temperature", "Humidity"]
X = df[FEATURE_COLS].values
y = df["Flood_Risk"].values

# 3c. Train / Test split (stratified to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.20,
    random_state = RANDOM_SEED,
    stratify     = y,
)
print(f"  Training samples : {X_train.shape[0]}")
print(f"  Testing  samples : {X_test.shape[0]}")

# 3d. Feature scaling (fit ONLY on training data to prevent leakage)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[INFO] Training XGBoost classifier ...")

model = XGBClassifier(
    n_estimators     = 300,
    max_depth        = 6,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    eval_metric      = "logloss",   # use_label_encoder removed in XGBoost 2.x
    random_state     = RANDOM_SEED,
    n_jobs           = -1,          # use all available CPU cores
)

model.fit(
    X_train, y_train,
    eval_set            = [(X_test, y_test)],
    verbose             = False,
)

print("  Training complete.")

# ─────────────────────────────────────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[INFO] Evaluating model …")
y_pred   = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"  Accuracy : {accuracy * 100:.2f} %")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))

# ─────────────────────────────────────────────────────────────────────────────
# 6. PERSIST ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────
print("[INFO] Saving artefacts ...")
joblib.dump(model,  MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print(f"  Model  saved -> {MODEL_PATH}")
print(f"  Scaler saved -> {SCALER_PATH}")
print("\n[DONE] Model training complete. You can now run the Flask app.")
