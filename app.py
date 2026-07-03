"""
app.py
------
Flask web application for the Flood Prediction System.

Routes:
  GET  /         → renders index.html (input form)
  GET  /about    → renders about.html (project info)
  GET  /predict  → redirects to /  (guard against direct GET access)
  POST /predict  → reads form inputs, applies scaler, runs model,
                   renders result.html with prediction & confidence

Usage (development):
    python app.py

Usage (production with Gunicorn):
    gunicorn app:app
"""

import os
import numpy as np
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash

# ─────────────────────────────────────────────────────────────────────────────
# INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "flood-prediction-secret-2024")

# Paths to persisted artefacts (relative to this file)
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "Model", "flood_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "Model", "scaler.pkl")

# Load model and scaler at startup (avoids reloading on every request)
try:
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("[INFO] Model and scaler loaded successfully.")
except FileNotFoundError as exc:
    model  = None
    scaler = None
    print(f"[WARNING] Could not load model artefacts: {exc}")
    print("[WARNING] Run train_model.py first, then restart the app.")

# Feature names in the exact order the model was trained on
FEATURE_NAMES = [
    "Annual_Rainfall",
    "Seasonal_Rainfall",
    "Cloud_Visibility",
    "Temperature",
    "Humidity",
]

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Render the main dashboard / input form."""
    return render_template("index.html")


@app.route("/about")
def about():
    """Render the About page."""
    return render_template("about.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    GET  → redirect back to the form (user shouldn't land here directly).
    POST → parse form, scale inputs, predict, return results.
    """
    if request.method == "GET":
        return redirect(url_for("index"))

    # ── 1. Check model is loaded ──────────────────────────────────────────
    if model is None or scaler is None:
        flash(
            "Model not found. Please run train_model.py first, "
            "then restart the server.",
            "danger",
        )
        return redirect(url_for("index"))

    # ── 2. Parse & validate form inputs ──────────────────────────────────
    try:
        annual_rainfall   = float(request.form["annual_rainfall"])
        seasonal_rainfall = float(request.form["seasonal_rainfall"])
        cloud_visibility  = float(request.form["cloud_visibility"])
        temperature       = float(request.form["temperature"])
        humidity          = float(request.form["humidity"])
    except (KeyError, ValueError) as exc:
        flash(f"Invalid input: {exc}. Please fill in all fields correctly.", "danger")
        return redirect(url_for("index"))

    # ── 3. Scale the input features ───────────────────────────────────────
    raw_features = np.array([[
        annual_rainfall,
        seasonal_rainfall,
        cloud_visibility,
        temperature,
        humidity,
    ]])
    scaled_features = scaler.transform(raw_features)

    # ── 4. Predict ────────────────────────────────────────────────────────
    prediction_class = int(model.predict(scaled_features)[0])
    probabilities    = model.predict_proba(scaled_features)[0]  # [P(0), P(1)]

    # Human-readable label
    if prediction_class == 1:
        prediction_label = "High Flood Risk"
        confidence       = round(float(probabilities[1]) * 100, 2)
        risk_class       = "danger"          # Bootstrap colour class
    else:
        prediction_label = "Low Flood Risk"
        confidence       = round(float(probabilities[0]) * 100, 2)
        risk_class       = "success"         # Bootstrap colour class

    # ── 5. Collect user-friendly input summary for display ─────────────
    input_summary = {
        "Annual Rainfall (mm/year)":   annual_rainfall,
        "Seasonal Rainfall (mm/season)": seasonal_rainfall,
        "Cloud Visibility (km)":       cloud_visibility,
        "Temperature (°C)":            temperature,
        "Humidity (%)":                humidity,
    }

    return render_template(
        "result.html",
        prediction       = prediction_label,
        confidence       = confidence,
        risk_class       = risk_class,
        input_summary    = input_summary,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # debug=True is convenient for development; set debug=False in production
    app.run(debug=True, host="0.0.0.0", port=5000)
