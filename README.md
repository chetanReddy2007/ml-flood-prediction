# FloodSense 🌊 — ML-Powered Flood Risk Prediction

An intelligent early-warning flood prediction web application powered by **XGBoost** machine learning and built with **Flask**.

---

## 🚀 Features

- **Real-time flood risk prediction** — High / Low verdict in under 1 second
- **Confidence score** — probabilistic output via `predict_proba()`
- **5-feature ML model** — Annual Rainfall, Seasonal Rainfall, Cloud Visibility, Temperature, Humidity
- **95.1% accuracy** — XGBoost classifier trained on 5,000 synthetic weather samples
- **Premium dark UI** — Glassmorphism, particle animations, Bootstrap 5
- **Production-ready** — Gunicorn + Procfile for Heroku / Render deployment
- **Load-tested** — Locust scripts for scalability validation

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.14, Flask 3.1 |
| ML | XGBoost 3.3, Scikit-learn 1.9, Pandas 3.0, NumPy 2.5 |
| Frontend | Bootstrap 5, HTML5, Vanilla JS |
| Deployment | Gunicorn, Procfile, runtime.txt |
| Load Testing | Locust 2.44 |

---

## 📁 Project Structure

```
FloodSense/
├── app.py                  # Flask backend (routes: /, /about, /predict)
├── train_model.py          # ML pipeline — data gen, train, save
├── locustfile.py           # Load testing script
├── requirements.txt        # Python dependencies
├── Procfile                # Gunicorn deployment command
├── runtime.txt             # Python version spec
├── .gitignore
├── Model/
│   ├── flood_model.pkl     # Trained XGBoost model
│   └── scaler.pkl          # Fitted StandardScaler
└── Templates/
    ├── index.html          # Dashboard + prediction form
    ├── about.html          # Project info page
    └── result.html         # Prediction results page
```

---

## ⚙️ Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/chetanReddy2007/ml-flood-prediction.git
cd ml-flood-prediction
```

### 2. Create and activate a virtual environment
```bash
# Windows
py -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt --only-binary=:all:
```

### 4. Train the model
```bash
python train_model.py
```
This creates `Model/flood_model.pkl` and `Model/scaler.pkl`.

### 5. Run the Flask app
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser.

---

## 🧪 Load Testing with Locust

While the app is running:
```bash
locust -f locustfile.py --host=http://localhost:5000
```
Then open **http://localhost:8089**, set the number of users and spawn rate, and click **Start**.

---

## ☁️ Deployment (Heroku / Render)

```bash
# Heroku
heroku create
git push heroku main

# The Procfile already contains:
# web: gunicorn app:app
```

---

## 📊 Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | XGBoost Classifier |
| Training samples | 4,000 (80% split) |
| Test samples | 1,000 (20% split) |
| Accuracy | **95.1%** |
| Estimators | 300 |
| Max depth | 6 |
| Learning rate | 0.05 |
| Subsample | 0.8 |
| Feature scaling | StandardScaler |

---

## 📸 Pages

| Page | Route | Description |
|------|-------|-------------|
| Dashboard | `/` | Input form with client-side validation |
| Result | `/predict` | Risk verdict + confidence + input summary |
| About | `/about` | Project info, tech stack, ML pipeline |

---

## 📄 License

MIT License — free to use, modify and distribute.
