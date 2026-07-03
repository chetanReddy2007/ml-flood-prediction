"""
locustfile.py
─────────────
Load-testing script for the FloodSense Flask application using Locust.

Usage:
  # Start the Locust web UI (open http://localhost:8089 in a browser):
      locust -f locustfile.py --host=http://localhost:5000

  # Headless / CI mode (100 users, 10 users spawned per second, run 60 s):
      locust -f locustfile.py --host=http://localhost:5000 \
             --headless -u 100 -r 10 --run-time 60s

The script simulates two types of user behaviour:
  1. FloodUser   – visits the home page and submits prediction forms.
  2. BrowserUser – browses the home and about pages without submitting.
"""

import random
from locust import HttpUser, task, between


# ─────────────────────────────────────────────────────────────────────────────
# Helper: generate realistic random weather data
# ─────────────────────────────────────────────────────────────────────────────
def random_weather_payload() -> dict:
    """Return a dict of form values within the accepted input ranges."""
    return {
        "annual_rainfall":   round(random.uniform(0,    5000), 1),
        "seasonal_rainfall": round(random.uniform(0,    3000), 1),
        "cloud_visibility":  round(random.uniform(0,    10),   1),
        "temperature":       round(random.uniform(-20,  60),   1),
        "humidity":          round(random.uniform(0,    100),  1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FloodUser: primary simulated user — visits the dashboard and predicts
# ─────────────────────────────────────────────────────────────────────────────
class FloodUser(HttpUser):
    """
    Simulates a user who:
      - Loads the dashboard (GET /)
      - Submits the prediction form (POST /predict)

    Think time: 1–3 seconds between requests (mimics real user behaviour).
    Weight: 3 — 3× more frequent than BrowserUser.
    """

    wait_time = between(1, 3)
    weight    = 3

    @task(1)
    def load_dashboard(self):
        """Load the main dashboard page."""
        with self.client.get(
            "/",
            name   = "GET /  (Dashboard)",
            catch_response = True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(
                    f"Unexpected status {response.status_code} on GET /"
                )

    @task(3)
    def submit_prediction(self):
        """Submit a prediction with random weather parameters."""
        payload = random_weather_payload()
        with self.client.post(
            "/predict",
            data           = payload,
            name           = "POST /predict",
            catch_response = True,
            allow_redirects= True,
        ) as response:
            if response.status_code == 200:
                # Verify the result page contains expected content
                if "Prediction Result" in response.text or \
                   "Flood Risk" in response.text:
                    response.success()
                else:
                    response.failure("Result page missing expected content")
            else:
                response.failure(
                    f"POST /predict returned status {response.status_code}"
                )

    @task(1)
    def load_about(self):
        """Load the About page."""
        with self.client.get(
            "/about",
            name   = "GET /about",
            catch_response = True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(
                    f"Unexpected status {response.status_code} on GET /about"
                )


# ─────────────────────────────────────────────────────────────────────────────
# BrowserUser: lighter user that only browses pages (no form submission)
# ─────────────────────────────────────────────────────────────────────────────
class BrowserUser(HttpUser):
    """
    Simulates a user who navigates the site but does not submit the form.
    Represents search-engine crawlers or casual visitors.
    Think time: 2–5 seconds. Weight: 1.
    """

    wait_time = between(2, 5)
    weight    = 1

    @task(2)
    def view_home(self):
        """Visit the home/dashboard page."""
        self.client.get("/", name="GET /  (Browse)")

    @task(1)
    def view_about(self):
        """Visit the About page."""
        self.client.get("/about", name="GET /about (Browse)")
