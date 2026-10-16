
from pathlib import Path
import json
import joblib
import pandas as pd

ARTIFACTS = Path("artifacts")

model_path = ARTIFACTS / "lab_cvd_model.joblib"
metadata_path = ARTIFACTS / "lab_cvd_metadata.json"

if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

if not metadata_path.exists():
    raise FileNotFoundError(f"Metadata not found: {metadata_path}")

model = joblib.load(model_path)

with open(metadata_path, encoding="utf-8") as file:
    metadata = json.load(file)

features = metadata["features"]

# Synthetic values for a software smoke test only.
# These are not a real patient's results.
sample = pd.DataFrame([{
    "age": 50,
    "sex": 1,
    "bmi": 25.0,
    "systolic_bp": 120.0,
    "diastolic_bp": 80.0,
    "total_cholesterol": 190.0,
    "hdl": 50.0,
    "diabetes": 0
}])[features]

prediction = model.predict(sample)[0]
score = model.predict_proba(sample)[0, 1]

print("Model loaded successfully.")
print("Features:", features)
print("Smoke-test output class:", int(prediction))
print("Model score (not a clinical probability):", round(float(score), 4))
