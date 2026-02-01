from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import logging
from pathlib import Path


# LOGGING

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# LOAD MODEL

BASE_DIR = Path(__file__).resolve().parent.parent
FINAL_MODEL_PATH = BASE_DIR / "models" / "final_model1.pkl"
RIDGE_MODEL_PATH = BASE_DIR / "models" / "ridge_salary_pipeline.pkl"

if FINAL_MODEL_PATH.exists():
    MODEL_PATH = FINAL_MODEL_PATH
elif RIDGE_MODEL_PATH.exists():
    MODEL_PATH = RIDGE_MODEL_PATH
else:
    raise FileNotFoundError(
        f"No model file found. Looked for '{FINAL_MODEL_PATH}' and '{RIDGE_MODEL_PATH}'."
    )

model = joblib.load(MODEL_PATH)
logger.info(f"Loaded model from: {MODEL_PATH}")

app = FastAPI(title="AI Job Salary Prediction API")


# INPUT SCHEMA

class JobFeatures(BaseModel):
    experience_level: str
    min_experience_years: int
    employment_type: str
    job_title: str
    company_size: str
    remote_type: str
    posted_year: int
    industry: str
    country: str
    company_type: str


VALID_EXPERIENCE = ["Entry", "Mid", "Senior"]

def normalize_experience(exp: str) -> str:
    """
    Normalizes user input to the canonical experience_level categories.
    Accepts UI values (Junior/Mid/Senior/Executive) and common variants.

    Note: The model is not trained on 'Executive', so we map executive-like inputs to 'Senior'.
    """
    if exp is None:
        return "Mid"

    raw = exp.strip()
    e = raw.lower()

    # Entry / Junior
    if e.startswith("junior") or e in {"jr", "entry", "entry-level", "entry level"}:
        return "Entry"

    # Mid
    if e.startswith("mid") or e in {"mid-level", "mid level", "intermediate"}:
        return "Mid"

    # Senior
    if e.startswith("senior") or e in {"sr", "sr.", "senior-level", "senior level"}:
        return "Senior"

    # Executive (map to Senior)
    if e.startswith("executive") or e in {"lead", "principal", "director", "head"}:
        return "Senior"

    
    return "Mid"


def safe_category(value: str, fallback: str = "Other") -> str:
    """
    Ensures unseen categories do not crash encoders.
    """
    if value is None or value.strip() == "":
        return fallback
    return value.strip()



# PREDICTION ENDPOINT

@app.post("/predict")
def predict(features: JobFeatures):

    # -------- EXPERIENCE NORMALIZATION --------
    experience_level = normalize_experience(features.experience_level)

    # -------- LOG INPUT --------
    logger.info(f"RAW experience_level: {features.experience_level}")
    logger.info(f"NORMALIZED experience_level: {experience_level}")

    # INPUT
    data = {
        "experience_level": experience_level,
        "min_experience_years": max(features.min_experience_years, 0),
        "employment_type": safe_category(features.employment_type, "Full-time"),
        "job_title": safe_category(features.job_title, "Data Scientist"),
        "company_size": safe_category(features.company_size, "Medium"),
        "remote_type": safe_category(features.remote_type, "Remote"),
        "posted_year": max(features.posted_year, 2020),
        "industry": safe_category(features.industry, "Technology"),
        "country": safe_category(features.country, "United States"),
        "company_type": safe_category(features.company_type, "MNC")
    }

    df = pd.DataFrame([data])
    logger.info(f"MODEL INPUT DF:\n{df}")

    # PREDICT SALARY
    try:
        prediction = model.predict(df)[0]
        return {
            "predicted_salary_usd": round(prediction),
            "normalized_experience": experience_level
        }
   
    except Exception as e:
        import traceback
        traceback.print_exc()   
        return {
            "error": "Prediction failed",
            "exception": str(e)
        }
