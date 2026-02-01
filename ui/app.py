import streamlit as st
import requests
import os
from datetime import datetime

DEFAULT_API_URL = os.getenv("SALARY_API_URL", "https://ds-salary-trend.onrender.com/predict")

st.set_page_config(
    page_title="AI Job Salary Predictor",
    layout="centered"
)

st.title("💼 AI Job Salary Predictor")
st.caption("Data-driven salary estimation using machine learning")

# Sidebar config
st.sidebar.subheader("Settings")
API_URL = st.sidebar.text_input(
    "API URL",
    value=DEFAULT_API_URL,
    help="FastAPI predict endpoint, e.g. https://ds-salary-trend.onrender.com/predict",
)

# ======================
# BASIC (HIGH-IMPACT) FEATURES
# ======================

st.subheader("Basic Details")

experience_level = st.selectbox(
    "Experience Level",
    ["Junior", "Mid", "Senior", "Executive"],
    index=2
)

min_experience_years = st.slider(
    "Years of Experience",
    min_value=0,
    max_value=20,
    value=5
)

job_title = st.selectbox(
    "Job Title",
    [
        "Machine Learning Engineer",
        "Data Scientist",
        "Data Analyst",
        "AI Researcher",
        "MLOps Engineer",
        "Applied Scientist"
    ]
)

company_size = st.selectbox(
    "Company Size",
    ["Small", "Medium", "Large"],
    index=1
)

remote_type = st.selectbox(
    "Work Type",
    ["Onsite", "Hybrid", "Remote"],
    index=2
)

# ======================
# ADVANCED (LOW-IMPACT) FEATURES
# ======================

with st.expander("Advanced Options (Optional)"):
    country = st.selectbox(
        "Country",
        ["United States", "India", "Canada", "Germany", "United Kingdom", "Australia"],
        index=0
    )

    industry = st.selectbox(
        "Industry",
        ["Technology", "Finance", "Healthcare", "Education", "Retail"],
        index=0
    )

    company_type = st.selectbox(
        "Company Type",
        ["Startup", "MNC", "Research Lab"],
        index=0
    )

    employment_type = st.selectbox(
        "Employment Type",
        ["Full-time", "Part-time", "Contract", "Freelance"],
        index=0
    )

    posted_year = st.number_input(
        "Posted Year",
        min_value=2018,
        max_value=2030,
        value=datetime.now().year
    )

# ======================
# PREDICTION
# ======================

if st.button("🔮 Predict Salary"):
    payload = {
        "experience_level": experience_level,
        "employment_type": employment_type,
        "job_title": job_title,
        "company_size": company_size,
        "remote_type": remote_type,
        "posted_year": int(posted_year),
        "industry": industry,
        "country": country,
        "company_type": company_type,
        "min_experience_years": int(min_experience_years),
    }

    with st.spinner("Predicting salary..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=30)

            if response.status_code == 200:
                data = response.json()
                salary = data.get("predicted_salary_usd")
                normalized_exp = data.get("normalized_experience")
                st.success(f"💰 Estimated Salary: **${salary:,.0f} USD**")
                if normalized_exp:
                    st.caption(f"Model used experience level: `{normalized_exp}`")

                st.caption(
                    "Note: Advanced options use sensible defaults unless modified."
                )
            else:
                st.error(
                    f"❌ Prediction failed ({response.status_code}). "
                    "Please ensure the API is running and the URL is correct."
                )
                st.code(response.text)

        except Exception as e:
            st.error(f"Error connecting to API: {e}")
