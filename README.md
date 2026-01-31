💼 AI Job Salary Prediction

An end-to-end Machine Learning project that predicts AI & Data Science job salaries using real-world job market data.

🚀 Overview

📊 Real dataset (50k+ job postings)

🧠 Feature engineering with Scikit-learn pipelines

📈 Model tracking using MLflow

🤖 Models: Ridge Regression & Random Forest

⚡ FastAPI backend for predictions

🎨 Simple UI for user interaction

📂 Project Structure
job-salary-ml/
├── data/raw/ai_jobs.csv
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_mlflow_tracking.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│   └── 07_final_model.ipynb
├── src/salary_api.py
├── ui/app.py
├── requirements.txt
└── README.md

🧠 ML Workflow

Data Understanding & EDA

Feature Engineering (pipelines & encoders)

Model Training & Evaluation

Experiment Tracking (MLflow)

API & UI Integration

📈 Model Performance
Model	MAE (USD)	RMSE (USD)
Ridge	~3,520	~4,220
Random Forest	~3,525	~4,225

Low error compared to average salaries (~100k+ USD).

⚡ Run Locally
Backend (FastAPI)
uvicorn src.salary_api:app --reload

UI
python ui/app.py

🛠️ Tech Stack

Python · Pandas · Scikit-learn · MLflow · FastAPI · Joblib · VS Code

👤 Author

Ashish Shirke
