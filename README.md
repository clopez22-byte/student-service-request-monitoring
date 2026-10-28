# Student Service Request Monitoring System (Streamlit)

A Streamlit decision-support prototype for Section B. It reuses the midterm Decision Tree classification approach, target definition, preprocessing, and feature names.

## Modules
- Home
- Dataset upload/schema validation, preview, null and duplicate summary
- Data Analysis with descriptive statistics and interactive charts
- Prediction using the saved `models/trained_model.joblib`
- Dashboard with workload KPIs and a staff review queue
- System Design with functional/non-functional requirements, actors, use-case diagram, and analytical flow

## Run locally
1. Install Python 3.10 or later.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment (recommended).
4. Run `pip install -r requirements.txt`.
5. Run `streamlit run app.py`.
6. Open the local URL shown in the terminal (usually http://localhost:8501).

The saved model is created by `train_model.py` from the bundled CSV. The Streamlit app loads the model and does not retrain on each page refresh.

## Deploy on Streamlit Community Cloud
1. Create a GitHub repository and upload the project files.
2. Confirm that `app.py`, `requirements.txt`, `models/trained_model.joblib`, `models/model_metrics.json`, `modules/`, `assets/`, and `data/` are committed.
3. Go to https://share.streamlit.io and sign in.
4. Select **Create app**, choose the repository and branch, and set the main file path to `app.py`.
5. Deploy, open the generated URL, and test all six navigation pages plus CSV upload and prediction.
6. If deployment fails, inspect the app logs and check package compatibility.

## Model fidelity and limitations
- Target: `SLA_Breached = 1` when `Processing_Days > SLA_Days`, only for eligible completed records.
- Features: `Service_Type`, `Requester_Type`, `Requested_Channel`, `Priority`, `Assigned_Office`, `Queue_Length`, `Required_Documents_Complete`, `SLA_Days`, `Submission_Weekday`, `Queue_Category`.
- Derived fields: weekday from `Submission_Date`; queue categories are Low (0–5), Medium (6–15), High (>15).
- Decision Tree uses `random_state=42`, `class_weight='balanced'`, `OneHotEncoder(handle_unknown='ignore')`, and an 80/20 train/test split with random state 42 and stratification when possible.
- Predictions are advisory only. Do not automatically deny student services or make final decisions based only on the model.
- Use synthetic or properly authorized data for any public deployment. Never publish identifiable student records.
