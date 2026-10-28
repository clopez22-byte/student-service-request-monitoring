# Deployment Report — Student Service Request Monitoring System

## Project
Section B — Student Service Request Monitoring System (Systems Analysis and Design)

## Reused midterm work
- Reuses the saved Decision Tree pipeline at `models/trained_model.joblib`.
- Uses the bundled synthetic dataset at `data/student_service_requests.csv`.
- Target: `SLA_Breached = 1` when `Processing_Days > SLA_Days` for eligible completed requests.
- Uses the saved model's original features: `Service_Type`, `Requester_Type`, `Requested_Channel`, `Priority`, `Assigned_Office`, `Queue_Length`, `Required_Documents_Complete`, `SLA_Days`, `Submission_Weekday`, and `Queue_Category`.
- Model metrics are stored in `models/model_metrics.json`; the app loads the saved pipeline rather than training it on each page refresh.

## Implemented modules
1. Home
2. Dataset upload, schema validation, preview, missing-value and duplicate summaries
3. Data Analysis with descriptive statistics and interactive charts
4. Prediction using the saved model and intake-time fields
5. Dashboard with workload KPIs and a staff review queue
6. System Design with functional/non-functional requirements, actors, use-case diagram, and process flow

## Local run
1. Extract this project folder.
2. Open a terminal in the `student_service_streamlit` folder (the folder containing `app.py`).
3. Install a supported Python version such as Python 3.13.
4. Run `py -3.13 -m pip install --timeout 120 --retries 10 -r requirements.txt` on Windows.
5. Run `py -3.13 -m streamlit run app.py`.
6. Open the local URL printed in the terminal, usually `http://localhost:8501`.

## Streamlit Community Cloud deployment
1. Create a GitHub repository and upload the *contents* of the `student_service_streamlit` folder.
2. Ensure `app.py`, `requirements.txt`, `README.md`, `modules/`, `models/`, `assets/`, and `data/` are at the repository root.
3. Open https://share.streamlit.io and sign in with the account connected to GitHub.
4. Choose **Create app**, select the repository and branch, and set **Main file path** to `app.py`.
5. Click **Deploy**. Wait for the build to finish, open the generated app URL, and test each module.
6. If the build fails, open the app's **Manage app / Logs** panel and check the first error.

## Verification checklist
- [ ] Home page loads.
- [ ] Dataset page previews data and reports missing values / duplicates.
- [ ] A valid CSV is accepted.
- [ ] A CSV with missing required columns shows a clear error.
- [ ] Data Analysis charts render.
- [ ] A prediction can be generated from intake-time fields.
- [ ] Dashboard KPIs and staff review queue render.
- [ ] System Design displays requirements, actors, use-case diagram, and process flow.
- [ ] Deployed URL opens without an error.
- [ ] Capture screenshots of the five required working modules and five sample predictions.

## Limitations and responsible use
- This is a decision-support prototype. Predictions and queue indicators are advisory and must not automatically deny student services.
- Model performance is based on the bundled synthetic dataset and a held-out test split; it is not proof of performance on real university data.
- The model was trained using completed historical requests, because the target requires known processing duration. Apply extra caution when interpreting risk for new/pending requests.
- Public deployments should use synthetic or properly authorized data only. Do not commit passwords or identifiable student records.

## Deployment status
The project files are prepared for deployment. A public GitHub repository and Streamlit URL must be created from the user's own GitHub/Streamlit account; those external publishing steps are not completed by this local project package.
