from pathlib import Path
import sys
import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from modules.data_processing import (
    REQUIRED_COLUMNS, FEATURES, validate_dataset, prepare_data,
    data_quality_summary, create_labeled_data
)
from modules.exploratory_analysis import render_eda
from modules.prediction import load_model, load_metrics, predict_one
from modules.dashboard import render_dashboard

st.set_page_config(
    page_title="Student Service Request Monitoring System",
    page_icon="📊",
    layout="wide"
)

DEFAULT_DATA = ROOT / "data" / "student_service_requests.csv"

@st.cache_data
def read_default_data(path):
    return pd.read_csv(path)

def get_data():
    st.sidebar.markdown("### Dataset source")
    uploaded = st.sidebar.file_uploader("Upload a student-service CSV", type=["csv"])
    if uploaded is not None:
        try:
            return pd.read_csv(uploaded), "Uploaded CSV"
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")
            return None, "Invalid upload"
    try:
        return read_default_data(str(DEFAULT_DATA)), "Bundled sample dataset"
    except Exception as exc:
        st.error(f"Default dataset could not be loaded: {exc}")
        return None, "Unavailable"

st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose module",
    ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"]
)
st.sidebar.caption("Section B · Systems Analysis and Design")
df_raw, data_source = get_data()

if page == "Home":
    st.title("Student Service Request Monitoring System")
    st.write("A decision-support prototype for monitoring student-service workload and identifying possible SLA-breach risk.")
    st.info("The application reuses the midterm Decision Tree pipeline and its original intake-time feature set. It does not make final service decisions.")
    st.markdown("### Modules")
    cols = st.columns(3)
    for col, title, desc in zip(cols,
        ["Dataset", "Data Analysis", "Prediction"],
        ["Validate uploaded CSV files and inspect data quality.",
         "Explore request volume, status, queues, and observed breach rates.",
         "Estimate SLA-breach risk for one request using the saved model."]):
        with col:
            st.markdown(f"**{title}**")
            st.write(desc)
    cols = st.columns(3)
    for col, title, desc in zip(cols,
        ["Dashboard", "System Design", "Human review"],
        ["View workload KPIs and a staff review queue.",
         "Review requirements, actors, use cases, and process flow.",
         "Authorized staff verify records and decide follow-up actions."]):
        with col:
            st.markdown(f"**{title}**")
            st.write(desc)
    st.markdown("### Dataset status")
    if df_raw is not None:
        st.write(f"Current source: **{data_source}** · **{len(df_raw):,} rows**")
        st.write("Columns:", ", ".join(map(str, df_raw.columns)))
    st.caption("For public deployment, use synthetic or properly authorized records. Do not upload identifiable student data.")

elif page == "Dataset":
    st.title("Dataset Upload and Validation")
    if df_raw is not None:
        st.caption(f"Source: {data_source}")
        valid, message = validate_dataset(df_raw)
        if valid:
            st.success(message)
        else:
            st.error(message)
        st.metric("Rows", len(df_raw))
        st.metric("Columns", len(df_raw.columns))
        st.markdown("#### Data preview")
        st.dataframe(df_raw.head(50), use_container_width=True, hide_index=True)
        st.markdown("#### Data quality summary")
        st.dataframe(data_quality_summary(df_raw), use_container_width=True, hide_index=True)
        duplicate_count = int(df_raw["Request_ID"].duplicated().sum()) if "Request_ID" in df_raw else None
        c1, c2, c3 = st.columns(3)
        c1.metric("Duplicate Request_ID", duplicate_count if duplicate_count is not None else "Column missing")
        c2.metric("Rows with any missing value", int(df_raw.isna().any(axis=1).sum()))
        c3.metric("Total missing cells", int(df_raw.isna().sum().sum()))
        if not valid:
            st.warning("This file cannot be used with the midterm model until the required columns are present.")
    else:
        st.warning("Load a valid CSV using the uploader in the sidebar.")

elif page == "Data Analysis":
    st.title("Exploratory Data Analysis")
    if df_raw is not None:
        valid, msg = validate_dataset(df_raw)
        if valid:
            prepared = prepare_data(df_raw)
            st.markdown("#### Descriptive statistics")
            st.dataframe(prepared.describe(include="all").transpose(), use_container_width=True)
            st.markdown("#### Interactive charts")
            render_eda(prepared)
            st.markdown("#### Target distribution on eligible completed requests")
            labeled = create_labeled_data(df_raw)
            if not labeled.empty:
                counts = labeled["SLA_Breached"].map({0: "No SLA breach", 1: "SLA breach"}).value_counts().rename_axis("Outcome").reset_index(name="Count")
                st.plotly_chart(px.bar(counts, x="Outcome", y="Count", color="Outcome", title="SLA breach class distribution", text="Count"), use_container_width=True)
                st.write(f"Eligible labeled completed requests: **{len(labeled)}**")
            else:
                st.info("No eligible completed requests with all required label fields were found.")
        else:
            st.error(msg)
    else:
        st.warning("Load a CSV first.")

elif page == "Prediction":
    st.title("Predictive Modeling")
    st.write("Enter details available at submission/intake time. Actual processing days and final status are not prediction inputs.")
    try:
        model = load_model()
        metrics = load_metrics()
    except Exception as exc:
        st.error(f"Saved model could not be loaded: {exc}")
        st.stop()

    if metrics:
        st.markdown("#### Midterm model evaluation")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
        m2.metric("Precision", f"{metrics.get('precision', 0):.1%}")
        m3.metric("Recall", f"{metrics.get('recall', 0):.1%}")
        m4.metric("F1-score", f"{metrics.get('f1', 0):.1%}")
        st.caption("Metrics are calculated from the held-out test split during project setup, not retrained on each page refresh.")
    st.divider()
    st.markdown("#### New request inputs")
    options = {
        "Service_Type": ["Certification", "ID Replacement", "IT Support", "Payment Concern", "Records Request", "Clearance", "Other"],
        "Requester_Type": ["Student", "Staff", "Faculty"],
        "Requested_Channel": ["Portal", "Walk-in", "Email", "Phone", "Other"],
        "Priority": ["Normal", "High", "Urgent", "Low"],
        "Assigned_Office": ["Registrar", "Student Affairs", "IT Office", "Accounting Office", "Guidance Office", "Other"],
        "Required_Documents_Complete": ["Yes", "No"]
    }
    c1, c2 = st.columns(2)
    with c1:
        service = st.selectbox("Service type", options["Service_Type"])
        requester = st.selectbox("Requester type", options["Requester_Type"])
        channel = st.selectbox("Requested channel", options["Requested_Channel"])
        priority = st.selectbox("Priority", options["Priority"])
        office = st.selectbox("Assigned office", options["Assigned_Office"])
    with c2:
        queue = st.number_input("Queue length at submission", min_value=0, max_value=10000, value=8, step=1)
        docs = st.selectbox("Required documents complete?", options["Required_Documents_Complete"])
        sla = st.number_input("SLA days", min_value=1, max_value=365, value=5, step=1)
        weekday = st.selectbox("Submission weekday", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        queue_category = "Low" if queue <= 5 else ("Medium" if queue <= 15 else "High")
        st.write(f"Derived queue category: **{queue_category}**")
    if st.button("Predict SLA-breach risk", type="primary"):
        values = {
            "Service_Type": service, "Requester_Type": requester,
            "Requested_Channel": channel, "Priority": priority,
            "Assigned_Office": office, "Queue_Length": int(queue),
            "Required_Documents_Complete": docs, "SLA_Days": int(sla),
            "Submission_Weekday": weekday, "Queue_Category": queue_category
        }
        try:
            prediction, probability = predict_one(model, values)
            if prediction == 1:
                st.error("Model result: Possible SLA breach")
            else:
                st.success("Model result: No SLA breach predicted")
            if probability is not None:
                st.metric("Estimated breach probability", f"{probability:.1%}")
            st.markdown("**Recommended human follow-up**")
            st.write("Check the request details, document completeness, current queue, applicable SLA deadline, and office workload. Confirm the facts with the requester or responsible office when needed.")
            st.caption("The model estimates a pattern learned from historical labeled records. It does not guarantee the outcome and should not be the sole basis for decisions.")
            st.write("Prediction input summary")
            st.dataframe(pd.DataFrame([values]), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

elif page == "Dashboard":
    st.title("Decision-Support Dashboard")
    if df_raw is not None:
        valid, msg = validate_dataset(df_raw)
        if valid:
            render_dashboard(prepare_data(df_raw))
        else:
            st.error(msg)
    else:
        st.warning("Load a CSV first.")

elif page == "System Design":
    st.title("Systems Analysis and Design Integration")
    st.markdown("### Functional requirements")
    requirements = [
        "The system shall validate uploaded CSV files against the midterm dataset schema and report missing columns, nulls, and duplicate request identifiers.",
        "The system shall display service-request volume by service type and status using interactive charts.",
        "The system shall calculate and display observed SLA-breach rates for completed requests with known processing and SLA days.",
        "The system shall use the saved midterm Decision Tree pipeline to predict SLA-breach risk from the original model feature set.",
        "The system shall display a staff review queue and recommended follow-up while leaving final decisions to authorized staff."
    ]
    for i, item in enumerate(requirements, 1):
        st.write(f"{i}. {item}")
    st.markdown("### Non-functional requirements")
    for item in [
        "Usability: provide clear navigation, labels, error messages, and readable charts.",
        "Reliability: validate schema and input values; load the saved model without retraining on page refresh.",
        "Privacy and security: use synthetic or authorized data and do not commit sensitive student records or credentials."
    ]:
        st.write(f"- {item}")
    st.markdown("### Actors and use cases")
    st.write("**Student Services Administrator:** View Service Volume, View SLA Performance, Review Delay Risk, Monitor Queue Status.")
    st.write("**Service Staff:** Monitor Queue Status, Manage Student Requests, review the model's advisory flag, and record follow-up outside this prototype.")
    img_path = ROOT / "assets" / "use_case_diagram.png"
    if img_path.exists():
        st.image(str(img_path), caption="Provided Student Service Request Monitoring System use-case diagram", use_container_width=True)
    st.markdown("### Analytical process flow")
    st.code("CSV / authorized records → schema and quality checks → feature engineering (weekday, queue category) → saved Decision Tree pipeline → predicted risk + probability (if supported) → staff review → authorized human decision and follow-up", language="text")
    st.markdown("### Model scope and limitations")
    st.write("The target is SLA_Breached, derived for completed records as Processing_Days > SLA_Days. The model uses the midterm features: Service_Type, Requester_Type, Requested_Channel, Priority, Assigned_Office, Queue_Length, Required_Documents_Complete, SLA_Days, Submission_Weekday, and Queue_Category.")
    st.warning("The midterm target is created using completed requests only. Therefore, model training labels come from historical completed requests; prediction-time use is advisory and should be monitored for data drift and fairness.")
