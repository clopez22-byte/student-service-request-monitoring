import pandas as pd
import numpy as np

REQUIRED_COLUMNS = [
    "Request_ID", "Service_Type", "Requester_Type", "Submission_Date",
    "Requested_Channel", "Priority", "Assigned_Office", "Queue_Length",
    "Required_Documents_Complete", "Processing_Days", "SLA_Days",
    "Status", "Reopened"
]
FEATURES = [
    "Service_Type", "Requester_Type", "Requested_Channel", "Priority",
    "Assigned_Office", "Queue_Length", "Required_Documents_Complete",
    "SLA_Days", "Submission_Weekday", "Queue_Category"
]

def validate_dataset(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    if df.empty:
        return False, "The uploaded CSV contains no rows."
    return True, "Required schema found."

def prepare_data(raw):
    df = raw.copy()
    if "Submission_Date" in df:
        df["Submission_Date"] = pd.to_datetime(df["Submission_Date"], errors="coerce")
    for col in ["Processing_Days", "SLA_Days", "Queue_Length"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Submission_Date" in df:
        df["Submission_Weekday"] = df["Submission_Date"].dt.day_name()
    if "Queue_Length" in df:
        df["Queue_Category"] = pd.cut(
            df["Queue_Length"], bins=[-1, 5, 15, float("inf")],
            labels=["Low", "Medium", "High"]
        )
    return df

def create_labeled_data(raw):
    df = prepare_data(raw)
    eligible = (
        df["Status"].eq("Completed")
        & df["Processing_Days"].notna()
        & df["SLA_Days"].notna()
        & df["Submission_Date"].notna()
        & df["Queue_Length"].notna()
    )
    model_data = df.loc[eligible].copy()
    model_data["SLA_Breached"] = (
        model_data["Processing_Days"] > model_data["SLA_Days"]
    ).astype(int)
    return model_data

def data_quality_summary(df):
    return pd.DataFrame({
        "Column": df.columns.astype(str),
        "Data type": [str(x) for x in df.dtypes],
        "Missing values": df.isna().sum().values,
        "Missing %": (df.isna().mean().values * 100).round(2)
    })
