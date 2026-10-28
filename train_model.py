from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
from modules.data_processing import create_labeled_data, FEATURES

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "student_service_requests.csv")
data = create_labeled_data(df)
if data.empty:
    raise ValueError("No eligible completed records found.")
X = data[FEATURES].copy()
y = data["SLA_Breached"]
if y.nunique() < 2:
    raise ValueError("SLA_Breached contains only one class; binary classification requires both.")

categorical = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
numerical = X.select_dtypes(include=["number"]).columns.tolist()
preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("numerical", "passthrough", numerical)
])
stratify_target = y if y.value_counts().min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=stratify_target
)
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(random_state=42, class_weight="balanced"))
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
metrics = {
    "dataset_rows": int(len(df)),
    "eligible_completed_records": int(len(data)),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "target_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision": float(precision_score(y_test, pred, pos_label=1, zero_division=0)),
    "recall": float(recall_score(y_test, pred, pos_label=1, zero_division=0)),
    "f1": float(f1_score(y_test, pred, pos_label=1, zero_division=0)),
    "confusion_matrix_labels_0_1": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
    "classification_report": classification_report(
        y_test, pred, labels=[0, 1],
        target_names=["No SLA Breach", "SLA Breach"], zero_division=0, output_dict=True
    ),
    "features": FEATURES,
    "target_definition": "SLA_Breached = 1 when Processing_Days > SLA_Days for eligible completed requests"
}
(ROOT / "models" / "trained_model.joblib").parent.mkdir(exist_ok=True)
joblib.dump(model, ROOT / "models" / "trained_model.joblib")
(ROOT / "models" / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))
