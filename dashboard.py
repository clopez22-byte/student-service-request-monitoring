import pandas as pd
import streamlit as st
import plotly.express as px

def render_dashboard(df, model=None):
    st.subheader("Request-delay and workload dashboard")
    total = len(df)
    completed = int(df["Status"].eq("Completed").sum()) if "Status" in df else 0
    pending = int(df["Status"].eq("Pending").sum()) if "Status" in df else 0
    labeled = df[
        df["Status"].eq("Completed")
        & df["Processing_Days"].notna()
        & df["SLA_Days"].notna()
    ].copy()
    breaches = int((labeled["Processing_Days"] > labeled["SLA_Days"]).sum()) if len(labeled) else 0
    rate = (breaches / len(labeled) * 100) if len(labeled) else 0
    avg_queue = pd.to_numeric(df["Queue_Length"], errors="coerce").mean()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total requests", f"{total:,}")
    k2.metric("Pending", f"{pending:,}")
    k3.metric("Completed", f"{completed:,}")
    k4.metric("Observed SLA breach rate", f"{rate:.1f}%" if len(labeled) else "N/A")
    k5.metric("Average queue", f"{avg_queue:.1f}" if pd.notna(avg_queue) else "N/A")

    st.markdown("#### Staff review queue")
    if "Status" in df:
        queue = df[df["Status"].eq("Pending")].copy()
        if not queue.empty:
            queue["Queue_Length"] = pd.to_numeric(queue["Queue_Length"], errors="coerce")
            queue["SLA_Days"] = pd.to_numeric(queue["SLA_Days"], errors="coerce")
            queue["Review reason"] = queue.apply(
                lambda r: "Urgent priority" if str(r.get("Priority", "")).lower() in ("urgent", "high")
                else ("Queue length above 15" if pd.notna(r.get("Queue_Length")) and r["Queue_Length"] > 15
                      else ("Documents incomplete" if str(r.get("Required_Documents_Complete", "")).lower() == "no"
                            else "Routine monitoring")),
                axis=1
            )
            st.dataframe(queue[[c for c in [
                "Request_ID", "Service_Type", "Priority", "Assigned_Office",
                "Queue_Length", "SLA_Days", "Required_Documents_Complete", "Review reason"
            ] if c in queue.columns]], use_container_width=True, hide_index=True)
            st.caption("This is a rules-based review list from observed intake fields, not a claim that these requests are model-predicted breaches.")
        else:
            st.success("No pending requests are present in the selected dataset.")
    st.warning("Human review required: predictions and dashboard indicators are advisory only. Do not automatically reject a request or deny student services.")
