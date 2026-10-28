import pandas as pd
import plotly.express as px
import streamlit as st

def render_eda(df):
    st.subheader("Exploratory data analysis")
    st.caption("Charts are interactive: hover for values, zoom, and use the legend where available.")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.histogram(df, x="Service_Type", title="Request volume by service type",
                         color="Service_Type"),
            use_container_width=True
        )
    with c2:
        if "Status" in df.columns:
            st.plotly_chart(
                px.pie(df, names="Status", title="Request status distribution",
                       hole=0.35),
                use_container_width=True
            )
    labeled = df[
        df["Status"].eq("Completed")
        & df["Processing_Days"].notna()
        & df["SLA_Days"].notna()
    ].copy()
    if not labeled.empty:
        labeled["SLA_Breached"] = labeled["Processing_Days"] > labeled["SLA_Days"]
        labeled["Queue_Category"] = pd.cut(
            pd.to_numeric(labeled["Queue_Length"], errors="coerce"),
            bins=[-1, 5, 15, float("inf")], labels=["Low", "Medium", "High"]
        )
        rates = labeled.groupby("Queue_Category", observed=False)["SLA_Breached"].mean().reset_index()
        rates["Breach rate (%)"] = rates["SLA_Breached"] * 100
        st.plotly_chart(
            px.bar(rates, x="Queue_Category", y="Breach rate (%)",
                   title="Observed SLA breach rate by queue category",
                   text_auto=".1f"),
            use_container_width=True
        )
        office = labeled.groupby("Assigned_Office", dropna=False)["SLA_Breached"].mean().reset_index()
        office["Breach rate (%)"] = office["SLA_Breached"] * 100
        st.plotly_chart(
            px.bar(office, x="Assigned_Office", y="Breach rate (%)",
                   title="Observed SLA breach rate by assigned office",
                   text_auto=".1f"),
            use_container_width=True
        )
    else:
        st.info("No completed requests with known processing and SLA days are available for breach-rate charts.")
