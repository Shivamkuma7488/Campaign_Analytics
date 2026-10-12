import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Campaign Analytics Dashboard", layout="wide")
st.title("📊 Campaign Analytics Dashboard")
st.write("Explore campaign performance, ROI, conversions, and data quality.")

uploaded = st.sidebar.file_uploader("Upload campaign CSV", type=["csv"])
default_file = "campaigns(1).csv"

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif os.path.exists(default_file):
    df = pd.read_csv(default_file)
else:
    st.warning("Upload your campaign CSV, or put campaigns(1).csv beside app.py.")
    st.stop()

df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
aliases = {"campaign": "campaign_name", "type": "campaign_type", "channel": "campaign_type",
           "spend": "budget", "roi_(%)": "roi", "roi_percent": "roi"}
for old, new in aliases.items():
    if old in df.columns and new not in df.columns:
        df.rename(columns={old: new}, inplace=True)

for col in ["budget", "impressions", "clicks", "conversions", "conversion_rate", "roi"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if {"clicks", "impressions"}.issubset(df.columns):
    df["ctr"] = np.where(df["impressions"] > 0, df["clicks"] / df["impressions"] * 100, np.nan)
if {"conversions", "clicks"}.issubset(df.columns):
    df["click_to_conversion_pct"] = np.where(df["clicks"] > 0, df["conversions"] / df["clicks"] * 100, np.nan)
if {"budget", "conversions"}.issubset(df.columns):
    df["cost_per_conversion"] = np.where(df["conversions"] > 0, df["budget"] / df["conversions"], np.nan)

st.sidebar.header("Filters")
view = df.copy()
if "campaign_type" in view.columns:
    types = sorted(view["campaign_type"].dropna().astype(str).unique())
    selected = st.sidebar.multiselect("Campaign type", types, default=types)
    if selected:
        view = view[view["campaign_type"].astype(str).isin(selected)]
if "campaign_name" in view.columns:
    names = sorted(view["campaign_name"].dropna().astype(str).unique())
    selected_names = st.sidebar.multiselect("Campaign name", names, default=names)
    if selected_names:
        view = view[view["campaign_name"].astype(str).isin(selected_names)]

def total(col):
    return pd.to_numeric(view[col], errors="coerce").sum() if col in view.columns else None
def average(col):
    return pd.to_numeric(view[col], errors="coerce").mean() if col in view.columns else None

a, b, c, d = st.columns(4)
a.metric("Campaign records", f"{len(view):,}")
b.metric("Total budget/spend", f"{total('budget'):,.2f}" if total("budget") is not None else "N/A")
c.metric("Total conversions", f"{total('conversions'):,.0f}" if total("conversions") is not None else "N/A")
d.metric("Average ROI", f"{average('roi'):,.2f}" if average("roi") is not None else "N/A")

st.subheader("Campaign data")
st.dataframe(view, use_container_width=True, hide_index=True)
st.download_button("Download filtered CSV", view.to_csv(index=False).encode("utf-8"),
                   "filtered_campaigns.csv", "text/csv")

if "campaign_type" in view.columns and "roi" in view.columns:
    grouped = view.groupby("campaign_type")["roi"].mean().dropna()
    if not grouped.empty:
        st.subheader("Average ROI by campaign type")
        fig, ax = plt.subplots()
        grouped.plot(kind="bar", ax=ax)
        ax.set_ylabel("Average ROI")
        plt.xticks(rotation=35, ha="right")
        st.pyplot(fig)
        plt.close(fig)

if "campaign_type" in view.columns and "conversions" in view.columns:
    grouped = view.groupby("campaign_type")["conversions"].sum().dropna()
    if not grouped.empty:
        st.subheader("Total conversions by campaign type")
        fig, ax = plt.subplots()
        grouped.plot(kind="bar", ax=ax)
        ax.set_ylabel("Conversions")
        plt.xticks(rotation=35, ha="right")
        st.pyplot(fig)
        plt.close(fig)

if "roi" in view.columns:
    values = pd.to_numeric(view["roi"], errors="coerce").dropna()
    if not values.empty:
        st.subheader("ROI distribution")
        fig, ax = plt.subplots()
        ax.hist(values, bins=min(20, max(5, int(np.sqrt(len(values))))))
        ax.set_xlabel("ROI")
        ax.set_ylabel("Records")
        st.pyplot(fig)
        plt.close(fig)
        cutoff = st.slider("Flag records with ROI at or below", float(values.min()),
                           float(values.max()), float(values.quantile(0.25)))
        flagged = view[pd.to_numeric(view["roi"], errors="coerce") <= cutoff]
        st.write(f"{len(flagged)} record(s) flagged for review")
        st.dataframe(flagged, use_container_width=True, hide_index=True)

numeric = view.select_dtypes(include=np.number)
if len(numeric.columns) >= 2:
    st.subheader("Correlation matrix")
    st.dataframe(numeric.corr().round(3), use_container_width=True)

with st.expander("Optional: predict ROI using campaign_model.pkl"):
    model_path = "campaign_model.pkl"
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            features = getattr(model, "feature_names_in_", None)
            if features is None:
                st.info("The model does not expose its expected feature names.")
            elif all(col in view.columns for col in features):
                X = view[list(features)].apply(pd.to_numeric, errors="coerce")
                valid = X.notna().all(axis=1)
                if valid.any():
                    out = view.loc[valid].copy()
                    out["predicted_roi"] = model.predict(X.loc[valid])
                    st.dataframe(out, use_container_width=True, hide_index=True)
                    st.download_button("Download predictions", out.to_csv(index=False).encode("utf-8"),
                                       "campaign_predictions.csv", "text/csv")
                else:
                    st.info("No rows contain complete numeric model inputs.")
            else:
                st.warning("Some required model feature columns are missing from the CSV.")
        except Exception as exc:
            st.error(f"Could not load model: {exc}")
    else:
        st.info("Put campaign_model.pkl in the same folder to enable predictions.")

st.caption("Check column definitions and units before interpreting ROI or cost metrics.")
