import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Electricity Theft Detection",
    page_icon="⚡",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>
.metric-card {
    background-color: #f5f7fa;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# TITLE
# ==========================================================

st.title("⚡ Electricity Theft Detection Dashboard")
st.markdown(
    "Detect suspicious electricity consumption patterns using Machine Learning."
)

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("⚙ Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Consumer Search",
        "About Project"
    ]
)

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

def clean_data(df):

    if "CHK_STATE" in df.columns:
        df = df.drop(columns=["CHK_STATE"])

    df = df.replace("None", pd.NA)

    for col in df.columns:
        if col != "CONS_NO":
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    df = df.fillna(
        df.mean(numeric_only=True)
    )

    return df


def feature_engineering(df):

    if "CONS_NO" in df.columns:
        df = df.drop(columns=["CONS_NO"])

    date_cols = df.columns

    features = pd.DataFrame()

    features["mean_usage"] = df[date_cols].mean(axis=1)
    features["max_usage"] = df[date_cols].max(axis=1)
    features["min_usage"] = df[date_cols].min(axis=1)
    features["std_usage"] = df[date_cols].std(axis=1)

    features["total_usage"] = df[date_cols].sum(axis=1)

    features["zero_usage_days"] = (
        df[date_cols] == 0
    ).sum(axis=1)

    features["consumption_variance"] = (
        df[date_cols].var(axis=1)
    )

    features["drop_ratio"] = (
        features["min_usage"] /
        (features["max_usage"] + 1)
    )

    features["peak_usage_ratio"] = (
        features["max_usage"] /
        (features["mean_usage"] + 1)
    )

    return features


# ==========================================================
# RISK LEVEL
# ==========================================================

def get_risk(score):

    if score < 30:
        return "Low"

    elif score < 70:
        return "Medium"

    return "High"


# ==========================================================
# FILE UPLOAD
# ==========================================================

uploaded_file = st.sidebar.file_uploader(
    "Upload Electricity CSV",
    type=["csv"]
)

# ==========================================================
# LOAD DATA
# ==========================================================

results = None

if uploaded_file is not None:

    raw_df = pd.read_csv(uploaded_file)

    consumer_ids = None

    if "CONS_NO" in raw_df.columns:
        consumer_ids = raw_df["CONS_NO"]

    cleaned_df = clean_data(raw_df.copy())

    features = feature_engineering(cleaned_df)

    probs = model.predict_proba(features)[:, 1]

    results = pd.DataFrame()

    if consumer_ids is not None:
        results["CONS_NO"] = consumer_ids

    results["Risk Score"] = (
        probs * 100
    ).round(2)

    results["Risk Level"] = (
        results["Risk Score"]
        .apply(get_risk)
    )

# ==========================================================
# DASHBOARD
# ==========================================================

if page == "Dashboard":

    st.header("📊 Dashboard")

    if results is None:

        st.info(
            "Upload an electricity dataset from the sidebar."
        )

    else:

        total_consumers = len(results)

        high_risk = len(
            results[results["Risk Level"] == "High"]
        )

        medium_risk = len(
            results[results["Risk Level"] == "Medium"]
        )

        low_risk = len(
            results[results["Risk Level"] == "Low"]
        )

        # ==================================================
        # KPI CARDS
        # ==================================================

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                f"""
                <div style="
                    background-color: #0449B9;
                    padding: 25px 20px;
                    border-radius: 16px;
                    text-align: center;
                    color: white;
                    box-shadow: 0 4px 12px rgba(31, 119, 180, 0.3);">
                    <p style="margin-bottom: 10px;">Total Consumers</p>
                    <h2 style="font-size: 48px; margin: 0;">{total_consumers}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div style="
                    background-color: #B91304;
                    padding: 25px 20px;
                    border-radius: 16px;
                    text-align: center;
                    color: white;
                    box-shadow: 0 4px 12px rgba(231, 76, 60, 0.3);">
                    <p style="margin-bottom: 10px;">High Risk</p>
                    <h3 style="font-size: 48px; margin: 0;">{high_risk}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col3:
            st.markdown(
                f"""
                <div style="
                    background-color: #FBA018;
                    padding: 25px 20px;
                    border-radius: 16px;
                    text-align: center;
                    color: white;
                    box-shadow: 0 4px 12px rgba(243, 156, 18, 0.3);">
                    <p style="margin-bottom: 10px;">Medium Risk</p>
                    <h3 style="font-size: 48px; margin: 0;">{medium_risk}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col4:
            st.markdown(
                f"""
                <div style="
                    background-color: #038125;
                    padding: 25px 20px;
                    border-radius: 16px;
                    text-align: center;
                    color: white;
                    box-shadow: 0 4px 12px rgba(39, 174, 96, 0.3);">
                    <p style="margin-bottom: 10px;">Low Risk</p>
                    <h3 style="font-size: 48px; margin: 0;">{low_risk}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.divider()

        # ==================================================
        # CHARTS
        # ==================================================

        chart1, chart2 = st.columns(2)

        with chart1:

            st.subheader("Risk Distribution")

            fig = px.pie(
                results,
                names="Risk Level",
                title="Consumer Risk Categories"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with chart2:

            st.subheader("Risk Score Distribution")

            fig = px.histogram(
                results,
                x="Risk Score",
                nbins=20,
                title="Risk Score Histogram"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.divider()

        # ==================================================
        # HIGH RISK CONSUMERS
        # ==================================================

        st.subheader("🚨 High Risk Consumers")

        high_risk_df = results[
            results["Risk Level"] == "High"
        ]

        st.dataframe(
            high_risk_df,
            use_container_width=True
        )

        st.divider()

        # ==================================================
        # ALL RESULTS
        # ==================================================

        st.subheader("📋 All Prediction Results")

        st.dataframe(
            results,
            use_container_width=True
        )

        csv = results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇ Download Results",
            csv,
            file_name="predictions.csv",
            mime="text/csv"
        )