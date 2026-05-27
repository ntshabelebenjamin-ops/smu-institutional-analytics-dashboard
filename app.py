import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SMU Institutional Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("📊 SMU Institutional Analytics & Predictive Intelligence Dashboard")

st.markdown("""
This dashboard demonstrates:
- Institutional analytics
- Executive reporting
- Strategic risk identification
- Data governance awareness
- Predictive analytics capability
- Student success intelligence
""")

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    df = pd.read_csv("SMU_FTEN_Prepared_Predictive_Dataset.csv")
    return df

df = load_data()

# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.header("Dashboard Filters")

selected_quintile = st.sidebar.multiselect(
    "School Quintile",
    options=sorted(df["School_Quintile"].unique()),
    default=sorted(df["School_Quintile"].unique())
)

selected_risk = st.sidebar.multiselect(
    "Risk Category",
    options=df["Risk_Category"].unique(),
    default=df["Risk_Category"].unique()
)

filtered_df = df[
    (df["School_Quintile"].isin(selected_quintile)) &
    (df["Risk_Category"].isin(selected_risk))
]

# =========================================================
# EXECUTIVE KPI SECTION
# =========================================================

st.header("Executive Institutional KPIs")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Students",
        len(filtered_df)
    )

with col2:
    st.metric(
        "Average Success Rate",
        f"{filtered_df['Success_Rate_%'].mean():.1f}%"
    )

with col3:
    at_risk = filtered_df["Dropout_Risk_Flag"].mean() * 100

    st.metric(
        "At-Risk Students",
        f"{at_risk:.1f}%"
    )

with col4:
    st.metric(
        "Average STEM Score",
        f"{filtered_df['STEM_Average'].mean():.1f}"
    )

# =========================================================
# STUDENT SUCCESS DISTRIBUTION
# =========================================================

st.header("Student Success Analytics")

fig_success = px.histogram(
    filtered_df,
    x="Success_Rate_%",
    color="Risk_Category",
    title="Distribution of Student Success Rates",
    nbins=20
)

st.plotly_chart(fig_success, use_container_width=True)

# =========================================================
# QUINTILE ANALYTICS
# =========================================================

fig_quintile = px.box(
    filtered_df,
    x="School_Quintile",
    y="Success_Rate_%",
    color="Risk_Category",
    title="Success Rate by School Quintile"
)

st.plotly_chart(fig_quintile, use_container_width=True)

# =========================================================
# SOCIO-ECONOMIC ANALYTICS
# =========================================================

st.header("Socio-Economic Performance Analytics")

socio_df = (
    filtered_df
    .groupby("Socio_Economic_Background")["Success_Rate_%"]
    .mean()
    .reset_index()
)

fig_socio = px.bar(
    socio_df,
    x="Socio_Economic_Background",
    y="Success_Rate_%",
    title="Average Success Rate by Socio-Economic Background"
)

st.plotly_chart(fig_socio, use_container_width=True)

# =========================================================
# LANGUAGE ANALYTICS
# =========================================================

st.header("Home Language Analytics")

lang_df = (
    filtered_df
    .groupby("Home_Language")["Success_Rate_%"]
    .mean()
    .reset_index()
)

fig_lang = px.bar(
    lang_df,
    x="Home_Language",
    y="Success_Rate_%",
    title="Average Success Rate by Home Language"
)

st.plotly_chart(fig_lang, use_container_width=True)

# =========================================================
# STRATEGIC RISK SECTION
# =========================================================

st.header("Strategic and Funding Risk Indicators")

high_risk_students = filtered_df[
    filtered_df["Risk_Category"] == "High Risk"
]

moderate_risk_students = filtered_df[
    filtered_df["Risk_Category"] == "Moderate Risk"
]

st.warning(f"High-Risk Students Identified: {len(high_risk_students)}")
st.info(f"Moderate-Risk Students Identified: {len(moderate_risk_students)}")

st.markdown("""
### Institutional Strategic Risk Interpretation

Potential risks identified include:
- Weak throughput efficiency
- Student dropout risks
- Potential DHET funding pressure
- Socio-economic inequality effects
- Strategic intervention requirements
- Institutional sustainability concerns
""")

# =========================================================
# DATA GOVERNANCE SECTION
# =========================================================

st.header("Data Governance & Ethical Analytics")

st.success("""
This dashboard demonstrates:
- Evidence-based decision-making
- Institutional transparency
- Ethical AI awareness
- Responsible analytics
- POPIA-aligned governance
- Institutional intelligence maturity
""")

# =========================================================
# PREDICTIVE ANALYTICS
# =========================================================

st.header("Predictive Analytics: Student Dropout Risk")

features = [
    "Maths_%",
    "Physical_Sciences_%",
    "Life_Sciences_%",
    "English_%",
    "School_Quintile",
    "Average_School_Score",
    "STEM_Average",
    "First_Semester_Average_%",
    "Second_Semester_Average_%"
]

target = "Dropout_Risk_Flag"

X = df[features]
y = df[target]

# =========================================================
# SCALE DATA
# =========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# LOGISTIC REGRESSION
# =========================================================

log_model = LogisticRegression()

log_model.fit(X_train, y_train)

log_preds = log_model.predict(X_test)

log_accuracy = accuracy_score(y_test, log_preds)

# =========================================================
# RANDOM FOREST
# =========================================================

rf_model = RandomForestClassifier(
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_preds)

# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.subheader("Predictive Model Performance")

col5, col6 = st.columns(2)

with col5:
    st.metric(
        "Logistic Regression Accuracy",
        f"{log_accuracy:.2%}"
    )

with col6:
    st.metric(
        "Random Forest Accuracy",
        f"{rf_accuracy:.2%}"
    )

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": rf_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

fig_importance = px.bar(
    importance_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Predictive Feature Importance"
)

st.plotly_chart(fig_importance, use_container_width=True)

# =========================================================
# CONFUSION MATRIX
# =========================================================

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, rf_preds)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Low Risk", "Actual At Risk"],
    columns=["Predicted Low Risk", "Predicted At Risk"]
)

st.dataframe(cm_df)

# =========================================================
# EXECUTIVE SUMMARY
# =========================================================

st.header("Executive Institutional Summary")

top_feature = importance_df.iloc[0]["Feature"]

st.markdown(f"""
### Key Institutional Findings

- Total Students Analysed: {len(filtered_df)}
- Average Success Rate: {filtered_df['Success_Rate_%'].mean():.1f}%
- Students At Risk: {at_risk:.1f}%
- Strongest Predictor of Risk: {top_feature}

### Strategic Institutional Implications

The analytics suggest:
- first semester performance strongly predicts risk;
- socio-economic background influences success;
- early warning systems are strategically important;
- predictive analytics supports evidence-based intervention planning.

### Governance Considerations

The dashboard aligns with:
- CHE Data Value Chain principles;
- DHET evidence-based planning;
- institutional intelligence frameworks;
- ethical AI governance practices.
""")

# =========================================================
# RAW DATA SECTION
# =========================================================

st.header("Institutional Dataset")

st.dataframe(filtered_df)

# =========================================================
# DOWNLOAD OPTION
# =========================================================

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Dataset",
    data=csv,
    file_name="filtered_institutional_analytics.csv",
    mime="text/csv"
)
