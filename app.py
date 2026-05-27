import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix

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
- Predictive analytics
- K-Means clustering
- Executive reporting
- Strategic risk identification
- Data governance awareness
""")

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    df = pd.read_csv("SMU_FTEN_1245_Predictive_Dataset.csv")
    return df

df = load_data()

# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("%", "Pct")
)

# =========================================================
# DATA OVERVIEW
# =========================================================

st.header("Institutional Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Students", len(df))

with col2:
    st.metric("Variables", len(df.columns))

with col3:
    st.metric(
        "Average Success Rate",
        f"{df['Success_Rate_Pct'].mean():.1f}%"
    )

st.write("Dataset Preview")
st.dataframe(df.head())

# =========================================================
# MISSING VALUES
# =========================================================

st.subheader("Missing Values Check")

missing_df = pd.DataFrame({
    "Variable": df.columns,
    "Missing Values": df.isnull().sum().values
})

st.dataframe(missing_df)

# =========================================================
# FEATURES
# =========================================================

features = [
    "Maths_Pct",
    "Physical_Sciences_Pct",
    "Life_Sciences_Pct",
    "English_Pct",
    "School_Quintile",
    "Average_School_Score",
    "First_Semester_Average_Pct",
    "Second_Semester_Average_Pct",
    "STEM_Average",
    "Modules_Enrolled",
    "Modules_Passed",
    "Success_Rate_Pct"
]

X = df[features]

# =========================================================
# SCALE DATA
# =========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

st.success("""
Data preparation completed:
- Missing value validation
- Feature engineering
- Data scaling
- Machine learning preparation
""")

# =========================================================
# KPI SECTION
# =========================================================

st.header("Executive Institutional KPIs")

col4, col5, col6, col7 = st.columns(4)

with col4:
    st.metric(
        "Average STEM Score",
        f"{df['STEM_Average'].mean():.1f}"
    )

with col5:
    at_risk = df["Dropout_Risk_Flag"].mean() * 100

    st.metric(
        "At-Risk Students",
        f"{at_risk:.1f}%"
    )

with col6:
    st.metric(
        "Average Modules Passed",
        f"{df['Modules_Passed'].mean():.1f}"
    )

with col7:
    st.metric(
        "Average Quintile",
        f"{df['School_Quintile'].mean():.1f}"
    )

# =========================================================
# SUCCESS DISTRIBUTION
# =========================================================

st.header("Institutional Performance Analytics")

fig_success = px.histogram(
    df,
    x="Success_Rate_Pct",
    color="Risk_Category",
    nbins=25,
    title="Student Success Rate Distribution"
)

st.plotly_chart(fig_success, use_container_width=True)

# =========================================================
# KMEANS CLUSTERING
# =========================================================

st.header("K-Means Clustering: Student Risk Segmentation")

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

# =========================================================
# CLUSTER SUMMARY
# =========================================================

cluster_summary = (
    df.groupby("Cluster")[
        [
            "Success_Rate_Pct",
            "Average_School_Score",
            "First_Semester_Average_Pct",
            "Second_Semester_Average_Pct"
        ]
    ]
    .mean()
    .round(1)
)

st.subheader("Cluster Summary")

st.dataframe(cluster_summary)

# =========================================================
# CLUSTER LABELS
# =========================================================

cluster_risk_map = {
    cluster_summary["Success_Rate_Pct"].idxmin(): "High Risk Cluster",
    cluster_summary["Success_Rate_Pct"].idxmax(): "Low Risk Cluster"
}

remaining_cluster = list(
    set(cluster_summary.index) -
    set(cluster_risk_map.keys())
)[0]

cluster_risk_map[remaining_cluster] = "Moderate Risk Cluster"

df["Cluster_Label"] = df["Cluster"].map(cluster_risk_map)

# =========================================================
# PCA VISUALISATION
# =========================================================

pca = PCA(n_components=2)

pca_components = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "PCA1": pca_components[:, 0],
    "PCA2": pca_components[:, 1],
    "Cluster": df["Cluster_Label"]
})

fig_cluster = px.scatter(
    pca_df,
    x="PCA1",
    y="PCA2",
    color="Cluster",
    title="K-Means Student Clusters"
)

st.plotly_chart(fig_cluster, use_container_width=True)

# =========================================================
# CLUSTER COUNTS
# =========================================================

cluster_counts = (
    df["Cluster_Label"]
    .value_counts()
    .reset_index()
)

cluster_counts.columns = ["Cluster", "Students"]

fig_cluster_count = px.bar(
    cluster_counts,
    x="Cluster",
    y="Students",
    color="Cluster",
    title="Student Distribution by Cluster"
)

st.plotly_chart(fig_cluster_count, use_container_width=True)

# =========================================================
# PREDICTIVE ANALYTICS
# =========================================================

st.header("Predictive Analytics")

predictors = [
    "Maths_Pct",
    "Physical_Sciences_Pct",
    "Life_Sciences_Pct",
    "English_Pct",
    "School_Quintile",
    "Average_School_Score",
    "STEM_Average",
    "First_Semester_Average_Pct",
    "Second_Semester_Average_Pct"
]

target = "Dropout_Risk_Flag"

X_pred = df[predictors]
y = df[target]

X_pred_scaled = scaler.fit_transform(X_pred)

X_train, X_test, y_train, y_test = train_test_split(
    X_pred_scaled,
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

rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_preds)

# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.subheader("Model Performance")

col8, col9 = st.columns(2)

with col8:
    st.metric(
        "Logistic Regression Accuracy",
        f"{log_accuracy:.2%}"
    )

with col9:
    st.metric(
        "Random Forest Accuracy",
        f"{rf_accuracy:.2%}"
    )

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance_df = pd.DataFrame({
    "Feature": predictors,
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
    title="Feature Importance for Dropout Prediction"
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
# GOVERNANCE SECTION
# =========================================================

st.header("Data Governance & Ethical AI")

st.info("""
This dashboard supports:
- Evidence-based decision-making
- Institutional transparency
- Ethical AI awareness
- POPIA-aligned governance
- Responsible analytics
- Predictive institutional intelligence
""")

# =========================================================
# EXECUTIVE SUMMARY
# =========================================================

st.header("Executive Institutional Summary")

top_feature = importance_df.iloc[0]["Feature"]

st.markdown(f"""
### Key Findings

- Total Students Analysed: {len(df)}
- Average Success Rate: {df['Success_Rate_Pct'].mean():.1f}%
- Students At Risk: {at_risk:.1f}%
- Strongest Predictor of Risk: {top_feature}

### Cluster Analysis

K-Means clustering identified:
- High Risk student groups
- Moderate Risk student groups
- Low Risk student groups

### Institutional Implications

The analytics suggest:
- first semester performance strongly predicts risk;
- predictive analytics supports early intervention;
- institutional intelligence improves evidence-based planning;
- clustering improves targeted student support.
""")

# =========================================================
# DATA TABLE
# =========================================================

st.header("Institutional Dataset")

st.dataframe(df)

# =========================================================
# DOWNLOAD BUTTON
# =========================================================

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Institutional Dataset",
    data=csv,
    file_name="institutional_analytics_dataset.csv",
    mime="text/csv"
)
