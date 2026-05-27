import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
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
- Predictive analytics
- Classification analytics
- K-Means clustering
- Student success intelligence
- Strategic risk identification
- Data governance awareness
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
        f"{df['Success_Rate_%'].mean():.1f}%"
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
# EXECUTIVE KPIs
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
        "Average School Score",
        f"{df['Average_School_Score'].mean():.1f}"
    )

# =========================================================
# SUCCESS DISTRIBUTION
# =========================================================

st.header("Institutional Performance Analytics")

fig_success = px.histogram(
    df,
    x="Success_Rate_%",
    color="Risk_Category",
    nbins=20,
    title="Student Success Rate Distribution"
)

st.plotly_chart(fig_success, use_container_width=True)

# =========================================================
# SOCIO-ECONOMIC ANALYTICS
# =========================================================

st.header("Socio-Economic Analytics")

socio_df = (
    df.groupby("Socio_Economic_Background")["Success_Rate_%"]
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
# K-MEANS CLUSTERING
# =========================================================

st.header("K-Means Clustering: Student Risk Groups")

features = [
    "Maths_%",
    "Physical_Sciences_%",
    "Life_Sciences_%",
    "English_%",
    "Average_School_Score",
    "First_Semester_Average_%",
    "Second_Semester_Average_%",
    "STEM_Average",
    "Modules_Passed",
    "Success_Rate_%"
]

X = df[features]

# =========================================================
# STANDARDISE DATA
# =========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =========================================================
# KMEANS MODEL
# =========================================================

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
            "Success_Rate_%",
            "Average_School_Score",
            "First_Semester_Average_%",
            "Second_Semester_Average_%"
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
    cluster_summary["Success_Rate_%"].idxmin(): "High Risk Cluster",
    cluster_summary["Success_Rate_%"].idxmax(): "Low Risk Cluster"
}

remaining_cluster = list(
    set(cluster_summary.index)
    - set(cluster_risk_map.keys())
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
    title="Student Distribution Across Clusters"
)

st.plotly_chart(fig_cluster_count, use_container_width=True)

# =========================================================
# PREDICTIVE ANALYTICS
# =========================================================

st.header("Predictive Analytics")

predictors = [
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

st.subheader("Predictive Model Performance")

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

st.subheader("Dropout Risk Confusion Matrix")

cm = confusion_matrix(y_test, rf_preds)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Low Risk", "Actual At Risk"],
    columns=["Predicted Low Risk", "Predicted At Risk"]
)

st.dataframe(cm_df)

# =========================================================
# CLASSIFICATION ANALYTICS
# =========================================================

st.header("Classification Analytics: Risk Category Prediction")

st.markdown("""
This section predicts:
- High Risk students
- Moderate Risk students
- Low Risk students

using supervised machine learning classification methods.
""")

classification_features = [
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

X_class = df[classification_features]

y_class = df["Risk_Category"]

# =========================================================
# SCALE DATA
# =========================================================

X_class_scaled = scaler.fit_transform(X_class)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class_scaled,
    y_class,
    test_size=0.2,
    random_state=42
)

# =========================================================
# RANDOM FOREST CLASSIFIER
# =========================================================

rf_classifier = RandomForestClassifier(
    random_state=42
)

rf_classifier.fit(
    X_train_class,
    y_train_class
)

y_pred_class = rf_classifier.predict(X_test_class)

# =========================================================
# CLASSIFICATION ACCURACY
# =========================================================

classification_accuracy = accuracy_score(
    y_test_class,
    y_pred_class
)

st.subheader("Classification Model Accuracy")

st.metric(
    "Risk Category Classification Accuracy",
    f"{classification_accuracy:.2%}"
)

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

st.subheader("Classification Report")

class_report = classification_report(
    y_test_class,
    y_pred_class,
    output_dict=True
)

class_report_df = pd.DataFrame(class_report).transpose()

st.dataframe(class_report_df)

# =========================================================
# CLASSIFICATION CONFUSION MATRIX
# =========================================================

st.subheader("Risk Category Confusion Matrix")

cm_class = confusion_matrix(
    y_test_class,
    y_pred_class
)

cm_class_df = pd.DataFrame(
    cm_class,
    index=rf_classifier.classes_,
    columns=rf_classifier.classes_
)

st.dataframe(cm_class_df)

# =========================================================
# CLASSIFICATION FEATURE IMPORTANCE
# =========================================================

st.subheader("Risk Category Predictor Importance")

importance_class_df = pd.DataFrame({
    "Feature": classification_features,
    "Importance": rf_classifier.feature_importances_
})

importance_class_df = importance_class_df.sort_values(
    by="Importance",
    ascending=False
)

fig_class_importance = px.bar(
    importance_class_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Feature Importance for Risk Category Classification"
)

st.plotly_chart(fig_class_importance, use_container_width=True)

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
- Average Success Rate: {df['Success_Rate_%'].mean():.1f}%
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
- predictive analytics supports intervention planning;
- institutional intelligence improves decision-making;
- clustering supports targeted student support.
""")

# =========================================================
# DATA TABLE
# =========================================================

st.header("Institutional Dataset")

st.dataframe(df)

# =========================================================
# DOWNLOAD DATA
# =========================================================

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Institutional Dataset",
    data=csv,
    file_name="institutional_analytics_dataset.csv",
    mime="text/csv"
)
