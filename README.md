# 📊 SMU Institutional Analytics Dashboard

## Overview

This project is a Streamlit-based Institutional Analytics and Predictive Intelligence Dashboard developed for Higher Education institutional planning, executive reporting, and student success analytics.

The dashboard demonstrates:
- Institutional performance analytics
- Executive reporting
- Strategic and funding risk identification
- Predictive analytics capability
- Student success modelling
- Data governance awareness
- Ethical AI and institutional intelligence concepts

---

# Objectives

The application was developed to demonstrate the ability to:

- Analyse institutional performance
- Identify strategic and funding risks
- Apply data governance principles
- Develop predictive analytics capability
- Support evidence-based decision-making
- Build institutional intelligence dashboards

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Data analytics and modelling |
| Streamlit | Dashboard development |
| Pandas | Data manipulation |
| Plotly | Interactive visualisations |
| Scikit-learn | Predictive analytics |
| OpenPyXL | Excel integration |

---

# Dataset

The project uses a simulated SMU FTEN (First-Time Entering Students) dataset containing:

- Mathematics results
- Physical Sciences results
- Life Sciences results
- English results
- School quintile
- Home language
- Socio-economic background
- Semester averages
- Modules enrolled
- Modules passed
- Success rate
- Dropout risk indicators

---

# Dashboard Features

## 1. Executive Institutional KPIs
- Total students
- Average success rate
- At-risk students
- Average STEM score

---

## 2. Institutional Analytics
- Student success distribution
- Quintile-based analytics
- Socio-economic performance analysis
- Language analytics

---

## 3. Strategic Risk Monitoring
The dashboard identifies:
- Student dropout risks
- Throughput concerns
- Potential funding risks
- Socio-economic inequalities
- Institutional sustainability pressures

---

## 4. Data Governance
The application demonstrates:
- Evidence-based decision-making
- Institutional transparency
- Ethical AI awareness
- POPIA-aligned governance
- Responsible analytics

---

## 5. Predictive Analytics

### Predictive Models Used
- Logistic Regression
- Random Forest Classification

### Response Variable
- Dropout_Risk_Flag

### Independent Variables
- Mathematics performance
- Physical Sciences performance
- Life Sciences performance
- English performance
- School quintile
- Semester averages
- STEM average
- Average school score

---

# Project Structure

```text
smu-institutional-analytics-dashboard/
│
├── app.py
├── requirements.txt
├── README.md
├── SMU_FTEN_Prepared_Predictive_Dataset.csv
└── data/
