# 🏥 Smart Hospital Flow Predictor

An interactive machine learning prototype that estimates hospital patient waiting times and provides department resource insights through a Streamlit dashboard.

## 📌 Project Overview

Long and unpredictable waiting times can affect patient experience and hospital workflow. Smart Hospital Flow Predictor explores how machine learning can estimate waiting times using patient visit details and hospital department resources.

The application combines a trained machine learning model with an interactive dashboard to support educational exploration of patient flow and resource planning.

## 🎯 Objectives

- Estimate patient waiting times using selected visit and department details.
- Visualize department resources, including staff count, bed count, and daily capacity.
- Explore how arrival time affects model-generated waiting-time estimates.
- Provide department resource comparisons to support further investigation.

## ✨ Key Features

### 1. Hospital Resource Overview
- Summary cards for departments, staff, beds, and daily capacity.
- Interactive department resource comparison charts.
- Department records table.

### 2. Waiting-Time Prediction
- Department selection.
- Visit type and triage-level inputs.
- Age and chronic-condition inputs.
- Arrival date and time.
- Estimated waiting time displayed in minutes.

### 3. Hourly Scenario Analysis
- Compares model-generated estimates across different arrival hours.
- Helps users explore how the model responds to changes in arrival time.

### 4. Resource Insights
- Ranks departments by staff count, bed count, or daily capacity.
- Supports further investigation of department resources.

## 🧠 Machine Learning Approach

The project uses a **Random Forest Regressor implemented in Scikit-learn**. The model was trained for this project using synthetic data. The Random Forest algorithm itself was not implemented from scratch.

### Input Features
- Department ID
- Visit type
- Triage level
- Arrival hour
- Day of week
- Month
- Staff count
- Bed count
- Daily capacity
- Age
- Chronic-condition category

### Output
An estimated patient waiting time in minutes.

## 🛠️ Technology Stack

- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit
- Google Colab for model training

## 📂 Project Structure

```text
Smart-Hospital-Flow-Predictor/
├── app.py
├── departments.csv
├── wait_time_model.joblib
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Run Instructions

### Prerequisites
Install Python and ensure that Python is available from your terminal.

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Smart-Hospital-Flow-Predictor
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with the URL of your GitHub repository. Alternatively, download the repository ZIP and open a terminal in the extracted folder.

### Step 2: Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 3: Run the application

```bash
python -m streamlit run app.py
```

### Step 4: Open the dashboard

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## 📊 Dataset and Transparency

- **Dataset:** Synthetic hospital-encounter data and department resource data.
- **Real patient data:** Not used in the current prototype.
- **Machine learning model:** Random Forest Regressor trained using Scikit-learn.
- **Pretrained models:** None used in the current prototype, if this matches the final code.
- **External APIs:** None used in the current prototype, if this matches the final code.
- **Additional datasets:** Any organizer-provided or other datasets used in the final submission must be declared here.

Update these statements if the final version uses an organizer-provided dataset, external API, or pretrained model.

## ⚠️ Limitations and Responsible Use

This is an educational prototype using synthetic data. Its predictions and resource insights have not been validated for real-world hospital use.

- Synthetic data may not represent actual hospital conditions.
- Predicted waiting times are estimates, not guaranteed outcomes.
- Resource comparisons alone do not establish staffing shortages or overcrowding.
- Hourly charts represent model-generated scenarios, not historical observations or live forecasts.
- The application must not be used for clinical decisions or to determine medical urgency.

Real-world use would require appropriate operational data, careful evaluation, validation, and review by qualified hospital professionals.

## 🚀 Future Improvements

- Evaluation using suitable real-world or organizer-provided data.
- Comparison of multiple machine learning models.
- More comprehensive prediction-error analysis.
- Additional patient-flow and department analytics.
- Further evaluation of resource-planning scenarios.

## 👥 Project Purpose

Developed as a machine learning and data science prototype to explore hospital waiting-time estimation and patient-flow analysis.
