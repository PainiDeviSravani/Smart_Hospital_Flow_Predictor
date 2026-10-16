
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="Smart Hospital Flow Predictor",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Smart Hospital Flow Predictor")
st.write(
    "An ML-powered educational prototype for estimating patient "
    "waiting times and exploring hospital resources."
)

st.info(
    "Demo notice: This prototype uses synthetic data. "
    "Predictions and planning insights are experimental and must not "
    "be used for clinical decisions."
)

# ---------------- LOAD PROJECT FILES ----------------
folder = Path(__file__).parent
model_path = folder / "wait_time_model.joblib"
csv_path = folder / "departments.csv"

if not model_path.exists() or not csv_path.exists():
    st.error(
        "Required files are missing. Keep wait_time_model.joblib "
        "and departments.csv in the same folder as app.py."
    )
    st.stop()

try:
    model = joblib.load(model_path)
    departments = pd.read_csv(csv_path)
except Exception as e:
    st.error(f"Could not load the model or department data: {e}")
    st.stop()

required_columns = [
    "department_name",
    "department_id",
    "staff_count",
    "bed_count",
    "daily_capacity"
]

missing = [c for c in required_columns if c not in departments.columns]

if missing:
    st.error(f"Missing CSV columns: {', '.join(missing)}")
    st.stop()

departments = departments.dropna(subset=["department_name"])

if departments.empty:
    st.error("No valid department records were found.")
    st.stop()

# ---------------- DASHBOARD OVERVIEW ----------------
overview_tab, prediction_tab, insights_tab = st.tabs(
    ["📊 Overview", "⏱️ Predict Waiting Time", "💡 Resource Insights"]
)

with overview_tab:
    st.subheader("Hospital Resource Overview")

    total_departments = departments["department_name"].nunique()
    total_staff = departments["staff_count"].sum()
    total_beds = departments["bed_count"].sum()
    total_capacity = departments["daily_capacity"].sum()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Departments", f"{total_departments:,}")
    c2.metric("Listed Staff", f"{total_staff:,.0f}")
    c3.metric("Listed Beds", f"{total_beds:,.0f}")
    c4.metric("Daily Capacity", f"{total_capacity:,.0f}")

    st.caption(
        "These totals describe the department records in the supplied CSV. "
        "They are not live hospital statistics."
    )

    st.divider()
    st.subheader("Compare Department Resources")

    resource_choice = st.selectbox(
        "Choose a resource to compare",
        ["staff_count", "bed_count", "daily_capacity"],
        format_func=lambda x: {
            "staff_count": "Staff count",
            "bed_count": "Bed count",
            "daily_capacity": "Daily capacity"
        }[x],
        key="overview_resource"
    )

    resource_data = departments[
        ["department_name", resource_choice]
    ].copy()

    resource_data[resource_choice] = pd.to_numeric(
        resource_data[resource_choice], errors="coerce"
    )
    resource_data = resource_data.dropna().set_index("department_name")

    st.bar_chart(resource_data)

    st.caption(
        "Resource comparisons only. Higher or lower resource counts alone "
        "do not establish whether a department is overcrowded or understaffed."
    )

    with st.expander("View department records"):
        st.dataframe(departments, use_container_width=True)

# ---------------- PREDICTION ----------------
with prediction_tab:
    st.subheader("Estimate Patient Waiting Time")
    st.write("Enter the visit details to generate a model prediction.")

    with st.form("prediction_form"):
        dept_name = st.selectbox(
            "Department",
            departments["department_name"].tolist()
        )

        visit_type = st.selectbox(
            "Visit Type",
            ["Walk-in", "Emergency", "Scheduled", "Follow-up"]
        )

        triage = st.selectbox(
            "Triage Level",
            ["Low", "Medium", "High", "Critical"]
        )

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age",
                min_value=1,
                max_value=110,
                value=30
            )

            condition = st.selectbox(
                "Chronic Condition",
                [
                    "Unknown", "Hypertension", "Diabetes",
                    "Other", "Asthma", "Cardiac"
                ]
            )

        with col2:
            arrival_date = st.date_input("Arrival Date")
            arrival_time = st.time_input("Arrival Time")

        submitted = st.form_submit_button(
            "Predict Waiting Time",
            use_container_width=True
        )

    if submitted:
        dept = departments[
            departments["department_name"] == dept_name
        ].iloc[0]

        arrival = datetime.combine(arrival_date, arrival_time)

        inputs = pd.DataFrame([{
            "department_id": dept["department_id"],
            "visit_type": visit_type,
            "triage_level": triage,
            "hour": arrival.hour,
            "day_of_week": arrival.weekday(),
            "month": arrival.month,
            "staff_count": dept["staff_count"],
            "bed_count": dept["bed_count"],
            "daily_capacity": dept["daily_capacity"],
            "age": age,
            "chronic_condition": condition
        }])

        try:
            prediction = max(0, float(model.predict(inputs)[0]))

            st.divider()
            st.subheader("Prediction Result")

            result_col, dept_col, time_col = st.columns(3)

            result_col.metric(
                "Estimated Waiting Time",
                f"{prediction:.1f} minutes"
            )
            dept_col.metric("Department", dept_name)
            time_col.metric(
                "Arrival Time",
                arrival.strftime("%I:%M %p")
            )

            st.write("**Visit summary**")
            st.write(
                f"- Visit type: {visit_type}\n"
                f"- Triage level: {triage}\n"
                f"- Age entered: {age}\n"
                f"- Arrival date: {arrival_date.strftime('%d %b %Y')}"
            )

            st.warning(
                "This is a model estimate, not a guaranteed actual waiting "
                "time. It does not determine medical urgency or replace "
                "hospital staff judgment."
            )

            # Explore predictions at different hours.
            st.subheader("Explore Estimated Waiting Time by Hour")

            hourly_inputs = pd.concat(
                [inputs.copy() for _ in range(24)],
                ignore_index=True
            )
            hourly_inputs["hour"] = list(range(24))

            hourly_predictions = model.predict(hourly_inputs)

            hourly_data = pd.DataFrame({
                "Hour": list(range(24)),
                "Estimated waiting time (minutes)": [
                    max(0, float(x)) for x in hourly_predictions
                ]
            })

            hourly_data["Time"] = hourly_data["Hour"].apply(
                lambda h: datetime.strptime(
                    f"{h:02d}:00", "%H:%M"
                ).strftime("%I %p")
            )

            st.line_chart(
                hourly_data.set_index("Time")[
                    ["Estimated waiting time (minutes)"]
                ]
            )

            st.caption(
                "These are model-generated scenario estimates for the same "
                "entered visit and department, varying only the hour. "
                "They are not historical observations or a live forecast."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ---------------- RESOURCE INSIGHTS ----------------
with insights_tab:
    st.subheader("Department Resource Insights")

    st.write(
        "Review department resource counts to support further investigation "
        "and planning discussions."
    )

    insight_metric = st.selectbox(
        "Analyze",
        ["staff_count", "bed_count", "daily_capacity"],
        format_func=lambda x: {
            "staff_count": "Staff count",
            "bed_count": "Bed count",
            "daily_capacity": "Daily capacity"
        }[x],
        key="insight_metric"
    )

    ranked = departments[
        ["department_name", insight_metric]
    ].copy()

    ranked[insight_metric] = pd.to_numeric(
        ranked[insight_metric], errors="coerce"
    )
    ranked = ranked.dropna().sort_values(
        insight_metric, ascending=False
    )

    if not ranked.empty:
        st.write(f"**Departments ranked by {insight_metric.replace('_', ' ')}**")
        st.dataframe(
            ranked.reset_index(drop=True),
            use_container_width=True
        )

        highest = ranked.iloc[0]
        lowest = ranked.iloc[-1]

        st.write(
            f"The highest listed value is for **{highest['department_name']}**; "
            f"the lowest listed value is for **{lowest['department_name']}**."
        )

        st.caption(
            "These comparisons describe the supplied dataset only. "
            "They do not prove resource shortages, excessive workload, "
            "or patient waiting-time causes. Validate any proposed staffing "
            "changes using reliable operational data and expert review."
        )
    else:
        st.warning("No numeric resource values are available to compare.")

# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "Smart Hospital Flow Predictor | Educational prototype using synthetic "
    "data | Not for clinical decisions"
)
