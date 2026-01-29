# ============================================================
# 🏨 HOTEL BOOKING DEMAND & CANCELLATION ANALYTICS DASHBOARD
# WITH DB HISTORY + PDF EXPORT + ARIMA FORECASTING (FINAL)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from statsmodels.tsa.arima.model import ARIMA

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Hotel Booking Analytics",
    page_icon="🏨",
    layout="wide"
)

sns.set_style("whitegrid")

# ============================================================
# PATHS
# ============================================================
DATA_PATH = "data/hotel_booking_cleaned.csv"
DB_PATH = "output/hotel_analytics.db"
os.makedirs("output", exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(DATA_PATH)

# Standardize room types
df["room_type"] = df["room_type"].replace({
    "Standard": "Ordinary",
    "standard": "Ordinary"
})

# ============================================================
# DATABASE SETUP
# ============================================================
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS room_demand_predictions (
    timestamp TEXT,
    room_type TEXT,
    predicted_demand INTEGER
)
""")
conn.commit()

# ============================================================
# HEADER
# ============================================================
st.title("🏨 Hotel Booking Demand & Cancellation Analytics")
st.caption("Database-Driven Dashboard | Review-3 Evaluation")

# ============================================================
# KPI METRICS
# ============================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Bookings", len(df))
c2.metric("Cancellation Rate (%)", round(df["is_canceled"].mean()*100, 2))
c3.metric("Avg Price / Night", round(df["price_per_night"].mean(), 2))
c4.metric("Avg Lead Time (Days)", round(df["lead_time"].mean(), 1))

# ============================================================
# CANCELLATION MODEL
# ============================================================
CANCEL_FEATURES = [
    "lead_time", "price_per_night", "stay_nights",
    "adults", "children", "previous_cancellations"
]

X = df[CANCEL_FEATURES]
y = df["is_canceled"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

cancel_model = RandomForestClassifier(n_estimators=200, random_state=42)
cancel_model.fit(X_train, y_train)

# ============================================================
# DEMAND MODEL (ROOM TYPE)
# ============================================================
room_counts = df["room_type"].value_counts().reset_index()
room_counts.columns = ["room_type", "demand"]

X_room_train = pd.get_dummies(room_counts["room_type"])
y_room_train = room_counts["demand"]

demand_model = RandomForestRegressor(n_estimators=150, random_state=42)
demand_model.fit(X_room_train, y_room_train)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "❌ Cancellation Analytics",
    "📌 Demand Forecasting",
    "📊 ARIMA Month-wise Forecast",
    "🗄️ Database History"
])

# ============================================================
# TAB 1: CANCELLATION ANALYTICS
# ============================================================
with tab1:
    st.subheader("🧪 Cancellation Prediction")

    with st.form("cancel_form"):
        lead_time = st.slider("Lead Time", 0, 300, 60)
        price = st.slider("Price per Night", 1000, 15000, 7000)
        stay = st.slider("Stay Nights", 1, 15, 3)
        adults = st.slider("Adults", 1, 4, 2)
        children = st.slider("Children", 0, 3, 0)
        submit = st.form_submit_button("Predict")

    if submit:
        X_input = pd.DataFrame([[lead_time, price, stay, adults, children, 0]],
                               columns=CANCEL_FEATURES)
        pred = cancel_model.predict(X_input)[0]
        prob = cancel_model.predict_proba(X_input)[0][1]

        st.success(f"Prediction: {'Canceled ❌' if pred else 'Not Canceled ✅'}")
        st.info(f"Cancellation Probability: {round(prob*100,2)}%")

# ============================================================
# TAB 2: ROOM TYPE DEMAND FORECAST (ML)
# ============================================================
with tab2:
    st.subheader("🏨 Room-Type Demand Forecast")

    selected_room = st.selectbox(
        "Select Room Type",
        ["Ordinary", "Deluxe", "Suite"]
    )

    if st.button("Predict Demand"):
        input_df = pd.DataFrame({"room_type": [selected_room]})
        X_input = pd.get_dummies(input_df["room_type"])
        X_input = X_input.reindex(columns=X_room_train.columns, fill_value=0)

        predicted_demand = int(demand_model.predict(X_input)[0])

        st.success(
            f"Predicted demand for {selected_room}: {predicted_demand}"
        )

        cursor.execute("""
            INSERT INTO room_demand_predictions VALUES (?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            selected_room,
            predicted_demand
        ))
        conn.commit()

        st.info("Prediction saved to database ✅")

    # ---------------- PDF EXPORT ----------------
    def generate_pdf(room, demand):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(0, 10, "Hotel Analytics Report", ln=True)
        pdf.ln(5)
        pdf.cell(0, 10, f"Generated: {datetime.now()}", ln=True)
        pdf.cell(0, 10, f"Room Type: {room}", ln=True)
        pdf.cell(0, 10, f"Predicted Demand: {demand}", ln=True)

        file_path = "output/dashboard_report.pdf"
        pdf.output(file_path)
        return file_path

    if st.button("📄 Export PDF Report"):
        pdf_path = generate_pdf(selected_room, predicted_demand)
        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download PDF",
                data=f,
                file_name="hotel_demand_report.pdf",
                mime="application/pdf"
            )

# ============================================================
# TAB 3: ARIMA MONTH-WISE FORECAST
# ============================================================
with tab3:
    st.subheader("📊 Month-wise Future Demand Forecast (ARIMA)")

    monthly_demand = (
        df.groupby("arrival_month")
        .size()
        .reset_index(name="bookings")
    )

    monthly_demand.index = pd.RangeIndex(start=1, stop=len(monthly_demand)+1)

    model = ARIMA(monthly_demand["bookings"], order=(1, 1, 1))
    model_fit = model.fit()

    future_steps = st.slider("Forecast Months", 1, 12, 6)
    forecast = model_fit.forecast(steps=future_steps)

    fig, ax = plt.subplots()
    ax.plot(monthly_demand["bookings"], label="Historical")
    ax.plot(
        range(len(monthly_demand)+1, len(monthly_demand)+future_steps+1),
        forecast,
        label="Forecast",
        linestyle="--"
    )
    ax.set_title("ARIMA Month-wise Demand Forecast")
    ax.set_xlabel("Time")
    ax.set_ylabel("Bookings")
    ax.legend()

    st.pyplot(fig)

# ============================================================
# TAB 4: DATABASE HISTORY
# ============================================================
with tab4:
    st.subheader("🗄️ Demand Prediction History")

    history_df = pd.read_sql(
        "SELECT * FROM room_demand_predictions ORDER BY timestamp DESC",
        conn
    )

    st.dataframe(history_df, width="stretch")

    if not history_df.empty:
        fig, ax = plt.subplots()
        sns.barplot(
            data=history_df,
            x="room_type",
            y="predicted_demand",
            hue="room_type",
            legend=False,
            ax=ax
        )
        ax.set_title("Historical Demand Predictions by Room Type")
        st.pyplot(fig)

# ============================================================
# FOOTER
# ============================================================
st.success("Dashboard executed successfully ")
