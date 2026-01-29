# ============================================================
# HOTEL BOOKING DEMAND FORECASTING & CANCELLATION ANALYTICS
# FULL COMBINED PIPELINE + DEMO REVIEW 2
# ============================================================
# This script covers:
# 1. Output folder creation
# 2. Logging to output/output.txt
# 3. Data loading
# 4. Data cleaning & preprocessing
# 5. Feature engineering
# 6. EDA with saved plots
# 7. Correlation analysis
# 8. Statistical insights
# 9. Clean data saving
# 10. Baseline ML models
# 11. Feature importance
# 12. Learning curves
# 13. Hyperparameter tuning
# 14. Model calibration
# 15. Business threshold optimization
# 16. Statistical significance testing
# ============================================================


# ============================================================
# STEP 0: OUTPUT DIRECTORY & LOGGING SETUP
# ============================================================

import os
import sys

BASE_OUTPUT_DIR = "output"
PLOTS_DIR = os.path.join(BASE_OUTPUT_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

output_file_path = os.path.join(BASE_OUTPUT_DIR, "output.txt")
output_file = open(output_file_path, "w", encoding="utf-8")
sys.stdout = output_file

print("============================================================")
print("PROJECT EXECUTION STARTED")
print("Output directory created:", BASE_OUTPUT_DIR)
print("Plots directory created:", PLOTS_DIR)
print("============================================================")


# ============================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import time

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    learning_curve
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import ttest_rel

sns.set_style("whitegrid")

print("\nSTEP 1 COMPLETED: Libraries imported successfully")


# ============================================================
# STEP 2: LOAD RAW DATA
# ============================================================

df = pd.read_csv("data/hotel_booking_raw_1000.csv")

print("\nSTEP 2 COMPLETED: Raw dataset loaded")
print("Dataset shape:", df.shape)
# ============================================================
# STEP 2.1: BEFORE CLEANING SUMMARY
# ============================================================

before_rows = df.shape[0]
before_missing = df.isnull().sum().sum()
before_duplicates = df.duplicated().sum()

print("\nBEFORE CLEANING")
print("Rows:", before_rows)
print("Missing values:", before_missing)
print("Duplicate rows:", before_duplicates)



# ============================================================
# STEP 3: DATA TYPE CORRECTION (5 TYPES)
# ============================================================

df["lead_time"] = df["lead_time"].fillna(df["lead_time"].median()).astype(int)
df["arrival_day"] = df["arrival_day"].astype(int)
df["price_per_night"] = df["price_per_night"].astype(float)
df["children"] = df["children"].fillna(0).astype(int)
df["is_canceled"] = df["is_canceled"].fillna(0).astype(int)

print("\nSTEP 3 COMPLETED: Data types corrected (5 fields)")


# ============================================================
# STEP 4: CLEANING PIPELINE
# ============================================================

def cleaning_pipeline(data):
    data = data.drop_duplicates()

    data["booking_channel"].fillna(
        data["booking_channel"].mode()[0], inplace=True
    )
    data["special_requests"].fillna(0, inplace=True)

    data["lead_time"] = np.where(
        data["lead_time"] > 300, 300, data["lead_time"]
    )
    data["price_per_night"] = np.where(
        data["price_per_night"] > 12000, 12000, data["price_per_night"]
    )

    data = data[data["adults"] > 0]
    return data

df = cleaning_pipeline(df)
# ============================================================
# STEP 4.1: AFTER CLEANING SUMMARY
# ============================================================

after_rows = df.shape[0]
after_missing = df.isnull().sum().sum()
after_duplicates = df.duplicated().sum()

comparison_df = pd.DataFrame({
    "Metric": ["Rows", "Missing Values", "Duplicate Rows"],
    "Before Cleaning": [before_rows, before_missing, before_duplicates],
    "After Cleaning": [after_rows, after_missing, after_duplicates]
})

print("\nAFTER CLEANING SUMMARY")
print(comparison_df)


print("\nSTEP 4 COMPLETED: Cleaning pipeline applied")


# ============================================================
# STEP 5: FEATURE ENGINEERING (6 FEATURES)
# ============================================================

df["total_guests"] = df["adults"] + df["children"]
df["price_per_person"] = df["price_per_night"] / df["total_guests"]
df["is_weekend_stay"] = (df["stay_nights"] >= 2).astype(int)
df["is_long_stay"] = (df["stay_nights"] >= 5).astype(int)
df["high_cancellation_risk"] = (df["previous_cancellations"] >= 2).astype(int)
df["high_value_booking"] = (df["price_per_night"] >= 8000).astype(int)

print("\nSTEP 5 COMPLETED: Feature engineering completed")


# ============================================================
# STEP 6: EDA WITH SAVED PLOTS
# ============================================================

def save_plot(fig_name):
    plt.savefig(os.path.join(PLOTS_DIR, fig_name))
    plt.close()

plt.figure()
sns.countplot(x="room_type", data=df)
plt.title("Booking Distribution by Room Type")
save_plot("room_type_distribution.png")

plt.figure()
sns.countplot(x="is_canceled", data=df)
plt.title("Cancellation Distribution")
save_plot("cancellation_distribution.png")

plt.figure()
sns.boxplot(x="is_canceled", y="price_per_night", data=df)
plt.title("Price vs Cancellation")
save_plot("price_vs_cancellation.png")

plt.figure()
sns.histplot(df["lead_time"], bins=30, kde=True)
plt.title("Lead Time Distribution")
save_plot("lead_time_distribution.png")

plt.figure()
sns.countplot(x="booking_channel", hue="is_canceled", data=df)
plt.title("Booking Channel vs Cancellation")
save_plot("booking_channel_vs_cancellation.png")

print("\nSTEP 6 COMPLETED: EDA plots saved")


# ============================================================
# STEP 7: CORRELATION ANALYSIS
# ============================================================

plt.figure(figsize=(12, 8))
corr = df.select_dtypes(include="number").corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
save_plot("correlation_heatmap.png")

print("\nSTEP 7 COMPLETED: Correlation analysis saved")


# ============================================================
# STEP 8: STATISTICAL INSIGHTS
# ============================================================

print("\nSTEP 8: Statistical Insights")

print("\nDescriptive Statistics:")
print(df.describe())

print("\nOverall Cancellation Rate:")
print(df["is_canceled"].mean() * 100)

print("\nLead Time by Cancellation Status:")
print(df.groupby("is_canceled")["lead_time"].mean())

print("\nAverage Price by Cancellation Status:")
print(df.groupby("is_canceled")["price_per_night"].mean())

print("\nRoom Type Demand:")
print(df["room_type"].value_counts())

print("\nAverage Stay Nights by Room Type:")
print(df.groupby("room_type")["stay_nights"].mean())


# ============================================================
# STEP 9: SAVE CLEANED DATA
# ============================================================

df.to_csv("data/hotel_booking_cleaned.csv", index=False)
# ============================================================
# STEP 9.1: DATABASE INTEGRATION (SQLite)
# ============================================================

import sqlite3

db_path = os.path.join(BASE_OUTPUT_DIR, "hotel_analytics.db")
conn = sqlite3.connect(db_path)

df.to_sql("cleaned_hotel_bookings", conn, if_exists="replace", index=False)

conn.execute("""
CREATE TABLE IF NOT EXISTS cancellation_predictions (
    lead_time INTEGER,
    price_per_night REAL,
    stay_nights INTEGER,
    adults INTEGER,
    children INTEGER,
    prediction INTEGER,
    probability REAL
)
""")

conn.commit()

print("\nSTEP 9.1 COMPLETED: Database integration done")

print("\nSTEP 9 COMPLETED: Cleaned dataset saved")


# ============================================================
# STEP 10: ML – BASELINE MODEL BUILDING (DEMO REVIEW 2)
# ============================================================

target = "is_canceled"
features = [
    "lead_time",
    "price_per_night",
    "stay_nights",
    "adults",
    "children",
    "previous_cancellations"
]

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nSTEP 10 COMPLETED: Stratified train-test split")


models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
}

results = []

for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    results.append([name, acc, roc, fit_time])

baseline_results = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "ROC-AUC", "Fit Time"]
)

print("\nBaseline Model Performance:")
print(baseline_results)


# ============================================================
# STEP 11: CONFUSION MATRIX
# ============================================================

best_model = models["Random Forest"]
y_pred_best = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix – Random Forest")
save_plot("confusion_matrix.png")

print("\nSTEP 11 COMPLETED: Confusion matrix saved")


# ============================================================
# STEP 12: FEATURE IMPORTANCE (PIPELINE SAFE)
# ============================================================

rf_model = best_model.named_steps["model"]
importances = rf_model.feature_importances_

feat_imp_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(feat_imp_df)

plt.figure(figsize=(6, 4))
sns.barplot(x="Importance", y="Feature", data=feat_imp_df)
plt.title("Feature Importance – Random Forest")
save_plot("feature_importance.png")


# ============================================================
# STEP 13: LEARNING CURVE
# ============================================================

train_sizes, train_scores, test_scores = learning_curve(
    best_model, X, y, cv=5, scoring="roc_auc", n_jobs=-1
)

plt.plot(train_sizes, train_scores.mean(axis=1), label="Train")
plt.plot(train_sizes, test_scores.mean(axis=1), label="Validation")
plt.xlabel("Training Size")
plt.ylabel("ROC-AUC")
plt.title("Learning Curve – Random Forest")
plt.legend()
save_plot("learning_curve.png")

print("\nSTEP 13 COMPLETED: Learning curve saved")


# ============================================================
# STEP 14: HYPERPARAMETER TUNING
# ============================================================

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20]
}

grid = GridSearchCV(
    Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(random_state=42))
    ]),
    param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_rf = grid.best_estimator_

print("\nSTEP 14 COMPLETED: GridSearchCV finished")
print("Best Parameters:", grid.best_params_)


# ============================================================
# STEP 15: MODEL CALIBRATION
# ============================================================

calibrated_model = CalibratedClassifierCV(
    best_rf, method="isotonic", cv=5
)

calibrated_model.fit(X_train, y_train)
y_prob_cal = calibrated_model.predict_proba(X_test)[:, 1]

print("\nSTEP 15 COMPLETED: Model calibration done")


# ============================================================
# STEP 16: BUSINESS THRESHOLD OPTIMIZATION
# ============================================================

thresholds = np.arange(0.1, 0.9, 0.05)
profits = []

for t in thresholds:
    preds = (y_prob_cal >= t).astype(int)

    tp = ((preds == 1) & (y_test == 1)).sum()
    fp = ((preds == 1) & (y_test == 0)).sum()
    fn = ((preds == 0) & (y_test == 1)).sum()

    profit = tp * 100 - fp * 20 - fn * 50
    profits.append(profit)

plt.plot(thresholds, profits)
plt.xlabel("Threshold")
plt.ylabel("Profit")
plt.title("Business Threshold Optimization")
save_plot("threshold_optimization.png")

print("\nSTEP 16 COMPLETED: Threshold optimization saved")


# ============================================================
# STEP 17: STATISTICAL SIGNIFICANCE TEST
# ============================================================

rf_probs = best_model.predict_proba(X_test)[:, 1]
lr_probs = models["Logistic Regression"].predict_proba(X_test)[:, 1]

t_stat, p_value = ttest_rel(rf_probs, lr_probs)

print("\nPaired t-test Results:")
print("t-statistic:", t_stat)
print("p-value:", p_value)

print("\n============================================================")
print("PROJECT COMPLETED SUCCESSFULLY 🎯")
print("Check 'output/' folder for logs and plots")
print("============================================================")

# ============================================================
# CLOSE LOG FILE
# ============================================================

sys.stdout = sys.__stdout__
output_file.close()
