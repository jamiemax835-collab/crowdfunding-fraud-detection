import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest

st.set_page_config(
page_title="Crowdfunding Fraud Detection",
page_icon="🛡️",
layout="wide"
)

st.title("🛡️ Crowdfunding Donation Fraud Detection")
st.caption("Anomaly detection dashboard for identifying donations that may need review.")

st.info(
"This system identifies unusual donation patterns. "
"A suspicious result is a warning for review, not proof of fraud."
)

SAMPLE_DATA = {
"donation_id": [
f"DON{i:03d}" for i in range(1, 21)
],
"amount": [
20, 50, 100, 30, 75, 5000, 40, 60, 80, 100,
25, 9000, 45, 70, 55, 150, 30, 6500, 90, 35
],
"donor_age_days": [
200, 400, 120, 300, 250, 1, 180, 220, 90, 350,
160, 1, 280, 190, 240, 310, 130, 2, 170, 260
],
"donations_last_24h": [
1, 2, 1, 1, 3, 20, 1, 2, 1, 2,
1, 25, 2, 1, 3, 2, 1, 18, 2, 1
],
"payment_failures": [
0, 0, 0, 0, 1, 5, 0, 0, 0, 0,
0, 6, 0, 0, 1, 0, 0, 4, 0, 0
],
"country_mismatch": [
0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
0, 1, 0, 0, 0, 0, 0, 1, 0, 0
]
}

REQUIRED_COLUMNS = [
"donation_id",
"amount",
"donor_age_days",
"donations_last_24h",
"payment_failures",
"country_mismatch"
]

st.sidebar.header("Donation data")
uploaded_file = st.sidebar.file_uploader(
"Upload a donations CSV file",
type=["csv"]
)

if uploaded_file is not None:
   try:
      df = pd.read_csv(uploaded_file)
   except Exception as exc:
      st.error(f"Could not read the CSV file: {exc}")
      st.stop()
else:
  df = pd.DataFrame(SAMPLE_DATA)
  st.sidebar.success("Using illustrative sample donations.")

missing = [
column for column in REQUIRED_COLUMNS
if column not in df.columns
]

if missing:
   st.error(
      "Your CSV is missing these columns: "
      + ", ".join(missing)
   )
   st.stop()

df = df[REQUIRED_COLUMNS].copy()

numeric_columns = [
"amount",
"donor_age_days",
"donations_last_24h",
"payment_failures",
"country_mismatch"
]

for column in numeric_columns:
   df[column] = pd.to_numeric(df[column],
errors="coerce")

df = df.dropna(subset=numeric_columns)

if len(df) < 5:
st.error("Please provide at least 5 valid donation records.")
st.stop()

if (df[numeric_columns] < 0).any().any():
st.error("Numeric donation features cannot contain negative values.")
st.stop()

features = df[numeric_columns].copy()
features["country_mismatch"] = features["country_mismatch"].clip(0, 1)

model = IsolationForest(
contamination=0.12,
random_state=42
)

predictions = model.fit_predict(features)
df["Anomaly score"] = model.decision_function(features)
df["Status"] = [
"Suspicious — review" if prediction == -1 else "Normal"
for prediction in predictions
]

def explain_row(row):
reasons = []

if row["amount"] >= 1000:
    reasons.append("Unusually large amount")
if row["donor_age_days"] <= 3:
    reasons.append("Very new donor account")
if row["donations_last_24h"] >= 10:
    reasons.append("Many donations in 24 hours")
if row["payment_failures"] >= 3:
    reasons.append("Repeated payment failures")
if row["country_mismatch"] == 1:
    reasons.append("Country mismatch")

if not reasons:
    return "Flagged by the anomaly model; no simple rule matched."

return "; ".join(reasons)

df["Review notes"] = df.apply(explain_row, axis=1)

total = len(df)
suspicious = int((df["Status"] == "Suspicious — review").sum())
normal = total - suspicious
risk_percentage = suspicious / total * 100
total_amount = df["amount"].sum()

st.subheader("Dashboard overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total donations", f"{total:,}")
col2.metric("Flagged for review", f"{suspicious:,}")
col3.metric("Normal results", f"{normal:,}")
col4.metric("Review rate", f"{risk_percentage:.1f}%")

st.metric("Total donation amount", f"{total_amount:,.2f}")

st.subheader("Donation status breakdown")
status_counts = df["Status"].value_counts()
st.bar_chart(status_counts)

st.subheader("Donation amount overview")
st.bar_chart(
df.set_index("donation_id")["amount"]
)

st.subheader("Search and filter donations")

search = st.text_input("Search by donation ID")
status_filter = st.selectbox(
"Filter by status",
["All", "Normal", "Suspicious — review"]
)

filtered = df.copy()

if search.
