import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Crowdfunding Fraud Detection", page_icon="🛡️", layout="wide")
st.title("🛡️ Crowdfunding Donation Fraud Detection System")
st.caption("Academic prototype using anomaly detection. A flag means 'review this donation', not proof of fraud.")

REQUIRED = ["donation_id", "amount", "donor_age_days", "donations_last_24h", "payment_failures", "country_mismatch"]

@st.cache_data
def load_sample():
    return pd.read_csv("sample_donations.csv")

def detect(df):
    result = df.copy()
    features = ["amount", "donor_age_days", "donations_last_24h", "payment_failures", "country_mismatch"]
    X = result[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    model = IsolationForest(n_estimators=150, contamination=0.12, random_state=42)
    result["anomaly"] = model.fit_predict(X)
    result["anomaly_score"] = model.decision_function(X)
    result["status"] = result["anomaly"].map({1: "Normal", -1: "Suspicious — review"})
    def reasons(row):
        reasons = []
        if row["amount"] >= 500:
            reasons.append("unusually high amount")
        if row["donor_age_days"] <= 2:
            reasons.append("very new donor account")
        if row["donations_last_24h"] >= 5:
            reasons.append("many donations in 24 hours")
        if row["payment_failures"] >= 2:
            reasons.append("repeated payment failures")
        if row["country_mismatch"] == 1:
            reasons.append("country mismatch")
        return ", ".join(reasons) if reasons else "No simple rule-based warning"
    result["review_notes"] = result.apply(reasons, axis=1)
    return result

uploaded = st.file_uploader("Upload a donations CSV file", type=["csv"])
if uploaded:
    raw = pd.read_csv(uploaded)
else:
    st.info("You are viewing the included sample dataset. Upload your own CSV to test other donations.")
    raw = load_sample()

missing = [c for c in REQUIRED if c not in raw.columns]
if missing:
    st.error("The CSV is missing required columns: " + ", ".join(missing))
    st.write("Required columns:", REQUIRED)
    st.stop()

for col in REQUIRED[1:]:
    raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0)

if st.button("Analyze donations", type="primary"):
    result = detect(raw)
    suspicious = result[result["status"] == "Suspicious — review"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Total donations", len(result))
    c2.metric("Flagged for review", len(suspicious))
    c3.metric("Flag rate", f"{len(suspicious)/max(len(result),1):.1%}")
    st.subheader("Analysis results")
    st.dataframe(result[["donation_id", "amount", "status", "anomaly_score", "review_notes"]], use_container_width=True)
    st.download_button("Download results as CSV", result.to_csv(index=False).encode("utf-8"), "donation_analysis_results.csv", "text/csv")
    st.subheader("Flagged donations")
    if suspicious.empty:
        st.success("No donations were flagged in this dataset.")
    else:
        st.dataframe(suspicious[["donation_id", "amount", "status", "review_notes"]], use_container_width=True)
    st.caption("Model limitation: this is a teaching prototype trained on the uploaded batch, not a production fraud detector. Review flags manually and never use it as the sole basis for blocking a donor.")
else:
    st.subheader("Preview of donation data")
    st.dataframe(raw, use_container_width=True)
    st.write("Tap **Analyze donations** to run the anomaly detector.")
