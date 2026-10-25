
import hashlib

import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sqlalchemy import create_engine, text

st.set_page_config(
    page_title="Crowdfunding Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Crowdfunding Donation Fraud Detection")
st.caption("Anomaly detection dashboard for reviewing donations.")

st.info(
    "Suspicious results are warnings for investigation, "
    "not proof of fraud."
)

SAMPLE_DATA = {
    "donation_id": [f"DON{i:03d}" for i in range(1, 21)],
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

NUMERIC_COLUMNS = [
    "amount",
    "donor_age_days",
    "donations_last_24h",
    "payment_failures",
    "country_mismatch"
]


# DATABASE CONNECTION
@st.cache_resource
def get_engine():
    return create_engine(
        st.secrets["database"]["url"],
        pool_pre_ping=True
    )


def initialize_database(engine):
    with engine.begin() as connection:
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS donations_history (
                record_id BIGSERIAL PRIMARY KEY,
                batch_key TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                donation_id TEXT NOT NULL,
                amount DOUBLE PRECISION NOT NULL,
                donor_age_days DOUBLE PRECISION NOT NULL,
                donations_last_24h DOUBLE PRECISION NOT NULL,
                payment_failures DOUBLE PRECISION NOT NULL,
                country_mismatch DOUBLE PRECISION NOT NULL,
                anomaly_score DOUBLE PRECISION NOT NULL,
                status TEXT NOT NULL,
                review_notes TEXT NOT NULL,
                UNIQUE (batch_key, donation_id)
            )
        """))


def save_donations(engine, data, batch_key):
    records = data[
        REQUIRED_COLUMNS + [
            "Anomaly score",
            "Status",
            "Review notes"
        ]
    ].copy()

    records = records.rename(columns={
        "Anomaly score": "anomaly_score",
        "Status": "status",
        "Review notes": "review_notes"
    })

    records["batch_key"] = batch_key

    columns = [
        "batch_key",
        "donation_id",
        "amount",
        "donor_age_days",
        "donations_last_24h",
        "payment_failures",
        "country_mismatch",
        "anomaly_score",
        "status",
        "review_notes"
    ]

    rows = records[columns].to_dict(orient="records")

    query = text("""
        INSERT INTO donations_history (
            batch_key, donation_id, amount, donor_age_days,
            donations_last_24h, payment_failures,
            country_mismatch, anomaly_score, status, review_notes
        )
        VALUES (
            :batch_key, :donation_id, :amount, :donor_age_days,
            :donations_last_24h, :payment_failures,
            :country_mismatch, :anomaly_score, :status, :review_notes
        )
        ON CONFLICT (batch_key, donation_id) DO NOTHING
    """)

    with engine.begin() as connection:
        result = connection.execute(query, rows)

    return result.rowcount


def load_history(engine):
    with engine.connect() as connection:
        return pd.read_sql(
            text("""
                SELECT
                    record_id,
                    created_at,
                    donation_id,
                    amount,
                    donor_age_days,
                    donations_last_24h,
                    payment_failures,
                    country_mismatch,
                    anomaly_score AS "Anomaly score",
                    status AS "Status",
                    review_notes AS "Review notes"
                FROM donations_history
                ORDER BY created_at DESC, record_id DESC
            """),
            connection
        )


# SIDEBAR AND DATA INPUT
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
df["donation_id"] = df["donation_id"].astype("string").str.strip()

if df["donation_id"].isna().any() or (
    df["donation_id"] == ""
).any():
    st.error("Every donation must have a donation ID.")
    st.stop()

if df["donation_id"].duplicated().any():
    st.error("Each donation ID must be unique within the uploaded dataset.")
    st.stop()

for column in NUMERIC_COLUMNS:
    df[column] = pd.to_numeric(df[column], errors="coerce")

if df[NUMERIC_COLUMNS].isna().any().any():
    st.error("Some numeric fields are empty or invalid. Please correct your CSV.")
    st.stop()

if len(df) < 5:
    st.error("Please provide at least 5 valid donation records.")
    st.stop()

if not df[NUMERIC_COLUMNS].apply(
    lambda column: column.map(lambda value: abs(value) != float("inf"))
).all().all():
    st.error("Numeric fields must contain finite values.")
    st.stop()

if (df[NUMERIC_COLUMNS] < 0).any().any():
    st.error("Numeric donation features cannot contain negative values.")
    st.stop()

if not df["country_mismatch"].isin([0, 1]).all():
    st.error("country_mismatch must contain only 0 or 1.")
    st.stop()


# FRAUD DETECTION
features = df[NUMERIC_COLUMNS].copy()

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
        return "Flagged by model; no simple rule matched."

    return "; ".join(reasons)


df["Review notes"] = df.apply(explain_row, axis=1)


# DASHBOARD OVERVIEW
total = len(df)
suspicious = int(
    (df["Status"] == "Suspicious — review").sum()
)
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
st.bar_chart(df["Status"].value_counts())

st.subheader("Donation amount overview")
st.bar_chart(df.set_index("donation_id")["amount"])


# SAVE CURRENT ANALYSIS
st.subheader("Save donation history")

st.write(
    "Save this analysis to your online database. "
    "The same dataset will not be saved twice under the same batch."
)

batch_data = df[REQUIRED_COLUMNS].sort_values(
    "donation_id"
).to_json(orient="records")

batch_key = hashlib.sha256(
    batch_data.encode("utf-8")
).hexdigest()

if st.button("Save donations to database", type="primary"):
    try:
        engine = get_engine()
        initialize_database(engine)
        saved_count = save_donations(engine, df, batch_key)

        if saved_count:
            st.success(
                f"Saved {saved_count} donation record(s) to history."
            )
        else:
            st.info(
                "This dataset has already been saved. "
                "No duplicate records were added."
            )

    except Exception as exc:
        st.error(
            "Could not save to the database. Check your Streamlit "
            "Secrets and Neon connection settings."
        )
        st.caption(f"Technical details: {exc}")


# SEARCH AND FILTER
st.subheader("Search and filter donations")

search = st.text_input("Search by donation ID")

status_filter = st.selectbox(
    "Filter by status",
    ["All", "Normal", "Suspicious — review"]
)

filtered = df.copy()

if search.strip():
    filtered = filtered[
        filtered["donation_id"].astype(str).str.contains(
            search.strip(), case=False, na=False
        )
    ]

if status_filter != "All":
    filtered = filtered[filtered["Status"] == status_filter]

st.dataframe(filtered, use_container_width=True)


# FLAGGED DONATIONS
st.subheader("Review flagged donations")

flagged = df[df["Status"] == "Suspicious — review"]

if flagged.empty:
    st.success("No donations were flagged in this dataset.")
else:
    st.warning(
        "Review flagged records before taking action. "
        "An anomaly alone does not prove fraud."
    )
    st.dataframe(
        flagged[
            [
                "donation_id",
                "amount",
                "Status",
                "Anomaly score",
                "Review notes"
            ]
        ],
        use_container_width=True
    )


# PERSISTENT DATABASE HISTORY
st.subheader("Saved donation history")

try:
    engine = get_engine()
    initialize_database(engine)
    history = load_history(engine)

    if history.empty:
        st.info(
            "No donation history has been saved yet. "
            "Use the Save donations to database button above."
        )
    else:
        st.metric("Saved donation records", len(history))
        st.dataframe(history, use_container_width=True)

        history_csv = history.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download saved history (CSV)",
            data=history_csv,
            file_name="saved_donation_history.csv",
            mime="text/csv"
        )

except Exception:
    st.warning(
        "Database history is currently unavailable. "
        "Check your Streamlit Secrets and Neon connection."
    )


# DOWNLOAD CURRENT ANALYSIS
csv_data = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download analysis report (CSV)",
    data=csv_data,
    file_name="donation_fraud_analysis.csv",
    mime="text/csv"
)

st.caption(
    "Academic prototype. Sample donations are illustrative, "
    "and model results depend on the supplied data."
)
