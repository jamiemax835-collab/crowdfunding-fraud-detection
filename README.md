# Crowdfunding Donation Fraud Detection System

A beginner-friendly academic web-app prototype that uses Isolation Forest anomaly detection to flag unusual donation patterns for manual review.

## Features
- Preview a sample donations dataset
- Upload a CSV file
- Run Isolation Forest anomaly detection
- Show suspicious donations and simple review notes
- Download analysis results as CSV

## Run on a computer
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Run `pip install -r requirements.txt`.
4. Run `streamlit run app.py`.
5. Open the local URL shown in the terminal.

## Run from an Android phone
The easiest route is to upload these files to a GitHub repository and deploy the app using Streamlit Community Cloud. Create a GitHub account/repository, upload all four files, then connect the repository at https://share.streamlit.io/ and deploy `app.py`. Deployment providers may change their setup and free-tier rules.

## CSV format
Required columns:
- `donation_id`: transaction identifier
- `amount`: donation amount (use one currency consistently)
- `donor_age_days`: age of donor account in days
- `donations_last_24h`: number of donations by this donor in the last 24 hours
- `payment_failures`: recent failed payment attempts
- `country_mismatch`: 1 if country details mismatch, otherwise 0

## Important limitations
This is a teaching prototype, not a production-grade fraud system. The sample dataset is illustrative, not real donor data. Isolation Forest detects unusual patterns; it does not prove fraud. Do not upload real personal or payment-card data. Review flagged donations manually.
