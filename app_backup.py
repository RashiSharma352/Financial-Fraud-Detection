import streamlit as st
import pandas as pd
import joblib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_excel("Data/Fraud Detection.xlsx")

# Keep only valid transactions
df = df[df["isFraud"].isin(["Safe", "Fraud"])].copy()

# Target label
df["Fraud_Label"] = df["isFraud"].map({
    "Safe": 0,
    "Fraud": 1
})

# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("models/fraud_model.pkl")

# ============================================================
# TITLE
# ============================================================

st.title("💳 Financial Fraud Detection Dashboard")

st.write(
    "Machine Learning based system for analyzing and detecting "
    "potential fraudulent financial transactions."
)

st.divider()

# ============================================================
# KPI CARDS
# ============================================================

total_transactions = len(df)
fraud_transactions = int(df["Fraud_Label"].sum())
safe_transactions = total_transactions - fraud_transactions
fraud_rate = (fraud_transactions / total_transactions) * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Transactions",
    f"{total_transactions:,}"
)

col2.metric(
    "Fraud Transactions",
    f"{fraud_transactions:,}"
)

col3.metric(
    "Safe Transactions",
    f"{safe_transactions:,}"
)

col4.metric(
    "Fraud Rate",
    f"{fraud_rate:.2f}%"
)

st.divider()

# ============================================================
# FRAUD DISTRIBUTION
# ============================================================

st.header("📊 Fraud Overview")

col1, col2 = st.columns(2)

with col1:

    fraud_counts = df["isFraud"].value_counts()

    st.subheader("Safe vs Fraud")

    st.bar_chart(fraud_counts)

with col2:

    st.subheader("Fraud by Transaction Type")

    transaction_fraud = pd.crosstab(
        df["type"],
        df["isFraud"]
    )

    st.bar_chart(transaction_fraud)

# ============================================================
# AMOUNT ANALYSIS
# ============================================================

st.header("💰 Transaction Amount Analysis")

amount_summary = (
    df.groupby("isFraud")["amount"]
    .mean()
    .round(2)
)

st.bar_chart(amount_summary)

# ============================================================
# BRANCH ANALYSIS
# ============================================================

st.header("🏦 Fraud by Branch")

branch_fraud = pd.crosstab(
    df["branch"],
    df["isFraud"]
)

st.bar_chart(branch_fraud)

# ============================================================
# ACCOUNT BALANCE ANALYSIS
# ============================================================

st.header("💰 Balance Analysis")

balance_summary = df.groupby("isFraud")[
    ["oldbalanceOrg", "newbalanceOrig"]
].mean()

st.dataframe(
    balance_summary.round(2),
    use_container_width=True
)

# ============================================================
# FRAUD PREDICTION
# ============================================================

st.divider()

st.header("🔍 Fraud Prediction")

st.write(
    "Enter transaction details below. The trained Machine Learning "
    "model will predict whether the transaction is Safe or potentially Fraudulent."
)

col1, col2 = st.columns(2)

with col1:

    step = st.number_input(
        "Step",
        min_value=0,
        value=7
    )

    transaction_type = st.selectbox(
        "Transaction Type",
        sorted(df["type"].dropna().unique())
    )

    branch = st.selectbox(
        "Branch",
        sorted(df["branch"].dropna().unique())
    )

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=1000.0
    )

    oldbalance_org = st.number_input(
        "Old Balance (Origin)",
        min_value=0.0,
        value=10000.0
    )

    newbalance_orig = st.number_input(
        "New Balance (Origin)",
        min_value=0.0,
        value=9000.0
    )

with col2:

    oldbalance_dest = st.number_input(
        "Old Balance (Destination)",
        min_value=0.0,
        value=5000.0
    )

    newbalance_dest = st.number_input(
        "New Balance (Destination)",
        min_value=0.0,
        value=6000.0
    )

    unusual_login = st.number_input(
        "Unusual Login",
        min_value=0.0,
        value=10.0
    )

    flagged_fraud = st.selectbox(
        "Flagged Fraud",
        [0, 1]
    )

    account_type = st.selectbox(
        "Account Type",
        sorted(df["Acct type"].dropna().unique())
    )

    time_of_day = st.selectbox(
        "Time of Day",
        sorted(df["Time of day"].dropna().unique())
    )

    day_of_week = st.number_input(
        "Day of Week",
        min_value=1,
        max_value=7,
        value=3
    )

    day_name = st.selectbox(
        "Day Name",
        sorted(df["DayOfWeek(new)"].dropna().unique())
    )

    transaction_year = st.number_input(
        "Transaction Year",
        min_value=2000,
        max_value=2100,
        value=2026
    )

    transaction_month = st.number_input(
        "Transaction Month",
        min_value=1,
        max_value=12,
        value=8
    )

    transaction_day = st.number_input(
        "Transaction Day",
        min_value=1,
        max_value=31,
        value=10
    )

# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔎 Predict Transaction",
    type="primary"
):

    input_data = pd.DataFrame({

        "step": [step],

        "type": [transaction_type],

        "branch": [branch],

        "amount": [amount],

        "oldbalanceOrg": [oldbalance_org],

        "newbalanceOrig": [newbalance_orig],

        "oldbalanceDest": [oldbalance_dest],

        "newbalanceDest": [newbalance_dest],

        "unusuallogin": [unusual_login],

        "isFlaggedFraud": [flagged_fraud],

        "Acct type": [account_type],

        "Time of day": [time_of_day],

        "DayOfWeek": [day_of_week],

        "DayOfWeek(new)": [day_name],

        "Transaction_Year": [transaction_year],

        "Transaction_Month": [transaction_month],

        "Transaction_Day": [transaction_day]
    })

    prediction = model.predict(input_data)[0]

    if prediction == 1:

        st.error(
            "🚨 Potential Fraudulent Transaction Detected!"
        )

    else:

        st.success(
            "✅ Transaction appears to be Safe."
        )

st.divider()

st.caption(
    "Financial Fraud Detection | Python + Scikit-learn + Random Forest + Streamlit"
)
