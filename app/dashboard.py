import streamlit as st
import pandas as pd
import joblib
import networkx as nx
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="AI B2B Fraud Triage Dashboard", layout="wide")

st.title("🛡️ B2B Financial Fraud Detection & Triage Engine")
st.caption("Real-Time Graph Network Analysis & Explainable AI Fraud Scoring")

@st.cache_data
def load_data_and_model():
    model_artifact = joblib.load("data/fraud_model.pkl")
    df = pd.read_csv("data/processed_features.csv")
    
    # Generate Predictions
    model = model_artifact["model"]
    features = model_artifact["features"]
    
    X = df[features]
    df["risk_score"] = model.predict_proba(X)[:, 1]
    df["is_flagged"] = (df["risk_score"] >= 0.50).astype(int)
    
    return df, model_artifact

df, model_artifact = load_data_and_model()

# Metrics Overview
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{len(df):,}")
col2.metric("Total Spend Analyzed", f"${df['amount'].sum():,.2f}")
col3.metric("Flagged High Risk", f"{df['is_flagged'].sum()}")
col4.metric("Bank Account Collisions", f"{df['is_bank_collision'].sum()}")

st.markdown("---")

# Filter Section
st.sidebar.header("Filter Options")
risk_filter = st.sidebar.slider("Min Risk Score Threshold", 0.0, 1.0, 0.50, 0.05)

filtered_df = df[df["risk_score"] >= risk_filter].sort_values("risk_score", ascending=False)

st.subheader(f"🚨 High Risk Transactions ({len(filtered_df)} items found)")

# Main Table View
display_cols = [
    "transaction_id", "employee_id", "vendor_id", "amount", 
    "merchant_category", "risk_score", "is_bank_collision", "is_near_approval_threshold"
]

st.dataframe(
    filtered_df[display_cols].style.background_gradient(subset=["risk_score"], cmap="Reds"),
    use_container_width=True
)

st.markdown("---")

# Detailed Case Auditor Section
st.subheader("🔎 Case Investigation & Explainable AI Analyst Notes")

selected_txn_id = st.selectbox("Select Transaction ID to Audit:", filtered_df["transaction_id"].unique())

if selected_txn_id:
    txn_row = df[df["transaction_id"] == selected_txn_id].iloc[0]
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown(f"### Transaction Details: `{txn_row['transaction_id']}`")
        st.write(f"**Employee ID:** {txn_row['employee_id']}")
        st.write(f"**Vendor ID:** {txn_row['vendor_id']}")
        st.write(f"**Amount:** ${txn_row['amount']:,.2f} (Limit: ${txn_row['employee_approval_limit']:,.2f})")
        st.write(f"**Merchant Category:** {txn_row['merchant_category']}")
        st.write(f"**Destination Bank Hash:** `{txn_row['dest_bank_account_hash']}`")
        st.metric("Fraud Risk Score", f"{txn_row['risk_score']:.2%}")

    with col_b:
        st.markdown("### 🤖 Automated Risk Summary")
        reasons = []
        if txn_row["is_bank_collision"] == 1:
            reasons.append("⚠️ **CRITICAL:** Destination bank account matches an internal employee hash (Ghost Vendor Risk).")
        if txn_row["is_near_approval_threshold"] == 1:
            reasons.append(f"⚠️ Transaction amount (${txn_row['amount']}) sits right below manager approval limit.")
        if txn_row["is_off_hours"] == 1:
            reasons.append("⚠️ Transaction was executed during non-standard business hours.")
        if txn_row["emp_txn_count_1h"] > 1:
            reasons.append(f"⚠️ High transaction velocity: {int(txn_row['emp_txn_count_1h'])} purchases in last 1 hour.")
            
        if not reasons:
            reasons.append("✅ No critical automated flags detected.")
            
        for reason in reasons:
            st.write(reason)
            
        st.markdown("---")
        st.markdown("### Analyst Action")
        action = st.radio("Decision:", ["Pending", "Approve Transaction", "Flag for Formal Audit"], horizontal=True)
        if st.button("Submit Audit Log"):
            st.success(f"Audit decision '{action}' recorded for {txn_row['transaction_id']}.")