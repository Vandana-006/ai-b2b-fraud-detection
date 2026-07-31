import pandas as pd
import numpy as np
import joblib
import shap

def generate_fraud_explanations(
    model_path="data/fraud_model.pkl", 
    data_path="data/processed_features.csv"
):
    print("🔍 Loading trained model and processed dataset for Explainable AI (XAI)...")
    
    # 1. Load saved model artifact and dataset
    model_artifact = joblib.load(model_path)
    model = model_artifact["model"]
    features = model_artifact["features"]
    
    df = pd.read_csv(data_path)
    X = df[features]
    
    # 2. Initialize SHAP Explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Handle LightGBM binary classification SHAP array format
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    # 3. Predict probabilities on full dataset
    df["fraud_probability"] = model.predict_proba(X)[:, 1]
    df["is_flagged"] = (df["fraud_probability"] >= 0.5).astype(int)
    
    # Filter high-risk flagged transactions
    flagged_txns = df[df["is_flagged"] == 1].copy()
    print(f"🚨 Total Transactions Flagged for Human Review: {len(flagged_txns)}")
    
    # 4. Generate Natural Language Summaries for Top Flagged Cases
    summaries = []
    
    for idx, row in flagged_txns.head(5).iterrows():
        # Get top SHAP feature contributions for this specific transaction
        row_shap = shap_vals[idx]
        top_feature_indices = np.argsort(np.abs(row_shap))[::-1][:3]
        
        reason_strings = []
        for feat_idx in top_feature_indices:
            feat_name = features[feat_idx]
            feat_val = row[feat_name]
            
            if feat_name == "is_bank_collision" and feat_val == 1:
                reason_strings.append("Destination bank account matches employee entity (Ghost Vendor Risk)")
            elif feat_name == "is_near_approval_threshold" and feat_val == 1:
                reason_strings.append(f"Transaction amount (${row['amount']:.2f}) sits just under approval threshold")
            elif feat_name == "is_off_hours" and feat_val == 1:
                reason_strings.append("Transaction occurred during non-business off-hours")
            elif feat_name == "emp_txn_count_1h" and feat_val > 1:
                reason_strings.append(f"High employee velocity ({int(feat_val)} transactions within 1 hour)")
            else:
                reason_strings.append(f"Anomalous metric for {feat_name} (Value: {feat_val})")
                
        summary_text = (
            f"TXN ID: {row['transaction_id']} | Risk Score: {row['fraud_probability']:.2%}\n"
            f"   Primary Risk Drivers:\n"
            f"   - " + "\n   - ".join(reason_strings)
        )
        summaries.append(summary_text)
        
    print("\n--- Sample AI Analyst Explanation Reports ---\n")
    for summary in summaries:
        print(summary)
        print("-" * 50)

if __name__ == "__main__":
    generate_fraud_explanations()