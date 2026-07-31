import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import os

def train_fraud_model(input_path="data/processed_features.csv", model_output_path="data/fraud_model.pkl"):
    print("🤖 Loading processed dataset...")
    df = pd.read_csv(input_path)
    
    # 1. Define Features and Target
    # We drop raw IDs, raw dates, and categorical strings that aren't encoded
    features = [
        "amount", 
        "employee_approval_limit", 
        "mcc_code",
        "is_bank_collision", 
        "amount_to_limit_ratio", 
        "is_near_approval_threshold",
        "is_off_hours", 
        "emp_txn_count_1h", 
        "emp_spent_24h"
    ]
    
    X = df[features]
    y = df["is_fraud"]
    
    # 2. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)} | Testing set size: {len(X_test)}")
    
    # 3. Calculate Class Weight Ratio for Imbalanced Data
    # scale_pos_weight compensates for fraud being a small percentage
    fraud_count = y_train.sum()
    legit_count = len(y_train) - fraud_count
    ratio = legit_count / fraud_count
    
    print(f"⚖️ Applying Imbalance Weight Ratio: {ratio:.2f}")
    
    # 4. Initialize and Train LightGBM Classifier
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        scale_pos_weight=ratio,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # 5. Evaluate the Model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\n--- Model Evaluation Results ---")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    # 6. Save Model Artifact
    joblib.dump({"model": model, "features": features}, model_output_path)
    print(f"\n✅ Fraud Detection Model saved to '{model_output_path}'!")

if __name__ == "__main__":
    train_fraud_model()