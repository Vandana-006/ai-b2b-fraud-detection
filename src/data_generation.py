import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()
np.random.seed(42)
random.seed(42)

def generate_b2b_fraud_dataset(num_records=10000, output_path="data/raw_b2b_transactions.csv"):
    print("🚀 Generating B2B financial transaction dataset...")
    
    # 1. Generate entities
    vendor_ids = [f"VEND_{1000+i}" for i in range(100)]
    employee_ids = [f"EMP_{5000+i}" for i in range(200)]
    
    vendor_bank_hashes = [f"BANK_HASH_{fake.hexify(text='^^^^^^^^')}" for _ in range(100)]
    employee_bank_hashes = [f"BANK_HASH_{fake.hexify(text='^^^^^^^^')}" for _ in range(200)]
    
    vendor_bank_map = dict(zip(vendor_ids, vendor_bank_hashes))
    approval_limits = {emp: random.choice([5000, 10000, 25000]) for emp in employee_ids}
    
    mcc_codes = {
        "OFFICE_SUPPLIES": 5111,
        "IT_HARDWARE": 5732,
        "TRAVEL_FLIGHTS": 4511,
        "LUXURY_RETAIL": 5944,
        "CONSULTING_SERVICES": 7392
    }
    
    base_time = datetime(2026, 1, 1, 8, 0, 0)
    data = []
    
    for i in range(num_records):
        txn_id = f"TXN_{100000 + i}"
        emp_id = random.choice(employee_ids)
        vendor_id = random.choice(vendor_ids)
        mcc = random.choice(list(mcc_codes.keys()))
        
        timestamp = base_time + timedelta(
            days=random.randint(0, 90),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        amount = round(float(np.random.exponential(scale=1500) + 10), 2)
        bank_acc = vendor_bank_map[vendor_id]
        device_ip = fake.ipv4()
        is_fraud = 0
        fraud_type = "LEGITIMATE"
        
        # Inject fraud patterns (~1.5% fraud rate)
        fraud_roll = random.random()
        
        # Fraud Pattern 1: Split Transaction (Approval Bypass)
        if fraud_roll < 0.006:
            limit = approval_limits[emp_id]
            amount = round(limit - random.uniform(50, 300), 2)
            is_fraud = 1
            fraud_type = "SPLIT_APPROVAL_BYPASS"
            
        # Fraud Pattern 2: Ghost Vendor (Bank Collision)
        elif fraud_roll < 0.011:
            emp_index = employee_ids.index(emp_id)
            bank_acc = employee_bank_hashes[emp_index]
            amount = round(random.uniform(3000, 15000), 2)
            is_fraud = 1
            fraud_type = "GHOST_VENDOR_SELF_DEALING"
            
        # Fraud Pattern 3: Off-Hours Luxury Misuse
        elif fraud_roll < 0.015:
            mcc = "LUXURY_RETAIL"
            amount = round(random.uniform(2000, 8000), 2)
            timestamp = timestamp.replace(hour=random.choice([1, 2, 3]))
            is_fraud = 1
            fraud_type = "MCC_OFFHOURS_MISUSE"

        data.append({
            "transaction_id": txn_id,
            "timestamp": timestamp,
            "employee_id": emp_id,
            "vendor_id": vendor_id,
            "merchant_category": mcc,
            "mcc_code": mcc_codes[mcc],
            "amount": float(amount),
            "employee_approval_limit": approval_limits[emp_id],
            "dest_bank_account_hash": bank_acc,
            "device_ip": device_ip,
            "is_fraud": is_fraud,
            "fraud_type": fraud_type
        })
        
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"✅ Successfully saved dataset to {output_path}!\n")
    print("Fraud Class Breakdown:")
    print(df['fraud_type'].value_counts())
    return df

if __name__ == "__main__":
    generate_b2b_fraud_dataset()