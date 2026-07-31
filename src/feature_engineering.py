import pandas as pd
import numpy as np
import networkx as nx

def build_graph_features(df):
    print("🕸️ Building Graph Network Features...")
    
    # 1. Create a Bipartite Graph linking Employees, Vendors, and Bank Accounts
    G = nx.Graph()
    
    for _, row in df.iterrows():
        emp = row['employee_id']
        vendor = row['vendor_id']
        bank = row['dest_bank_account_hash']
        
        G.add_edge(emp, bank, relation="pays_to")
        G.add_edge(vendor, bank, relation="owned_by")
    
    # 2. Check for Bank Account Collisions (Employee linked to same bank account as a vendor)
    # Get set of all employee IDs
    employee_set = set(df['employee_id'].unique())
    
    collision_flags = []
    for _, row in df.iterrows():
        emp = row['employee_id']
        bank = row['dest_bank_account_hash']
        
        # Find all nodes connected to this bank account
        connected_nodes = set(G.neighbors(bank)) - {bank}
        
        # If another employee or same employee is directly tied to the destination bank account node
        if emp in connected_nodes and len(connected_nodes) > 1:
            collision_flags.append(1)
        else:
            collision_flags.append(0)
            
    df['is_bank_collision'] = collision_flags
    return df

def build_velocity_and_ratio_features(df):
    print("⏱️ Calculating Velocity and Amount Ratios...")
    
    # Ensure sorted by timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 1. Approval Threshold Ratio
    df['amount_to_limit_ratio'] = df['amount'] / df['employee_approval_limit']
    
    # 2. Threshold Proximity Flag (Amounts within 5% below approval threshold)
    df['is_near_approval_threshold'] = (
        (df['amount_to_limit_ratio'] >= 0.90) & (df['amount_to_limit_ratio'] < 1.00)
    ).astype(int)
    
    # 3. Off-Hours Feature (11 PM to 5 AM)
    df['is_off_hours'] = df['timestamp'].dt.hour.isin([23, 0, 1, 2, 3, 4, 5]).astype(int)
    
    # 4. Rolling Transaction Velocity (1 Hour Window per Employee)
    df = df.set_index('timestamp')
    
    # Calculate rolling counts
    df['emp_txn_count_1h'] = (
        df.groupby('employee_id')['transaction_id']
        .transform(lambda x: x.rolling('1h').count())
    )
    
    df['emp_spent_24h'] = (
        df.groupby('employee_id')['amount']
        .transform(lambda x: x.rolling('24h').sum())
    )
    
    df = df.reset_index()
    return df

def engineer_features(input_path="data/raw_b2b_transactions.csv", output_path="data/processed_features.csv"):
    df = pd.read_csv(input_path)
    
    # Run Feature Engineering
    df = build_graph_features(df)
    df = build_velocity_and_ratio_features(df)
    
    # Save processed features dataset
    df.to_csv(output_path, index=False)
    print(f"✅ Feature Engineering Complete! Processed dataset saved to '{output_path}'.")
    print(f"Total Features Created: {df.shape[1]}")
    return df

if __name__ == "__main__":
    engineer_features()