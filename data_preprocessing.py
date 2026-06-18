import pandas as pd
import numpy as np
import sqlite3
import os
import joblib
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

CSV_PATH    = r"C:\Users\lahar\Downloads\fraud_detection_cleaned.csv"
DB_PATH     = "fraud_data.db"
MODEL_DIR   = "models"
RANDOM_STATE = 42

N_LEGIT_SAMPLE = 20000

os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [
    "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "hour", "is_weekend",
    "sender_balance_change", "receiver_balance_change",
    "sender_utilization_ratio", "receiver_growth_ratio",
    "is_large_transaction",
    "sender_transaction_count", "receiver_transaction_count",
    "account_drained", "new_receiver_account",
    "high_risk_type", "type_encoded"
]

TARGET_COL = "isFraud"


def load_and_sample():
    print("Loading CSV (this may take a moment)...")
    df = pd.read_csv(CSV_PATH)
    print(f"  Full dataset: {df.shape}")

    fraud_df = df[df[TARGET_COL] == 1]
    legit_df = df[df[TARGET_COL] == 0].sample(
        n=min(N_LEGIT_SAMPLE, (df[TARGET_COL] == 0).sum()),
        random_state=RANDOM_STATE
    )

    sampled = pd.concat([fraud_df, legit_df], ignore_index=True)
    sampled = sampled.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"  Sampled dataset: {sampled.shape}  "
          f"(Fraud: {fraud_df.shape[0]}, Legit: {legit_df.shape[0]})")

    # Convert boolean columns to int
    for col in sampled.columns:
        if sampled[col].dtype == bool:
            sampled[col] = sampled[col].astype(int)

    return sampled


def preprocess(df):
    X = df[FEATURE_COLS].fillna(0).values
    y = df[TARGET_COL].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

    return X_scaled, y


def apply_smote(X, y):
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X, y)
    print(f"  Before SMOTE -> Legit: {(y==0).sum()}, Fraud: {(y==1).sum()}")
    print(f"  After  SMOTE -> Legit: {(y_res==0).sum()}, Fraud: {(y_res==1).sum()}")
    return X_res, y_res


def save_to_db(df):
    conn = sqlite3.connect(DB_PATH)
    cols_to_save = FEATURE_COLS + [TARGET_COL, "type", "step", "day"]
    cols_to_save = [c for c in cols_to_save if c in df.columns]
    df[cols_to_save].to_sql("transactions", conn, if_exists="replace", index=False)

    fraud_col = df[TARGET_COL]
    summary = pd.DataFrame({
        "metric": ["total_transactions", "fraud_count", "legit_count", "fraud_rate_pct"],
        "value": [
            len(df),
            int(fraud_col.sum()),
            int((fraud_col == 0).sum()),
            round(float(fraud_col.mean()) * 100, 2)
        ]
    })
    summary.to_sql("summary_stats", conn, if_exists="replace", index=False)
    conn.close()
    print(f"  Saved {len(df)} rows to {DB_PATH}")


if __name__ == "__main__":
    df = load_and_sample()

    print("Saving to SQLite...")
    save_to_db(df)

    print("Preprocessing & scaling...")
    X, y = preprocess(df)

    print("Applying SMOTE...")
    X_bal, y_bal = apply_smote(X, y)

    np.save(f"{MODEL_DIR}/X_balanced.npy", X_bal)
    np.save(f"{MODEL_DIR}/y_balanced.npy", y_bal)
    np.save(f"{MODEL_DIR}/X_raw.npy", X)
    np.save(f"{MODEL_DIR}/y_raw.npy", y)
    joblib.dump(FEATURE_COLS, f"{MODEL_DIR}/feature_cols.pkl")

    print("Preprocessing complete.")