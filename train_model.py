import numpy as np
import pandas as pd
import sqlite3
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
)
from xgboost import XGBClassifier

MODEL_DIR  = "models"
DB_PATH    = "fraud_data.db"
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    X = np.load(f"{MODEL_DIR}/X_balanced.npy")
    y = np.load(f"{MODEL_DIR}/y_balanced.npy")
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


def get_models():
    return {
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=1, use_label_encoder=False,
            eval_metric="logloss", random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            random_state=RANDOM_STATE
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        )
    }


def evaluate(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred)  * 100, 2),
        "precision": round(precision_score(y_test, y_pred) * 100, 2),
        "recall":    round(recall_score(y_test, y_pred)    * 100, 2),
        "f1_score":  round(f1_score(y_test, y_pred)        * 100, 2),
        "roc_auc":   round(roc_auc_score(y_test, y_proba)  * 100, 2),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred)
    }


def save_metrics(results: dict):
    rows = []
    for model_name, metrics in results.items():
        rows.append({
            "model_name": model_name,
            "accuracy":   metrics["accuracy"],
            "precision":  metrics["precision"],
            "recall":     metrics["recall"],
            "f1_score":   metrics["f1_score"],
            "roc_auc":    metrics["roc_auc"],
            "confusion_matrix": json.dumps(metrics["confusion_matrix"])
        })
    df = pd.DataFrame(rows)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("model_metrics", conn, if_exists="replace", index=False)
    conn.close()
    print("  Model metrics saved to DB.")


if __name__ == "__main__":
    print("Loading balanced data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"  Train size: {len(X_train)} | Test size: {len(X_test)}")

    # Initialize and fit scaler on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler and feature columns for app.py
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    print("  Scaler saved to disk.")
    
    # Feature names (18 features based on app.py prediction logic)
    feature_cols = [
        "amount", "old_bal_org", "new_bal_org", "old_bal_dest", "new_bal_dest",
        "hour", "is_weekend",
        "sender_balance_change", "receiver_balance_change",
        "sender_utilization", "receiver_growth",
        "is_large_transaction",
        "sender_transaction_count", "receiver_transaction_count",
        "account_drained", "new_receiver_account",
        "high_risk_type", "type"
    ]
    joblib.dump(feature_cols, f"{MODEL_DIR}/feature_cols.pkl")
    print("  Feature columns saved to disk.")

    models  = get_models()
    results = {}
    best_model_name, best_f1, best_model = None, 0, None

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        metrics = evaluate(model, X_test_scaled, y_test)
        results[name] = metrics

        print(f"  Accuracy : {metrics['accuracy']}%")
        print(f"  Precision: {metrics['precision']}%")
        print(f"  Recall   : {metrics['recall']}%")
        print(f"  F1 Score : {metrics['f1_score']}%")
        print(f"  ROC-AUC  : {metrics['roc_auc']}%")
        print(metrics["classification_report"])

        joblib.dump(model, f"{MODEL_DIR}/{name.replace(' ','_').lower()}.pkl")

        if metrics["f1_score"] > best_f1:
            best_f1, best_model_name, best_model = metrics["f1_score"], name, model

    print(f"\nBest Model: {best_model_name} (F1: {best_f1}%)")
    joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")
    joblib.dump(best_model_name, f"{MODEL_DIR}/best_model_name.pkl")

    save_metrics(results)
    print("\nTraining complete. All models saved to /models/")
