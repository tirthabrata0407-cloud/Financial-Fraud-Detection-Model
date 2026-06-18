import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
import json
import os
import plotly.express as px
import plotly.graph_objects as go

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── AUTO-SETUP: Generate DB & models if missing ───────────────────────────────
DB_PATH_CHECK    = "fraud_data.db"
MODEL_DIR_CHECK  = "models"

if not os.path.exists(DB_PATH_CHECK) or not os.path.exists(f"{MODEL_DIR_CHECK}/best_model.pkl"):
    with st.spinner("First-time setup: generating data and training models (1-2 min)..."):
        import subprocess
        subprocess.run(["python", "data_preprocessing.py"], check=True)
        subprocess.run(["python", "train_model.py"], check=True)
    st.rerun()

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }

    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1d2e, #252840);
        border: 1px solid #2e3250;
        border-radius: 12px;
        padding: 16px;
    }
    div[data-testid="metric-container"] label { color: #8b8fa8; font-size: 0.82rem; }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #e2e4f0; font-size: 1.7rem; font-weight: 700;
    }

    .section-title {
        font-size: 1.1rem; font-weight: 600; color: #a8b4e8;
        border-left: 3px solid #5c6bc0; padding-left: 10px;
        margin: 16px 0 12px 0;
    }

    .alert-fraud {
        background: rgba(239,83,80,0.15); border: 1px solid #ef5350;
        border-radius: 10px; padding: 14px 18px; color: #ef9a9a;
    }
    .alert-safe {
        background: rgba(102,187,106,0.15); border: 1px solid #66bb6a;
        border-radius: 10px; padding: 14px 18px; color: #a5d6a7;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1a2332; }
    [data-testid="stSidebar"] * { color: #e2e4f0 !important; }
    [data-testid="stSidebar"] label { color: #e2e4f0 !important; }

    .stTabs [data-baseweb="tab-list"] { background-color: #1a1d2e; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { color: #8b8fa8; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #a8b4e8; }
</style>
""", unsafe_allow_html=True)

DB_PATH   = "fraud_data.db"
MODEL_DIR = "models"

# ─── HELPERS ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_transactions():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM transactions", conn)
    conn.close()
    return df

@st.cache_data
def load_metrics():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM model_metrics", conn)
    conn.close()
    return df

@st.cache_resource
def load_model_artifacts():
    model    = joblib.load(f"{MODEL_DIR}/best_model.pkl")
    scaler   = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    features = joblib.load(f"{MODEL_DIR}/feature_cols.pkl")
    try:
        name = joblib.load(f"{MODEL_DIR}/best_model_name.pkl")
    except:
        name = "Best Model"
    return model, scaler, features, name

def check_setup():
    return os.path.exists(DB_PATH) and os.path.exists(f"{MODEL_DIR}/best_model.pkl")

# ─── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ FraudGuard")
    st.markdown("*Financial Fraud Detection System*")
    st.divider()

    if not check_setup():
        st.warning("⚠️ Models not trained yet!")
        st.markdown("""
        **Run these first:**
```bash
        python data_preprocessing.py
        python train_model.py
```
        """)
        st.stop()

    page = st.radio("Navigation", [
        "📊 Overview Dashboard",
        "🔍 Transaction Analysis",
        "🤖 Model Performance",
        "🎯 Predict Transaction"
    ])
    st.divider()
    df = load_transactions()
    fraud_count = df["isFraud"].sum()
    st.metric("Total Transactions", f"{len(df):,}")
    st.metric("Fraud Cases", f"{fraud_count:,}", delta=f"{fraud_count/len(df)*100:.1f}%", delta_color="inverse")

# ─── PAGE 1: OVERVIEW DASHBOARD ────────────────────────────────────────────────
if page == "📊 Overview Dashboard":
    st.title("📊 Fraud Overview Dashboard")

    df = load_transactions()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", f"{len(df):,}")
    with col2:
        st.metric("Fraud Cases", f"{df['isFraud'].sum():,}")
    with col3:
        st.metric("Legitimate", f"{(df['isFraud']==0).sum():,}")
    with col4:
        st.metric("Fraud Rate", f"{df['isFraud'].mean()*100:.2f}%")

    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">Transaction Distribution</div>', unsafe_allow_html=True)
        fig = px.pie(
            values=[df["isFraud"].sum(), (df["isFraud"]==0).sum()],
            names=["Fraud", "Legitimate"],
            color_discrete_sequence=["#ef5350", "#5c6bc0"],
            hole=0.5
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e4f0", height=320)
        st.plotly_chart(fig, width='stretch', key="overview_pie")

    with col_r:
        st.markdown('<div class="section-title">Fraud by Transaction Hour</div>', unsafe_allow_html=True)
        hourly = df.groupby(["hour","isFraud"]).size().reset_index(name="count")
        fig = px.bar(hourly, x="hour", y="count", color="isFraud",
                     color_discrete_map={0:"#5c6bc0", 1:"#ef5350"},
                     labels={"isFraud":"Type", "hour":"Hour", "count":"Count"},
                     barmode="stack")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e4f0", height=320,
                          legend=dict(title="", itemsizing="constant"))
        fig.for_each_trace(lambda t: t.update(name="Fraud" if t.name=="1" else "Legitimate"))
        st.plotly_chart(fig, width='stretch', key="overview_hourly")

    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.markdown('<div class="section-title">Transaction Amount Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x="amount", color="isFraud",
                           color_discrete_map={0:"#5c6bc0", 1:"#ef5350"},
                           nbins=60, barmode="overlay", opacity=0.7,
                           labels={"isFraud":"Type","amount":"Amount"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e4f0", height=300)
        fig.for_each_trace(lambda t: t.update(name="Fraud" if t.name=="1" else "Legitimate"))
        st.plotly_chart(fig, width='stretch', key="overview_amount_dist")

    with col_r2:
        st.markdown('<div class="section-title">Fraud Rate by Transaction Type</div>', unsafe_allow_html=True)
        cat_stats = df.groupby("type")["isFraud"].agg(["sum","count"])
        cat_stats["fraud_rate"] = (cat_stats["sum"] / cat_stats["count"] * 100).round(2)
        cat_stats = cat_stats.reset_index().sort_values("fraud_rate", ascending=True)
        fig = px.bar(cat_stats, x="fraud_rate", y="type",
                     orientation="h", color="fraud_rate",
                     color_continuous_scale=["#5c6bc0","#ef5350"],
                     labels={"fraud_rate":"Fraud Rate (%)","type":"Transaction Type"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e4f0", height=300, showlegend=False,
                          coloraxis_showscale=False)
        st.plotly_chart(fig, width='stretch', key="overview_category_fraud")


# ─── PAGE 2: TRANSACTION ANALYSIS ──────────────────────────────────────────────
elif page == "🔍 Transaction Analysis":
    st.title("🔍 Transaction Analysis")
    df = load_transactions()

    col1, col2, col3 = st.columns(3)
    with col1:
        label_filter = st.selectbox("Transaction Type Filter", ["All","Fraud Only","Legitimate Only"])
    with col2:
        cat_filter = st.multiselect("Transaction Type",
                                     df["type"].unique().tolist(),
                                     default=df["type"].unique().tolist())
    with col3:
        amount_range = st.slider("Amount Range", float(df["amount"].min()), float(df["amount"].max()),
                                  (float(df["amount"].min()), float(df["amount"].max())))

    filtered = df[df["type"].isin(cat_filter)]
    filtered = filtered[(filtered["amount"] >= amount_range[0]) &
                        (filtered["amount"] <= amount_range[1])]
    if label_filter == "Fraud Only":
        filtered = filtered[filtered["isFraud"] == 1]
    elif label_filter == "Legitimate Only":
        filtered = filtered[filtered["isFraud"] == 0]

    st.markdown(f'<div class="section-title">Showing {len(filtered):,} transactions</div>',
                unsafe_allow_html=True)

    if len(filtered) > 0:
        fig = px.scatter(filtered.sample(min(2000, len(filtered))),
                         x="sender_transaction_count", y="amount",
                         color="isFraud", color_discrete_map={0:"#5c6bc0", 1:"#ef5350"},
                         opacity=0.6,
                         labels={"sender_transaction_count":"Sender Transaction Count","amount":"Amount","isFraud":"Type"},
                         title="Amount vs Sender Transaction Count")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e4f0")
        fig.for_each_trace(lambda t: t.update(name="Fraud" if t.name=="1" else "Legitimate"))
        st.plotly_chart(fig, width='stretch', key="analysis_scatter")

    st.markdown('<div class="section-title">Feature Comparison: Fraud vs Legitimate</div>',
                unsafe_allow_html=True)
    num_cols = ["amount","sender_transaction_count","receiver_transaction_count",
                "is_large_transaction","account_drained","new_receiver_account","high_risk_type"]
    comp = filtered.groupby("isFraud")[num_cols].mean()
    comp = comp.reindex([0, 1])
    comp = comp.T.reset_index()
    comp.columns = ["Feature","Legitimate","Fraud"]
    comp["Legitimate"] = comp["Legitimate"].fillna(0)
    comp["Fraud"] = comp["Fraud"].fillna(0)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Legitimate", x=comp["Feature"], y=comp["Legitimate"],
                          marker_color="#5c6bc0"))
    fig.add_trace(go.Bar(name="Fraud", x=comp["Feature"], y=comp["Fraud"],
                          marker_color="#ef5350"))
    fig.update_layout(barmode="group", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e4f0",
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, width='stretch', key="analysis_feature_comparison")

    with st.expander("📋 View Raw Data"):
        show_cols = ["amount","type","hour",
                     "sender_transaction_count","account_drained","isFraud"]
        st.dataframe(
            filtered[show_cols].head(500).rename(columns={"isFraud":"Fraud?"}),
            width='stretch'
        )


# ─── PAGE 3: MODEL PERFORMANCE ─────────────────────────────────────────────────
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance Comparison")

    try:
        metrics_df = load_metrics()
        _, _, _, best_name = load_model_artifacts()
    except:
        st.error("Please run `train_model.py` first.")
        st.stop()

    st.markdown(f"**Best Model:** `{best_name}` (highest F1 Score)")
    st.divider()

    st.markdown('<div class="section-title">All Models — Key Metrics (%)</div>', unsafe_allow_html=True)
    display_cols = ["model_name","accuracy","precision","recall","f1_score","roc_auc"]
    styled = metrics_df[display_cols].copy()
    st.dataframe(styled, width='stretch', hide_index=True)

    st.markdown('<div class="section-title">Model Comparison Radar</div>', unsafe_allow_html=True)
    categories = ["Accuracy","Precision","Recall","F1 Score","ROC AUC"]
    fig = go.Figure()
    colors = ["#5c6bc0","#66bb6a","#ffa726"]
    for i, row in metrics_df.iterrows():
        vals = [row["accuracy"], row["precision"], row["recall"], row["f1_score"], row["roc_auc"]]
        vals += [vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=categories + [categories[0]],
            fill="toself", name=row["model_name"],
            line_color=colors[i % len(colors)], opacity=0.7
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[50, 100],
                                    gridcolor="#2e3250", color="#8b8fa8"),
                   angularaxis=dict(gridcolor="#2e3250", color="#8b8fa8"),
                   bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e4f0",
        legend=dict(bgcolor="rgba(0,0,0,0)"), height=420
    )
    st.plotly_chart(fig, width='stretch', key="performance_radar")

    st.markdown('<div class="section-title">Metric-by-Metric Breakdown</div>', unsafe_allow_html=True)
    melted = metrics_df.melt(id_vars="model_name",
                              value_vars=["accuracy","precision","recall","f1_score","roc_auc"],
                              var_name="Metric", value_name="Score")
    fig = px.bar(melted, x="Metric", y="Score", color="model_name",
                 barmode="group", color_discrete_sequence=["#5c6bc0","#66bb6a","#ffa726"],
                 labels={"Score":"Score (%)","model_name":"Model"})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#e2e4f0", yaxis_range=[50, 105],
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig, width='stretch', key="performance_grouped_bar")

    st.markdown('<div class="section-title">Confusion Matrices</div>', unsafe_allow_html=True)
    cms_cols = st.columns(len(metrics_df))
    for i, row in metrics_df.iterrows():
        cm = json.loads(row["confusion_matrix"])
        with cms_cols[i]:
            st.markdown(f"**{row['model_name']}**")
            fig = px.imshow(cm, text_auto=True, color_continuous_scale=["#1a1d2e","#5c6bc0"],
                            labels=dict(x="Predicted", y="Actual"),
                            x=["Legit","Fraud"], y=["Legit","Fraud"])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e4f0",
                               height=260, margin=dict(t=20,b=20,l=20,r=20),
                               coloraxis_showscale=False)
            st.plotly_chart(fig, width='stretch', key=f"cm_{row['model_name']}")


# ─── PAGE 4: PREDICT TRANSACTION ───────────────────────────────────────────────
elif page == "🎯 Predict Transaction":
    st.title("🎯 Predict a Transaction")
    st.markdown("Fill in the transaction details below to check if it's potentially fraudulent.")

    try:
        model, scaler, features, best_name = load_model_artifacts()
    except:
        st.error("Please run the preprocessing and training scripts first.")
        st.stop()

    st.markdown(f"*Using model: **{best_name}***")
    st.divider()

    TYPE_MAP = {"PAYMENT": 3, "TRANSFER": 4, "CASH_OUT": 1, "CASH_IN": 0, "DEBIT": 2}

    col1, col2, col3 = st.columns(3)
    with col1:
        amount          = st.number_input("Transaction Amount", value=5000.0, step=100.0)
        old_bal_org     = st.number_input("Sender Old Balance", value=10000.0, step=100.0)
        new_bal_org     = st.number_input("Sender New Balance", value=5000.0, step=100.0)
        old_bal_dest    = st.number_input("Receiver Old Balance", value=0.0, step=100.0)
        new_bal_dest    = st.number_input("Receiver New Balance", value=5000.0, step=100.0)
    with col2:
        hour            = st.slider("Transaction Hour (0-23)", 0, 23, 14)
        is_weekend      = st.checkbox("Weekend Transaction?")
        txn_type        = st.selectbox("Transaction Type", list(TYPE_MAP.keys()))
        sender_txn_cnt  = st.number_input("Sender Transaction Count", min_value=0, value=5, step=1)
        receiver_txn_cnt= st.number_input("Receiver Transaction Count", min_value=0, value=2, step=1)
    with col3:
        is_large_txn    = st.checkbox("Large Transaction?")
        account_drained = st.checkbox("Account Drained (balance -> 0)?")
        new_receiver    = st.checkbox("New Receiver Account?")
        high_risk_type  = st.checkbox("High-Risk Type (Transfer/Cash-Out)?")

    st.divider()

    if st.button("🔍 Analyze Transaction", width='stretch'):
        sender_balance_change   = new_bal_org - old_bal_org
        receiver_balance_change = new_bal_dest - old_bal_dest
        sender_utilization      = (amount / old_bal_org) if old_bal_org > 0 else 0
        receiver_growth         = (receiver_balance_change / old_bal_dest) if old_bal_dest > 0 else 0
        type_encoded            = TYPE_MAP[txn_type]

        raw = np.array([[
            amount, old_bal_org, new_bal_org, old_bal_dest, new_bal_dest,
            hour, int(is_weekend),
            sender_balance_change, receiver_balance_change,
            sender_utilization, receiver_growth,
            int(is_large_txn),
            sender_txn_cnt, receiver_txn_cnt,
            int(account_drained), int(new_receiver),
            int(high_risk_type), type_encoded
        ]])

        scaled = scaler.transform(raw)
        pred   = model.predict(scaled)[0]
        proba  = model.predict_proba(scaled)[0]

        fraud_prob = proba[1] * 100
        legit_prob = proba[0] * 100

        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            if pred == 1:
                st.markdown(f"""
                <div class="alert-fraud">
                    <h3>⚠️ FRAUD DETECTED</h3>
                    <p>Fraud Probability: <strong>{fraud_prob:.1f}%</strong></p>
                    <p>This transaction shows high-risk patterns. Recommend manual review.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="alert-safe">
                    <h3>✅ TRANSACTION SAFE</h3>
                    <p>Legitimate Probability: <strong>{legit_prob:.1f}%</strong></p>
                    <p>No suspicious patterns detected.</p>
                </div>""", unsafe_allow_html=True)

        with col_res2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fraud_prob,
                title={"text": "Fraud Risk Score (%)", "font": {"color": "#e2e4f0"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8b8fa8"},
                    "bar":  {"color": "#ef5350" if pred == 1 else "#5c6bc0"},
                    "steps": [
                        {"range": [0, 30],  "color": "rgba(92,107,192,0.2)"},
                        {"range": [30, 70], "color": "rgba(255,167,38,0.2)"},
                        {"range": [70,100], "color": "rgba(239,83,80,0.2)"}
                    ],
                    "threshold": {"line": {"color": "#ffa726","width": 3}, "value": 50}
                },
                number={"suffix": "%", "font": {"color": "#e2e4f0"}}
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e4f0", height=300)
            st.plotly_chart(fig, width='stretch', key="prediction_gauge")

        st.markdown('<div class="section-title">Risk Factor Summary</div>', unsafe_allow_html=True)
        risks = []
        if account_drained:  risks.append("🔴 Account fully drained (balance → 0)")
        if high_risk_type:   risks.append("🔴 High-risk transaction type (Transfer/Cash-Out)")
        if new_receiver:     risks.append("🟠 First-time / new receiver account")
        if is_large_txn:     risks.append("🟠 Large transaction amount")
        if sender_utilization > 0.9: risks.append("🟠 Sender used >90% of available balance")
        if hour < 6:         risks.append("🟡 Transaction at odd hours (midnight-6am)")
        if sender_txn_cnt < 2: risks.append("🟡 Low sender transaction history")

        if risks:
            for msg in risks:
                st.markdown(f"- {msg}")
        else:
            st.markdown("✅ No specific risk factors identified.")