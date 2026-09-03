import streamlit as st
import pandas as pd
import sqlite3
import sys
import os
from streamlit_autorefresh import st_autorefresh

# Ensure absolute imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import DATA_DIR
from src.services.llm_explainer import LLMExplainer

st.set_page_config(page_title="TraceGate | Live Audit", layout="wide", page_icon="🛡️")

# Refresh the dashboard automatically every 30 seconds to catch simulated transactions
st_autorefresh(interval=30000, limit=None, key="live_feed_refresh")

st.title("🛡️ TraceGate: Live ML Audit Monitor")
st.markdown("Real-time monitoring of incoming mechanical repair invoices scored dynamically against our Machine Learning model.")

def load_live_data():
    db_path = DATA_DIR / "tracegate.db"
    if not db_path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    # Fetch the most recent 100 transactions
    df = pd.read_sql_query("SELECT * FROM transactions ORDER BY timestamp DESC LIMIT 100", conn)
    conn.close()
    return df

df = load_live_data()

if df.empty:
    st.warning("⚠️ Live Database is empty. Please run `python pipeline.py` first to initialize.")
    st.stop()

# Evaluate anomalies in the current live feed slice
anomalies = df[df["status"] == "ANOMALY"]

if not anomalies.empty:
    st.error(f"🚨 ACTIVE ALERTS: {len(anomalies)} anomalies detected in the recent transaction stream!")
    
    # Show actionable expanders for the top 5 most recent anomalies
    for _, row in anomalies.head(5).iterrows():
        with st.expander(f"⚠️ Anomaly Alert | Dealership: {row['dealership_id']} | Time: {row['timestamp'][:19]}"):
            cols = st.columns(4)
            cols[0].metric("Billed Cost", f"${row['billed_cost']:.2f}")
            cols[1].metric("ML Expected", f"${row['ml_predicted_cost']:.2f}")
            cols[2].metric("Z-Score", f"{row['z_score']:.2f}")
            cols[3].metric("Engine Temp", f"{row['engine_temp_c']:.2f}°C")
            
            if st.button(f"Why? (Explain this anomaly)", key=f"btn_{row['vin']}"):
                with st.spinner("AI is analyzing the transaction context..."):
                    explainer = LLMExplainer()
                    explanation = explainer.generate_explanation(
                        row['billed_cost'], 
                        row['ml_predicted_cost'], 
                        row['engine_temp_c']
                    )
                    st.info(explanation)
else:
    st.success("✅ All recent transactions look normal. Monitoring live feed...")

st.subheader("Live Transaction Feed")
st.dataframe(df, use_container_width=True)
