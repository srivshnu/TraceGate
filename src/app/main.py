import streamlit as st
import pandas as pd
import io
import sys
import os
import pickle

# Ensure the root project directory is discoverable for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import PROCESSED_DATA_DIR, DATA_DIR
from src.services.llm_explainer import LLMExplainer

# Page Configuration
st.set_page_config(page_title="TraceGate | ML Audit", layout="wide", page_icon="🛡️")
st.title("🛡️ TraceGate: ML Audit Copilot")

@st.cache_data
def load_anomalies():
    """Loads the pre-calculated anomaly dataset."""
    path = PROCESSED_DATA_DIR / "flagged_anomalies.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)

# Initialize application dependencies
df = load_anomalies()

# Guard block
if df is None:
    st.error("⚠️ Anomaly data not found! Please run the backend pipeline first (`python pipeline.py`).")
    st.stop()

# --- Tab Layout ---
tab1, tab2 = st.tabs(["📊 Audit Dashboard", "🧪 Live ML Interaction Mode"])

with tab1:
    llm_service = LLMExplainer()
    
    # --- Sidebar Filters ---
    st.sidebar.header("Audit Filters")
    dealer = st.sidebar.selectbox("Select Flagged Dealership", df["dealership_id"].unique())
    filtered_df = df[df["dealership_id"] == dealer].copy()

    # --- Main Dashboard ---
    st.subheader(f"Flagged Claims for {dealer}")
    st.dataframe(filtered_df[["vin", "billed_cost", "ml_predicted_cost", "z_score", "engine_temp_c"]], use_container_width=True)

    col1, col2 = st.columns(2)

    # --- Feature 1: LLM Explainer ---
    with col1:
        st.subheader("🤖 Generate Audit Explanation")
        if st.button("Translate ML Result to Plain English", type="primary"):
            with st.spinner("LLM is analyzing the discrepancy context..."):
                sample = filtered_df.iloc[0]
                explanation = llm_service.generate_explanation(
                    billed_cost=sample['billed_cost'],
                    predicted_cost=sample['ml_predicted_cost'],
                    engine_temp=sample['engine_temp_c']
                )
                st.info(explanation)

    # --- Feature 2: Excel Export ---
    with col2:
        st.subheader("📊 Export for Field Audit")
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            filtered_df.to_excel(writer, index=False, sheet_name='ML_Anomalies')
            
        st.download_button(
            label="📥 Download MS Excel Audit Pack",
            data=buffer.getvalue(),
            file_name=f"TraceGate_Audit_{dealer}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

with tab2:
    st.header("🧪 Interact Directly with the Random Forest Model")
    st.markdown("Adjust the vehicle telemetry metrics to see how the Machine Learning model dynamically predicts expected costs and identifies fraud in real-time.")
    
    try:
        # Load the persisted ML model
        with open(DATA_DIR / "models" / "rf_model.pkl", "rb") as f:
            rf_model = pickle.load(f)
        with open(DATA_DIR / "models" / "std_dev.pkl", "rb") as f:
            std_dev = pickle.load(f)
            
        c1, c2, c3 = st.columns(3)
        with c1:
            live_temp = st.slider("Engine Temp (°C)", 60.0, 130.0, 90.0)
        with c2:
            live_rpm = st.slider("RPM Variance", 50.0, 400.0, 200.0)
        with c3:
            live_billed = st.number_input("Mechanic's Billed Cost ($)", min_value=0.0, value=250.0, step=50.0)
            
        if st.button("Run ML Prediction", type="primary", key="live_predict"):
            # Execute prediction
            input_features = pd.DataFrame([[live_temp, live_rpm]], columns=["engine_temp_c", "rpm_variance"])
            predicted = rf_model.predict(input_features)[0]
            discrepancy = live_billed - predicted
            z_score = discrepancy / std_dev
            
            st.markdown("### ML Output")
            m1, m2, m3 = st.columns(3)
            m1.metric("Expected Cost (ML)", f"${predicted:.2f}")
            m2.metric("Discrepancy (Overbill)", f"${discrepancy:.2f}")
            m3.metric("Z-Score (Deviation)", f"{z_score:.2f}")
            
            if z_score > 2.5:
                st.error("🚨 **ML VERDICT: ANOMALY DETECTED** - High likelihood of structural fraud.")
            else:
                st.success("✅ **ML VERDICT: NORMAL** - Invoice aligns closely with sensor data.")
                
    except FileNotFoundError:
        st.warning("⚠️ Live ML Mode requires the trained model. Run the backend pipeline first (`python pipeline.py`).")
