import streamlit as st
import pandas as pd
import io
import sys
import os

# Ensure the root project directory is discoverable for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import PROCESSED_DATA_DIR
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
llm_service = LLMExplainer()

# Guard block: require ML pipeline to have run
if df is None:
    st.error("⚠️ Anomaly data not found! Please run the backend pipeline first (`python pipeline.py`).")
    st.stop()

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
    
    # Process Excel export into memory buffer
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        filtered_df.to_excel(writer, index=False, sheet_name='ML_Anomalies')
        
    st.download_button(
        label="📥 Download MS Excel Audit Pack",
        data=buffer.getvalue(),
        file_name=f"TraceGate_Audit_{dealer}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
