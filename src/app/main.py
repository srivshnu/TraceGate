import streamlit as st
import pandas as pd
import sqlite3
from streamlit_autorefresh import st_autorefresh
from src.config import DATA_DIR
from src.services.llm_explainer import LLMExplainer

# ─────────────────────────────────────────────
#  PAGE CONFIG & THEME
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TraceGate | Audit Intelligence",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded",
)

# Auto-refresh every 30 seconds to pick up new simulator transactions
st_autorefresh(interval=30_000, limit=None, key="live_feed_refresh")

# ─────────────────────────────────────────────
#  CUSTOM CSS — Professional Card / KPI Layer
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Header bar */
    .header-bar {
        background: linear-gradient(90deg, #0f172a 0%, #1e3a5f 100%);
        padding: 1.4rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border-left: 5px solid #00D4FF;
    }
    .header-bar h1 { color: #00D4FF; margin: 0; font-size: 1.8rem; letter-spacing: 1px; }
    .header-bar p  { color: #94a3b8; margin: 0.2rem 0 0; font-size: 0.9rem; }

    /* KPI card */
    .kpi-card {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 1.1rem 1.4rem;
        text-align: center;
    }
    .kpi-label { color: #64748b; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }
    .kpi-value { color: #f1f5f9; font-size: 2rem; font-weight: 700; margin: 0.2rem 0; }
    .kpi-value.danger { color: #f87171; }
    .kpi-value.warning { color: #fbbf24; }
    .kpi-value.success { color: #34d399; }
    .kpi-sub   { color: #475569; font-size: 0.75rem; }

    /* Alert card */
    .alert-card {
        background: linear-gradient(135deg, #1c0a0a 0%, #1a0d0d 100%);
        border: 1px solid #7f1d1d;
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 1rem 1.4rem;
        margin-bottom: 0.7rem;
    }
    .alert-card h4 { color: #fca5a5; margin: 0 0 0.5rem; font-size: 0.9rem; }
    .alert-meta { color: #6b7280; font-size: 0.78rem; }

    /* Section header */
    .section-label {
        color: #64748b;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin: 1.5rem 0 0.6rem;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 0.4rem;
    }

    /* Sidebar branding */
    .brand-title { color: #00D4FF; font-size: 1.3rem; font-weight: 800; letter-spacing: 2px; }
    .brand-sub   { color: #475569; font-size: 0.75rem; }
    .status-dot  { display: inline-block; width:9px; height:9px; border-radius:50%; margin-right:6px; }
    .dot-live    { background: #34d399; animation: pulse 2s infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

    /* Scrollable feed table */
    [data-testid="stDataFrame"] { border-radius: 8px; border: 1px solid #1e293b; }
    div[data-testid="stDataFrame"] > div { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="brand-title">🛡️ TRACEGATE</p>', unsafe_allow_html=True)
    st.markdown('<p class="brand-sub">Audit Intelligence Platform v1.0</p>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<span class="status-dot dot-live"></span><span style="color:#34d399;font-size:0.82rem;">Live Monitor Active</span>', unsafe_allow_html=True)
    st.caption("Page refreshes every 30s automatically.")
    st.divider()
    st.markdown("**Filters**")
    z_threshold = st.slider("Z-Score Alert Threshold", 1.0, 5.0, 2.5, 0.1,
                             help="Transactions with a Z-Score above this value are flagged as anomalies.")
    st.divider()
    st.markdown("**System**")
    st.caption(f"DB: `{DATA_DIR / 'tracegate.db'}`")

# ─────────────────────────────────────────────
#  SINGLETON — LLM SERVICE
# ─────────────────────────────────────────────
@st.cache_resource
def get_llm_service():
    try:
        return LLMExplainer()
    except ValueError as e:
        return None, str(e)

# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(ttl=25)
def load_transactions() -> pd.DataFrame:
    db_path = DATA_DIR / "tracegate.db"
    if not db_path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM transactions ORDER BY timestamp DESC LIMIT 20", conn
    )
    conn.close()
    # Format timestamp to human-readable: "03 Sep 2026  11:36 PM"
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%d %b %Y  %I:%M %p")
    return df

df = load_transactions()

if df.empty:
    st.error("⚠️  Database is empty. Run `python pipeline.py` to initialise the system.")
    st.stop()

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
    <h1>🛡️ TRACEGATE — Audit Intelligence</h1>
    <p>Real-time ML-powered invoice anomaly detection &amp; explainability platform</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  KPI ROW
# ─────────────────────────────────────────────
anomalies_df = df[df["z_score"] > z_threshold]
normal_df    = df[df["z_score"] <= z_threshold]
top_offender = (
    anomalies_df.groupby("dealership_id").size().idxmax()
    if not anomalies_df.empty else "—"
)
max_z = anomalies_df["z_score"].max() if not anomalies_df.empty else 0
anomaly_count = len(anomalies_df)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Total Transactions</div>
        <div class="kpi-value">{len(df)}</div>
        <div class="kpi-sub">Last 20 records</div>
    </div>""", unsafe_allow_html=True)
with k2:
    color_class = "danger" if anomaly_count > 0 else "success"
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Active Anomalies</div>
        <div class="kpi-value {color_class}">{anomaly_count}</div>
        <div class="kpi-sub">Z-Score &gt; {z_threshold}</div>
    </div>""", unsafe_allow_html=True)
with k3:
    color_class = "danger" if max_z > 5 else "warning" if max_z > 2.5 else "success"
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Highest Z-Score</div>
        <div class="kpi-value {color_class}">{max_z:.2f}</div>
        <div class="kpi-sub">Deviation from model</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Most Flagged Dealer</div>
        <div class="kpi-value" style="font-size:1.4rem;">{top_offender}</div>
        <div class="kpi-sub">By anomaly frequency</div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  ANOMALY ALERTS
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">⚠️ Anomaly Alerts</div>', unsafe_allow_html=True)

if anomalies_df.empty:
    st.success("✅  All recent transactions are within normal parameters. Monitoring live feed...")
else:
    for _, row in anomalies_df.head(5).iterrows():
        overbill = row["billed_cost"] - row["ml_predicted_cost"]
        overbill_pct = (overbill / row["ml_predicted_cost"]) * 100

        with st.expander(
            f"🚨  {row['dealership_id']}  |  VIN: {row['vin']}  "
            f"|  Overbilled by ${overbill:,.2f} ({overbill_pct:.0f}%)  "
            f"|  {row['timestamp'][:19]}"
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Billed Cost",    f"${row['billed_cost']:,.2f}")
            c2.metric("ML Expected",    f"${row['ml_predicted_cost']:,.2f}",
                      delta=f"-${overbill:,.2f}", delta_color="inverse")
            c3.metric("Z-Score",        f"{row['z_score']:.2f}")
            c4.metric("Engine Temp",    f"{row['engine_temp_c']:.1f} °C")

            if st.button("🔍  Explain This Anomaly", key=f"explain_{row['vin']}",
                         type="primary"):
                llm = get_llm_service()
                if llm is None or isinstance(llm, tuple):
                    err = llm[1] if isinstance(llm, tuple) else "API key not configured."
                    st.error(f"⚠️ {err}")
                else:
                    with st.spinner("Analysing fraud context with AI..."):
                        explanation = llm.generate_explanation(
                            row["billed_cost"],
                            row["ml_predicted_cost"],
                            row["engine_temp_c"],
                        )
                    st.info(explanation)

# ─────────────────────────────────────────────
#  LIVE FEED TABLE
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">📋 Live Transaction Feed</div>', unsafe_allow_html=True)

display_cols = ["timestamp", "dealership_id", "vin", "billed_cost",
                "ml_predicted_cost", "z_score", "engine_temp_c", "status"]
feed_df = df[display_cols].copy()
feed_df.columns = ["Timestamp", "Dealer", "VIN", "Billed ($)",
                   "ML Expected ($)", "Z-Score", "Engine Temp (°C)", "Status"]
feed_df["Billed ($)"]      = feed_df["Billed ($)"].map("${:,.2f}".format)
feed_df["ML Expected ($)"] = feed_df["ML Expected ($)"].map("${:,.2f}".format)
feed_df["Z-Score"]         = feed_df["Z-Score"].map("{:.2f}".format)
feed_df["Engine Temp (°C)"] = feed_df["Engine Temp (°C)"].map("{:.1f}°C".format)

st.dataframe(
    feed_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Status": st.column_config.TextColumn("Status"),
        "Z-Score": st.column_config.TextColumn("Z-Score"),
    },
)
