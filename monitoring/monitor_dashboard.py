import streamlit as st
import pandas as pd
import json
import os
import time
import yaml
import altair as alt
from datetime import datetime

# ==========================================
# 1. Configuration & Setup
# ==========================================
st.set_page_config(
    page_title="AI Vision System Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

try:
    with open(CONFIG_PATH, "r") as f:
        CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    # Fallback default if config is missing
    CONFIG = {"log_file": "../logs/metrics.json", "dashboard": {"port": 8501}}

LOG_FILE = os.path.abspath(os.path.join(BASE_DIR, CONFIG.get("log_file", "../logs/metrics.json")))

# ==========================================
# 2. Custom CSS
# ==========================================
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        border: 1px solid #dce0e6;
        padding: 15px;
        border-radius: 8px;
    }
    @media (prefers-color-scheme: dark) {
        div[data-testid="stMetric"] {
            background-color: #262730;
            border: 1px solid #333;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. Sidebar Controls (MOVED HERE TO FIX ERROR)
# ==========================================
st.sidebar.title("⚙ Control Panel")
refresh_rate = st.sidebar.slider("Refresh Rate (sec)", 1, 10, 2)
filter_mode = st.sidebar.radio("Log Filter:", ["All Data", "Low Confidence Only (< 50%)"])
auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=True)

st.sidebar.divider()
st.sidebar.info("v3.0 - OCR & Voice Enabled")

# ==========================================
# 4. Main Dashboard Logic
# ==========================================
placeholder = st.empty()

def load_data():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()
    try:
        with open(LOG_FILE, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception:
        return pd.DataFrame()

while True:
    df = load_data()
    
    with placeholder.container():
        st.title("👁 AI Vision Operations Center")
        
        if df.empty:
            st.warning("Waiting for data stream...")
        else:
            # KPIs
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Requests", len(df))
            
            avg_conf = df['confidence'].mean() if 'confidence' in df.columns else 0
            k2.metric("Avg Confidence", f"{avg_conf:.1%}")
            
            top_obj = df['label'].mode()[0] if 'label' in df.columns and not df['label'].empty else "N/A"
            k3.metric("Top Detected", str(top_obj).title())
            
            last_mode = df['model'].iloc[-1] if 'model' in df.columns else "N/A"
            k4.metric("Current Mode", last_mode)

            # Charts
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Confidence Trend")
                if 'timestamp' in df.columns and 'confidence' in df.columns:
                    st.line_chart(df[['timestamp', 'confidence']].tail(50).set_index('timestamp'), height=250)
            
            with c2:
                st.subheader("Detection Distribution")
                if 'label' in df.columns:
                    st.bar_chart(df['label'].value_counts().head(10), height=250)

            # Logs Table
            st.subheader("Live Logs")
            view_df = df.copy()
            if filter_mode == "Low Confidence Only (< 50%)":
                view_df = view_df[view_df['confidence'] < 0.5]
            
            st.dataframe(
                view_df.sort_values(by="timestamp", ascending=False).head(10)[['timestamp', 'model', 'label', 'confidence']],
                use_container_width=True,
                hide_index=True
            )

    if not auto_refresh:
        break
    time.sleep(refresh_rate)