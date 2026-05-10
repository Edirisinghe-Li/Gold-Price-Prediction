import streamlit as st
import pandas as pd
import numpy as np
import pickle
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURATION & UI ---
st.set_page_config(page_title="Gold Intelligence Dashboard", layout="wide")

st.title("📈 Gold Price Intelligence & Scenario Simulator")
st.markdown("""
    This dashboard combines **Real-time Market Data** with a **Random Forest Regressor** to provide forecasts, risk scenarios, and macro-economic stress testing.
""")

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_model():
    with open('gold_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# --- 2. REAL-TIME DATA FETCHING ---
@st.cache_data(ttl=3600)  # Refreshes every hour
def fetch_market_data():
    tickers = {
        'SPX': '^GSPC',    # S&P 500
        'USO': 'USO',      # Oil Proxy
        'SLV': 'SLV',      # Silver Proxy
        'EURUSD': 'EURUSD=X' # Currency Proxy
    }
    data = yf.download(list(tickers.values()), period="5d", interval="1d")['Close']
    latest = data.iloc[-1]
    return latest

try:
    market_live = fetch_market_data()
except:
    # Fallback values if API fails
    market_live = pd.Series({'SPX': 4500, 'USO': 70, 'SLV': 22, 'EURUSD': 1.08})

# --- 3. SIDEBAR: MACRO SHOCKS & SCENARIOS ---
st.sidebar.header("🕹️ Scenario Control Center")

scenario = st.sidebar.selectbox(
    "Choose Market Scenario",
    ["Baseline (Current)", "Crisis (High Fear)", "Risk-On (Bullish)"]
)

st.sidebar.subheader("Custom Macro Shocks")
vix_shock = st.sidebar.slider("VIX (Volatility) Adjustment", 10.0, 50.0, 18.0)
oil_shock = st.sidebar.slider("Oil Price Shift (%)", -50, 50, 0)
usd_shock = st.sidebar.slider("USD Strength Shift (%)", -10, 10, 0)

# Apply Scenario logic to defaults
if scenario == "Crisis (High Fear)":
    vix_input = 35.0
    spx_input = market_live['SPX'] * 0.90
elif scenario == "Risk-On (Bullish)":
    vix_input = 12.0
    spx_input = market_live['SPX'] * 1.05
else:
    vix_input = vix_shock
    spx_input = market_live['SPX']

# --- 4. PREDICTION LOGIC ---
# Adjusting inputs based on live data + shocks
input_data = np.array([[
    spx_input, 
    vix_input, 
    market_live['USO'] * (1 + oil_shock/100), 
    market_live['SLV'], 
    market_live['EURUSD'] * (1 + usd_shock/100)
]])

# Prediction & Confidence Intervals
prediction = model.predict(input_data)[0]

# Calculate 95% Confidence Interval using individual trees
all_tree_preds = np.array([tree.predict(input_data) for tree in model.estimators_])
lower_ci = np.percentile(all_tree_preds, 2.5)
upper_ci = np.percentile(all_tree_preds, 97.5)

# --- 5. MAIN DASHBOARD UI ---
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Forecasted Gold Price", f"${prediction:.2f}")
with col2:
    st.metric("95% Confidence Range", f"${lower_ci:.0f} - ${upper_ci:.0f}")
with col3:
    st.metric("Current Scenario", scenario)

# --- 6. VISUALIZATIONS ---
st.divider()
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("Forecast Uncertainty (Distribution)")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=all_tree_preds.flatten(), name='Tree Predictions', marker_color='#FFD700'))
    fig.add_vline(x=prediction, line_width=3, line_dash="dash", line_color="red", annotation_text="Final Forecast")
    fig.update_layout(xaxis_title="Gold Price ($)", yaxis_title="Model Confidence")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Feature Importance")
    importances = pd.Series(model.feature_importances_, index=['SPX', 'VIX', 'USO', 'SLV', 'EURUSD'])
    st.bar_chart(importances)

# --- 7. EXPORT DATA ---
st.divider()
results_df = pd.DataFrame({
    'Metric': ['Forecast', 'Lower Bound', 'Upper Bound', 'Scenario', 'Timestamp'],
    'Value': [prediction, lower_ci, upper_ci, scenario, datetime.now()]
})

st.download_button(
    label="📥 Download Forecast Results (CSV)",
    data=results_df.to_csv(index=False),
    file_name='gold_forecast.csv',
    mime='text/csv'
)