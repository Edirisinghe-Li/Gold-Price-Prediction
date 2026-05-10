import streamlit as st
import pandas as pd
import numpy as np
import pickle
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# --- 1. CONFIGURATION & UI OPTIMIZATION ---
st.set_page_config(page_title="Gold Intelligence", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for "Single Screen" fit
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.7rem;
        }
        .stPlotlyChart {
            height: 350px;
        }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MODEL & DATA CORE ---
@st.cache_resource
def load_model():
    with open('gold_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data(ttl=3600)
def fetch_market_data():
    ticker_map = {
        '^GSPC': 'SPX',
        '^VIX': 'VIX',
        'USO': 'USO',
        'SLV': 'SLV',
        'EURUSD=X': 'EURUSD'
    }
    # Fetch data and ensure column names are clean
    data = yf.download(list(ticker_map.keys()), period="5d", interval="1d")['Close']
    
    # Handle potential MultiIndex from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(1)
        
    data = data.rename(columns=ticker_map)
    return data.iloc[-1]

# Load Resources
model = load_model()
try:
    market_live = fetch_market_data()
except Exception as e:
    st.error(f"Live Data Connection Failed. Using static baseline. Error: {e}")
    market_live = pd.Series({'SPX': 5100.0, 'VIX': 15.0, 'USO': 75.0, 'SLV': 24.0, 'EURUSD': 1.08})

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.header("🕹️ Scenario Control Center")

scenario = st.sidebar.selectbox(
    "Market Context",
    ["Baseline (Current)", "Crisis (High Fear)", "Risk-On (Bullish)"]
)

st.sidebar.divider()
st.sidebar.subheader("Manual Macro Shocks")
vix_manual = st.sidebar.slider("Volatility (VIX)", 10.0, 50.0, float(market_live['VIX']))
oil_shock = st.sidebar.slider("Oil Price Shift (%)", -50, 50, 0)
usd_shock = st.sidebar.slider("USD Strength Shift (%)", -10, 10, 0)

# Logic for inputs based on Scenario vs Manual
if scenario == "Crisis (High Fear)":
    vix_input = 38.0
    spx_input = market_live['SPX'] * 0.90
elif scenario == "Risk-On (Bullish)":
    vix_input = 12.0
    spx_input = market_live['SPX'] * 1.05
else:
    vix_input = vix_manual
    spx_input = market_live['SPX']

# --- 4. PREDICTION ENGINE ---
# Prepare features in exact order used during training
input_features = np.array([[
    spx_input, 
    vix_input, 
    market_live['USO'] * (1 + oil_shock/100), 
    market_live['SLV'], 
    market_live['EURUSD'] * (1 + usd_shock/100)
]])

prediction = model.predict(input_features)[0]

# Calculate Confidence using Forest Variance
all_tree_preds = np.array([tree.predict(input_features) for tree in model.estimators_])
lower_ci = np.percentile(all_tree_preds, 2.5)
upper_ci = np.percentile(all_tree_preds, 97.5)

# --- 5. DASHBOARD LAYOUT ---
st.title("📈 Gold Price Intelligence Dashboard")

# Top Metric Row
with st.container(border=True):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted Gold", f"${prediction:.2f}")
    m2.metric("95% Range", f"${lower_ci:.0f} - ${upper_ci:.0f}")
    m3.metric("S&P 500 (Adj)", f"{spx_input:.0f}")
    m4.metric("Current Scenario", scenario)

st.divider()

# Main Visuals Row
c1, c2 = st.columns([2, 1])

with c1:
    with st.container(border=True):
        st.subheader("Forecast Uncertainty Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=all_tree_preds.flatten(), 
            name='Tree Votes', 
            marker_color='#FFD700',
            nbinsx=25
        ))
        fig.add_vline(x=prediction, line_width=3, line_dash="dash", line_color="red")
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            height=350,
            xaxis_title="Price ($)",
            yaxis_title="Model Confidence"
        )
        st.plotly_chart(fig, use_container_width=True)

with c2:
    with st.container(border=True):
        st.subheader("Feature Impact")
        feat_names = ['SPX', 'VIX', 'USO', 'SLV', 'EURUSD']
        importances = pd.Series(model.feature_importances_, index=feat_names).sort_values()
        
        fig_imp = go.Figure(go.Bar(
            x=importances.values,
            y=importances.index,
            orientation='h',
            marker_color='#1f77b4'
        ))
        fig_imp.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            height=350
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# Footer Utilities
st.divider()
u1, u2 = st.columns(2)
with u1:
    results_df = pd.DataFrame({
        'Metric': ['Forecast', 'Lower Bound', 'Upper Bound', 'Scenario', 'Timestamp'],
        'Value': [prediction, lower_ci, upper_ci, scenario, datetime.now().strftime("%Y-%m-%d %H:%M")]
    })
    st.download_button(
        label="📥 Download Forecast CSV",
        data=results_df.to_csv(index=False),
        file_name='gold_forecast.csv',
        mime='text/csv'
    )
with u2:
    st.caption("Developed for 3rd Year IT Project | Data: Yahoo Finance | Model: Random Forest")