# 📈 Gold Price Intelligence Dashboard
### Real-Time Financial Forecasting with Random Forest Regression

This project is a 3rd-year IT academic project developed to predict Gold prices by analyzing inter-market relationships. Unlike traditional static models, this dashboard utilizes a live data pipeline to fetch real-market indicators and perform real-time inference.

---

## 🚀 Key Innovations
- **Serverless Architecture:** The project is "stateless"—it fetches historical data for training and live data for prediction directly via the **Yahoo Finance API**, eliminating the need for local CSV storage.
- **Uncertainty Quantification:** Uses the variance between individual trees in the **Random Forest Regressor** to calculate and visualize a 95% Confidence Interval (Prediction Intervals).
- **Macro-Economic Scenario Engine:** Allows users to perform "What-If" analysis by simulating market shocks (e.g., Financial Crisis, Bull Markets) to see how the model reacts to volatility.

## 🛠️ The "Inter-Market" Features
The model doesn't just look at Gold; it analyzes five critical market drivers:
1. **SPX (S&P 500):** General market health.
2. **VIX (Volatility Index):** The market's "Fear Gauge."
3. **USO (United States Oil Fund):** Commodity price trends.
4. **SLV (iShares Silver Trust):** Precious metal correlation.
5. **EUR/USD:** Strength of the US Dollar against the Euro.

## 📦 Tech Stack
- **Language:** Python 3.10+
- **Machine Learning:** Scikit-Learn (Random Forest)
- **Dashboard:** Streamlit
- **Visualization:** Plotly (Interactive Charts)
- **Data Source:** `yfinance` (Yahoo Finance API)

## 🔧 Installation & Local Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/Sonal-Github1/Gold-Price-Prediction.git](https://github.com/Sonal-Github1/Gold-Price-Prediction.git)