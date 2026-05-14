# Gold Price Prediction Dashboard

A machine learning project that predicts gold price using market indicators and serves results in an interactive Streamlit dashboard.Also this is a 3rd year project for the subject Artificial Neural Network (COU5300) in the BSc in Information Technology degree program at The Open University of Sri Lanka.

## Project Overview

This project uses a Random Forest regression model trained on historical Yahoo Finance data. It combines multiple financial indicators to estimate gold price and shows:

- Real-time forecast using current market data
- Forecast uncertainty range using tree-level prediction distribution
- Feature importance chart
- Scenario simulation (baseline, crisis, bullish)
- CSV export for forecast results

## Repository Structure

- app.py: Streamlit web app for live prediction and visualization
- Gold_Price_Predictor.ipynb: end-to-end model training notebook
- gold_model.pkl: serialized trained model used by app.py
- check_torch.py: environment check script for PyTorch setup

## Tech Stack

- Python 3.10+
- scikit-learn
- pandas
- numpy
- yfinance
- streamlit
- plotly
- matplotlib (used in notebook evaluation plots)

## Prerequisites

- Python 3.10 or newer
- pip
- Internet connection (required for Yahoo Finance API calls)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Sonal-Github1/Gold-Price-Prediction.git
cd Gold-Price-Prediction
```

### 2. Create and Activate Virtual Environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install streamlit pandas numpy scikit-learn yfinance plotly matplotlib
```

Optional (only if you want to run check_torch.py):

```bash
pip install torch
```

## Configuration

No manual config file is required.

The application uses:

- Model file path: gold_model.pkl in project root
- Live market source: Yahoo Finance via yfinance
- Cached live data TTL in app.py: 3600 seconds
- Input features expected by model (fixed order): SPX, VIX, USO, SLV, EURUSD

If you retrain the model, keep the same feature order to avoid inference mismatch.

## How to Run

### Run the Dashboard

```bash
streamlit run app.py
```

Then open the URL printed in terminal (usually http://localhost:8501).

### Train or Retrain the Model

Open and run all cells in Gold_Price_Predictor.ipynb. This generates or replaces:

- gold_model.pkl

You can run the notebook in VS Code or Jupyter Lab.

### Optional Environment Check

```bash
python check_torch.py
```

## Model Details

- Algorithm: RandomForestRegressor
- Training target: GLD (gold futures proxy from GC=F close)
- Features:
  - SPX
  - VIX
  - USO
  - SLV
  - EURUSD
- Split: train_test_split with test_size=0.2 and random_state=42
- Reported metrics in notebook:
  - R-squared
  - Mean Absolute Error

## Dashboard Usage

Use the sidebar controls to:

- Select market scenario
- Manually adjust VIX
- Apply oil and USD shocks

The app computes:

- Point estimate for gold price
- 95% interval from individual tree predictions
- Feature importance (global, model-based)

## Troubleshooting

1. FileNotFoundError for gold_model.pkl
   - Cause: model file is missing.
   - Fix: run all cells in Gold_Price_Predictor.ipynb to generate gold_model.pkl.

2. Live data fetch error from yfinance
   - Cause: network/API issue or temporary market data failure.
   - Behavior: app falls back to static baseline values.
   - Fix: verify internet connection and retry.

3. Streamlit command not found
   - Cause: dependency not installed in current environment.
   - Fix: activate your environment and run pip install streamlit.

4. Feature mismatch after retraining
   - Cause: changed column order or feature names during training.
   - Fix: ensure training and inference both use SPX, VIX, USO, SLV, EURUSD in the same order.

## Notes

- This project is intended for academic and educational use.
- Forecasts are model-based estimates and not financial advice.

## License

No license file is currently included. Add a LICENSE file if you plan to distribute this project publicly.