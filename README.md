# 📈 Transformer-Based Quantitative AI Forecasting Platform

A full-stack AI-powered quantitative trading and forecasting platform built using Transformer architecture, wavelet transforms, macroeconomic indicators, FastAPI, and Streamlit.

This project performs live market forecasting for the Indian stock market using deep learning and advanced quantitative finance techniques.

---

# 🚀 Live Demo

## 🌐 Streamlit Dashboard
[Open Dashboard](https://8cefdk3iis7o8vjprdresk.streamlit.app/)

## ⚡ FastAPI Backend
[Open Backend API](https://sensex-transformer-system-1.onrender.com/)

## 📚 Swagger API Docs
[Open Swagger Docs](https://sensex-transformer-system-1.onrender.com/docs)

---

# 🧠 Project Overview

This platform combines:

- Transformer Neural Networks
- Wavelet Signal Processing
- Macroeconomic Features
- Financial Technical Indicators
- Real-Time Market Data
- Quantitative Backtesting
- Cloud Deployment

to create a realistic end-to-end quantitative AI forecasting system.

---

# ✨ Features

## ✅ Transformer-Based Forecasting
- Deep learning sequence modeling
- Multi-head self-attention
- Time-series prediction architecture
- 60-day rolling market windows

---

## ✅ Advanced Feature Engineering

The model uses:
- SENSEX OHLCV data
- NIFTY Index
- India VIX
- NASDAQ
- Dow Jones
- Crude Oil
- USD/INR Exchange Rate

---

## ✅ Technical Indicators

Includes:
- RSI
- SMA
- Bollinger Bands
- Momentum
- Rolling Volatility

---

## ✅ Wavelet-Based Signal Processing

Uses Haar Wavelet Transform for:
- Trend extraction
- Noise reduction
- Multi-scale financial signal decomposition

---

## ✅ Real-Time Market Inference

The system:
1. Fetches live market data
2. Engineers features automatically
3. Applies wavelet transforms
4. Scales data
5. Generates Transformer predictions
6. Produces trading signals

---

## ✅ Quantitative Backtesting Engine

Includes:
- Strategy simulation
- Equity curve analysis
- Sharpe Ratio
- Max Drawdown
- Win Rate
- Trading Signal Evaluation

---

## ✅ Production Deployment

Deployed using:
- FastAPI
- Streamlit
- Render
- GitHub

---

# 🏗️ System Architecture

```text
User
 ↓
Streamlit Dashboard
 ↓
FastAPI Backend
 ↓
Transformer Model
 ↓
Live Market Data
```

---

# 🧪 Tech Stack

| Category | Technologies |
|---|---|
| Deep Learning | PyTorch |
| Backend | FastAPI |
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy |
| Financial Indicators | ta |
| Wavelets | PyWavelets |
| Visualization | Plotly, Matplotlib |
| Deployment | Render, Streamlit Cloud |
| Market Data | Yahoo Finance |

---

# 📂 Project Structure

```text
sensex-transformer-system/
│
├── app/
│   ├── api/
│   ├── models/
│   ├── schemas/
│   └── services/
│
├── preprocessing/
├── backtesting/
├── models/
│   └── saved/
│
├── data/
│   ├── features/
│   ├── scaler/
│   └── processed/
│
├── dashboard.py
├── requirements.txt
├── Procfile
├── runtime.txt
└── README.md
```

---

# 📊 Model Features

The Transformer model uses 26 engineered features.

## Market Features
- sensex_open
- sensex_high
- sensex_low
- sensex_close
- sensex_volume

---

## Global Markets
- nifty_close
- india_vix_close
- nasdaq_close
- dow_jones_close
- crude_oil_close
- usdinr_close

---

## Return Features
- sensex_close_return
- nifty_close_return
- nasdaq_close_return
- dow_jones_close_return
- crude_oil_close_return
- usdinr_close_return

---

## Technical Indicators
- sensex_rsi
- sensex_sma_20
- bb_high
- bb_low
- rolling_volatility
- momentum_10

---

## Wavelet Features
- wavelet_trend
- wavelet_detail_1
- wavelet_detail_2

---

# ⚙️ Local Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/Harshithpatali/sensex-transformer-system.git
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate virtual environment.

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Running The Project

## Start FastAPI Backend

```bash
uvicorn app.main:app --reload
```

Swagger Docs:

```text
http://127.0.0.1:8000/docs
```

---

## Start Streamlit Dashboard

```bash
streamlit run dashboard.py
```

Dashboard:

```text
http://localhost:8501
```

---

# 📈 API Endpoints

## GET /

Health check endpoint.

---

## POST /predict

Manual prediction using custom features.

---

## POST /live-predict

Fetches live market data and generates:
- predicted returns
- trading signal

Example Response:

```json
{
  "predicted_return": -0.015,
  "signal": "HOLD"
}
```

---

# 📉 Quantitative Metrics

The platform supports:
- Sharpe Ratio
- Max Drawdown
- Equity Curve
- Win Rate
- Strategy Returns
- Trade Statistics

---

# 🧠 Machine Learning Concepts Used

- Transformer Networks
- Self-Attention
- Time-Series Forecasting
- Wavelet Decomposition
- Feature Engineering
- Financial Signal Processing
- Quantitative Backtesting
- Regression Forecasting
- Sequence Modeling

---

# 🌍 Deployment

## Backend Deployment
- Render

## Frontend Deployment
- Streamlit Cloud

---

# 🔮 Future Improvements

- LSTM + Transformer Hybrid
- Reinforcement Learning
- Portfolio Optimization
- Live Trading Integration
- Zerodha API Integration
- Docker + Kubernetes
- Multi-Asset Forecasting
- Sentiment Analysis Integration
- Risk Management System
- Paper Trading Engine

---

# 📌 Disclaimer

This project is for educational and research purposes only.

This platform does NOT provide financial advice or guaranteed trading performance.

---

# 👨‍💻 Author

## Harshith Devraj

- Applied Mathematics & Computing
- AI/ML Engineer
- Quantitative Finance Enthusiast
- Deep Learning Researcher

---

# ⭐ If You Like This Project

Please consider giving the repository a star ⭐
