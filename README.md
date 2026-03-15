# Stock Price Forecasting with RNN & LSTM

An end-to-end deep learning pipeline that forecasts stock prices for any ticker using Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) architectures, enriched with technical indicators. Includes an interactive Streamlit web app for live predictions.

**Live Demo**: [Streamlit App](https://dakshbir-stock-price-prediction-lstm-rnn.streamlit.app)

---

## Table of Contents

1. [Overview](#overview)
2. [Tech Stack](#tech-stack)
3. [Project Structure](#project-structure)
4. [Models](#models)
5. [Technical Indicators](#technical-indicators)
6. [Setup & Usage](#setup--usage)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Results](#results)
9. [Future Work](#future-work)
10. [Contact](#contact)

---

## Overview

Financial time series are among the most challenging prediction targets due to noise, volatility, and non-stationarity. This project builds three sequential deep learning models — SimpleRNN, LSTM, and Multivariate LSTM — trained on historical OHLCV data augmented with technical indicators. The pipeline supports any stock ticker via Yahoo Finance and exposes predictions through both a CLI tool and an interactive web frontend.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Acquisition | `yfinance`, Yahoo Finance v8 Chart API |
| Data Processing | `pandas`, `numpy`, `scikit-learn` (MinMaxScaler) |
| Deep Learning | `TensorFlow`, `Keras` (SimpleRNN, LSTM, Dense, Dropout) |
| Visualization | `plotly`, `matplotlib`, `seaborn` |
| Frontend | `Streamlit` |
| Version Control | Git, GitHub |

---

## Project Structure

```
Stock-Price-Prediction-LSTM-main/
├── app.py                    # Streamlit web frontend
├── engine.py                 # CLI pipeline runner
├── ml_pipeline/
│   ├── __init__.py
│   ├── train.py              # RNN, LSTM, Multivariate LSTM training & plotting
│   └── utils.py              # Data splitting, sequence generation, metrics, indicators
├── lib/
│   ├── images/               # Architecture diagrams (RNN, LSTM, MLP)
│   └── lstm_dakshbir.ipynb   # Research & experimentation notebook
├── .streamlit/
│   └── config.toml           # Dark theme configuration
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Models

### 1. SimpleRNN
Baseline recurrent model. Captures short-term temporal dependencies using vanilla recurrent units.

```python
model = Sequential([
    SimpleRNN(50, return_sequences=True, input_shape=(60, 1)),
    Dropout(0.2),
    SimpleRNN(30),
    Dense(1)
])
```

### 2. LSTM
Addresses the vanishing gradient problem using forget, input, and output gates — better at capturing long-range patterns.

```python
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60, 1)),
    Dropout(0.2),
    LSTM(30),
    Dense(1)
])
```

### 3. Multivariate LSTM
Extends the LSTM to accept multiple input features — Close price plus RSI, MACD, EMA, and Bollinger Bands — enabling the model to learn from market signals beyond raw price.

**Training config (all models):**
- Loss: Mean Squared Error
- Optimizer: Adam (lr=0.001)
- Lookback window: 60 days
- EarlyStopping (patience=10) + ReduceLROnPlateau
- Train/test split: year-based (most recent year held out for testing)

---

## Technical Indicators

Computed from scratch using `pandas` and `numpy` in `ml_pipeline/utils.py`:

| Indicator | Description | Parameters |
|---|---|---|
| SMA | Simple Moving Average | Windows: 10, 20, 50 |
| EMA | Exponential Moving Average | Spans: 12, 26 |
| RSI | Relative Strength Index (momentum) | Period: 14 |
| MACD | Moving Average Convergence Divergence | Fast: 12, Slow: 26, Signal: 9 |
| Bollinger Bands | Volatility bands around SMA | Window: 20, Std: 2 |

---

## Setup & Usage

### 1. Clone & install

```bash
git clone https://github.com/Dakshbir/Stock-Price-Prediction-LSTM-RNN.git
cd Stock-Price-Prediction-LSTM-RNN
pip install -r requirements.txt
```

### 2. Run the Streamlit app (recommended)

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Enter any ticker, select a date range, and click **Fetch Data** → **Train LSTM**.

### 3. Run the CLI pipeline

```bash
python engine.py --ticker AAPL --start 2018-01-01 --end 2024-01-01
```

This trains all three models sequentially and saves prediction plots to `output/figures/`.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| RMSE | Root Mean Squared Error — penalises large errors |
| MAE | Mean Absolute Error — average prediction deviation |
| MAPE | Mean Absolute Percentage Error — scale-independent |
| R² | Coefficient of determination — variance explained |
| Directional Accuracy | % of days where predicted and actual price move in the same direction |

---

## Results

Tested on AAPL (2018–2024, ~1,500 trading days):

- **LSTM vs RNN**: LSTM achieved ~5% lower RMSE than the baseline SimpleRNN
- **Multivariate LSTM**: Adding RSI, MACD, and Bollinger Bands as features improved directional accuracy
- **Directional Accuracy**: ~60% on held-out test data
- **Training**: EarlyStopping typically triggered between epochs 15–30, preventing overfitting

---

## Future Work

- Bidirectional LSTM and GRU variants
- Attention mechanism over the 60-day window
- Sentiment analysis from financial news as additional features
- Multi-step ahead forecasting (5-day, 30-day horizons)

---

## Contact

**Dakshbir Singh Kapoor**
[dakshbirkapoor@gmail.com](mailto:dakshbirkapoor@gmail.com) · [GitHub](https://github.com/Dakshbir)
