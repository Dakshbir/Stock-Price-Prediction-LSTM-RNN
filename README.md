# Stock Price Forecasting with RNN & LSTM Models

Accurately forecasting stock prices empowers investors, traders, and analysts to navigate market dynamics with confidence. In this project, we employ **Recurrent Neural Networks (RNNs)** and **Long Short-Term Memory (LSTM)** architectures to model and predict daily closing prices for Apple Inc. (AAPL) using historical market data and technical indicators.

---

## Table of Contents

1. [Business Context](#business-context)
2. [Key Benefits](#key-benefits)
3. [Challenges & Limitations](#challenges--limitations)
4. [Project Goals](#project-goals)
5. [Data Acquisition & Features](#data-acquisition--features)
6. [Technical Indicators](#technical-indicators)
7. [Architecture & Code Structure](#architecture--code-structure)
8. [Setup & Execution](#setup--execution)
9. [Model Development](#model-development)
10. [Evaluation Metrics](#evaluation-metrics)
11. [Results & Analysis](#results--analysis)
12. [Future Work](#future-work)
13. [Contributing](#contributing)
14. [Contact](#contact)

---

## Business Context

Financial markets are highly dynamic. Reliable forecasts of stock prices help:

* **Investors** choose optimal entry and exit points.
* **Traders** implement algorithmic strategies that respond to predicted short-term movements.
* **Risk Managers** quantify and mitigate potential losses.

By leveraging deep learning models that capture temporal patterns, this project demonstrates how advanced sequence modeling can improve decision-making in finance.

---

## Key Benefits

* **Data-Driven Decisions**: Replaces intuition-based trading with quantitative signals.
* **Automation**: Streamlines forecasting pipelines for continuous monitoring.
* **Scalability**: Modular architecture supports multiple tickers and feature sets.

---

## Challenges & Limitations

1. **Market Volatility**: Sudden events (earnings releases, geopolitical news) introduce noise.
2. **Overfitting**: Models may memorize historical trends that don’t generalize to unseen data.
3. **Feature Selection**: Irrelevant indicators can degrade performance.
4. **Latency**: Real-time forecasting demands efficient inference.

This project addresses these issues via cross-validation, regularization, and careful indicator engineering.

---

## Project Goals

1. **Build** and **tune** RNN and LSTM models for sequence forecasting.
2. **Compare** performance using error and directional accuracy metrics.
3. **Enhance** predictions by incorporating technical indicators (moving averages, RSI, MACD).
4. **Document** an extensible pipeline for future enhancements.

---

## Data Acquisition & Features

* **Source**: Retrieved daily OHLCV data for AAPL from Yahoo Finance (`yfinance` API).
* **Time Period**: January 1, 2010 – June 30, 2025.
* **Primary Features**:

  * **Open, High, Low, Close, Volume**
  * **Adjusted Close** for corporate actions.
* **Data Cleaning**:

  * Forward-fill missing values.
  * Remove outliers using interquartile range filtering.

---

## Technical Indicators

Calculated via `pandas-ta`, including:

| Indicator                                    | Description                    | Parameters                    |
| -------------------------------------------- | ------------------------------ | ----------------------------- |
| Simple Moving Avg                            | Trend smoothing                | Windows: 10, 20, 50           |
| Exponential MA                               | Reacts faster to price changes | Spans: 12, 26                 |
| Relative Strength                            | Momentum oscillator            | Period: 14                    |
| Moving Average Convergence Divergence (MACD) | Trend-following momentum       | Fast: 12, Slow: 26, Signal: 9 |
| Bollinger Bands                              | Volatility measure             | Window: 20, Multiplier: 2     |

These indicators augment raw price inputs to help the network learn market signals.

---

## Architecture & Code Structure

```bash
stock-forecasting/
├── ml_pipeline/               # Core modules
│   ├── data_preparation.py    # Data loader, cleaning, feature engineering
│   ├── model_training.py      # Model definitions (RNN, LSTM), training loops
│   ├── evaluation.py          # Metric calculations and visualizations
│   └── engine.py              # Command-line interface to run stages
├── output/
│   ├── models/                # Serialized model weights (.h5)
│   ├── figures/               # Loss and prediction plots
│   └── reports/               # CSV of evaluation metrics
├── requirements.txt           # Pin exact package versions
├── .env.example               # Template for environment variables (e.g., API keys)
└── README.md                  # Project documentation
```

---

## Setup & Execution

1. **Clone the repository**:

   ```bash
   git clone <repository_url>
   cd stock-forecasting
   ```

2. **Configure environment**:

   ```bash
   cp .env.example .env
   # Set any required environment variables in .env
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the full pipeline**:

   ```bash
   python ml_pipeline/engine.py --start 2010-01-01 --end 2025-06-30 --ticker AAPL
   ```

5. **View results**:

   * Trained model files in `output/models/`.
   * Forecast vs. actual plots in `output/figures/`.
   * Summary of metrics in `output/reports/evaluation.csv`.

---

## Model Development

1. **Data Preparation**:

   * Normalize features with MinMaxScaler.
   * Generate sliding windows (lookback: 60 days).

2. **RNN Architecture**:

   ```python
   model = Sequential([
       SimpleRNN(50, return_sequences=True, input_shape=(lookback, n_features)),
       Dropout(0.2),
       SimpleRNN(30),
       Dense(1)
   ])
   ```

3. **LSTM Architecture**:

   ```python
   model = Sequential([
       LSTM(50, return_sequences=True, input_shape=(lookback, n_features)),
       Dropout(0.2),
       LSTM(30),
       Dense(1)
   ])
   ```

4. **Training**:

   * Loss: Mean Squared Error
   * Optimizer: Adam (learning rate=0.001)
   * Batch size: 32
   * Epochs: 100 with EarlyStopping (patience=10)

---

## Evaluation Metrics

* **Mean Absolute Error (MAE)**
* **Root Mean Squared Error (RMSE)**
* **Directional Accuracy**: percentage of days where predicted and actual price move in the same direction.

Visual comparisons and metric tables are generated in the `evaluation` module.

---

## Results & Analysis

* **RNN vs. LSTM**: LSTM outperformed RNN by \~5% lower RMSE on test data.
* **Indicator Impact**: Including MACD and RSI reduced MAE by \~3% compared to raw prices only.
* **Prediction Horizon**: 1-day ahead forecasts achieved \~60% directional accuracy.

Refer to `output/figures/` for loss curves and forecast plots.

---

## Future Work

* Experiment with **Bidirectional LSTMs** and **GRU** layers.
* Incorporate **sentiment analysis** from financial news.
* Deploy model as a real-time API using FastAPI or Flask.

---

## Contributing

Contributions, issues, and feature requests are welcome. To get started:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/<feature-name>`.
3. Commit changes: `git commit -m "Add <feature>"`.
4. Push to your branch: `git push origin feature/<feature-name>`.
5. Open a pull request.

---

## Contact

**Dakshbir Singh Kapoor**
✉️ [dakshbirkapoor@gmail.com](mailto:dakshbirkapoor@gmail.com)
GitHub: [Dakshbir](https://github.com/Dakshbir)

*Powered by Python, TensorFlow, and the open-source community.*
