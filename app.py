"""
Stock Price Prediction - Streamlit Frontend
Deployable on Streamlit Community Cloud (free)
"""

import warnings
warnings.filterwarnings("ignore")

import sys
sys.stdout.reconfigure(encoding="utf-8")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import io

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #262d40);
        border: 1px solid #2d3555;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card .label { color: #8b92a5; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card .value { color: #e2e8f0; font-size: 1.6rem; font-weight: 700; margin-top: 4px; }
    .metric-card .delta-up   { color: #22c55e; font-size: 0.85rem; }
    .metric-card .delta-down { color: #ef4444; font-size: 0.85rem; }
    .section-header {
        color: #94a3b8;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 1rem;
        border-bottom: 1px solid #2d3555;
        padding-bottom: 0.5rem;
    }
    div[data-testid="stMetric"] {
        background: #1e2130;
        border: 1px solid #2d3555;
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border: none;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"] { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ────────────────────────────────────────────────────────────────

def _fetch_via_direct_api(ticker: str, start: str, end: str):
    """
    Fetch OHLCV data directly from Yahoo Finance v8 chart API using requests.
    Bypasses yf.download() entirely — not subject to the same rate limits.
    """
    import requests, datetime as dt

    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts   = int(pd.Timestamp(end).timestamp())
    url      = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params   = {"interval": "1d", "period1": start_ts, "period2": end_ts}
    headers  = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    r = requests.get(url, params=params, headers=headers, timeout=15)
    if r.status_code != 200:
        return pd.DataFrame(), f"HTTP {r.status_code}"

    data   = r.json()
    result = data.get("chart", {}).get("result")
    if not result:
        err = data.get("chart", {}).get("error", {})
        return pd.DataFrame(), str(err)

    res        = result[0]
    timestamps = res.get("timestamp", [])
    quote      = res["indicators"]["quote"][0]
    adjclose   = res["indicators"].get("adjclose", [{}])[0].get("adjclose", [None]*len(timestamps))

    df = pd.DataFrame({
        "Open":      quote.get("open"),
        "High":      quote.get("high"),
        "Low":       quote.get("low"),
        "Close":     quote.get("close"),
        "Volume":    quote.get("volume"),
        "Adj Close": adjclose,
    }, index=pd.to_datetime([dt.datetime.fromtimestamp(t) for t in timestamps]))
    df.index      = df.index.normalize()
    df.index.name = "Date"
    df.dropna(inplace=True)
    return df, None


def fetch_data(ticker: str, start: str, end: str):
    """
    Download and clean stock data from Yahoo Finance.
    Returns (DataFrame, error_string). On success error_string is None.
    Tries direct v8 chart API first (not rate-limited), falls back to yf.download().
    Never cached — caller manages session-state caching so failures are never locked in.
    """
    # Primary: direct Yahoo Finance chart API (works even when yf.download is rate-limited)
    df, err = _fetch_via_direct_api(ticker, start, end)
    if err is None and not df.empty:
        return df, None

    # Fallback: yfinance download
    import time
    last_err = err or ""
    for attempt in range(1, 4):
        try:
            df = yf.download(
                ticker, start=start, end=end,
                auto_adjust=False, progress=False, threads=False,
            )
            if df is None or df.empty:
                last_err = "empty"
                if attempt < 3:
                    time.sleep(2 ** attempt)
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.dropna(inplace=True)
            return df, None
        except Exception as e:
            last_err = str(e)
            if "RateLimit" in type(e).__name__ or "Too Many" in last_err:
                last_err = "rate_limit"
            if attempt < 3:
                time.sleep(2 ** attempt)

    return pd.DataFrame(), last_err


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators on a copy of df."""
    d = df.copy()
    close = d["Close"]
    # EMAs
    d["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    d["EMA_26"] = close.ewm(span=26, adjust=False).mean()
    d["EMA_50"] = close.ewm(span=50, adjust=False).mean()
    # MACD
    d["MACD"]   = d["EMA_12"] - d["EMA_26"]
    d["Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_Hist"] = d["MACD"] - d["Signal"]
    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    d["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    # Bollinger Bands
    sma20      = close.rolling(20).mean()
    std20      = close.rolling(20).std()
    d["BB_Mid"]   = sma20
    d["BB_Upper"] = sma20 + 2 * std20
    d["BB_Lower"] = sma20 - 2 * std20
    return d


def build_sequences(arr: np.ndarray, n_steps: int):
    X, y = [], []
    for i in range(len(arr) - n_steps):
        X.append(arr[i : i + n_steps])
        y.append(arr[i + n_steps])
    return np.array(X), np.array(y)


def train_lstm(X_train, y_train, n_steps: int, epochs: int, progress_bar, status_text):
    """Train a lightweight LSTM and stream progress to Streamlit."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import Callback, EarlyStopping

    class StreamlitCallback(Callback):
        def __init__(self, total_epochs):
            self.total = total_epochs
            self.losses = []
        def on_epoch_end(self, epoch, logs=None):
            self.losses.append(logs.get("loss", 0))
            pct = (epoch + 1) / self.total
            progress_bar.progress(pct)
            status_text.text(f"Epoch {epoch+1}/{self.total}  |  loss: {logs.get('loss', 0):.5f}  |  val_loss: {logs.get('val_loss', 0):.5f}")

    tf.random.set_seed(42)
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(n_steps, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    cb = StreamlitCallback(epochs)
    es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.15,
        callbacks=[cb, es],
        verbose=0,
    )
    return model, history, cb.losses


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Stock Predictor")
    st.markdown("---")

    st.markdown('<p class="section-header">Data Settings</p>', unsafe_allow_html=True)
    ticker = st.text_input("Ticker Symbol", "AAPL", help="e.g. AAPL, TSLA, MSFT, GOOGL").strip().upper()
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime(2018, 1, 1))
    with col2:
        end_date = st.date_input("End Date", datetime.now())

    st.markdown("---")
    st.markdown('<p class="section-header">Model Settings</p>', unsafe_allow_html=True)
    epochs  = st.slider("Training Epochs",  10, 80, 30)
    n_steps = st.slider("Lookback Window (days)", 20, 120, 60)
    future_days = st.slider("Forecast Horizon (days)", 5, 60, 30)

    st.markdown("---")
    fetch_btn = st.button("Fetch & Analyze", type="primary", use_container_width=True)
    train_btn = st.button("Train LSTM Model", use_container_width=True)

    st.markdown("---")
    st.caption("Data source: Yahoo Finance")
    st.caption("Model: Stacked LSTM (TensorFlow)")

# ─── Session State ──────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df      = None
if "df_ind" not in st.session_state:
    st.session_state.df_ind  = None
if "model" not in st.session_state:
    st.session_state.model   = None
if "sc" not in st.session_state:
    st.session_state.sc      = None
if "pred_df" not in st.session_state:
    st.session_state.pred_df = None
if "future_df" not in st.session_state:
    st.session_state.future_df = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None

# ─── Fetch Data ─────────────────────────────────────────────────────────────
if fetch_btn:
    with st.spinner(f"Fetching {ticker} data from Yahoo Finance..."):
        df, err = fetch_data(ticker, str(start_date), str(end_date))
    if err == "rate_limit":
        st.warning(
            "**Yahoo Finance rate limit hit.** "
            "This happens after many rapid requests from the same IP. "
            "Please wait **2–3 minutes** and click **Fetch & Analyze** again."
        )
    elif err or df.empty:
        st.error(
            f"Could not fetch data for **{ticker}**. "
            f"Check the ticker symbol and internet connection, then try again. "
            f"{'(Error: ' + err + ')' if err and err != 'empty' else ''}"
        )
    else:
        st.session_state.df     = df
        st.session_state.df_ind = add_indicators(df)
        # Reset model state when new data is loaded
        st.session_state.model     = None
        st.session_state.pred_df   = None
        st.session_state.future_df = None
        st.session_state.metrics   = None
        st.success(f"Loaded **{len(df):,}** trading days for **{ticker}**")

# ─── Header ─────────────────────────────────────────────────────────────────
st.title(f"📈 Stock Price Prediction — {ticker}")

if st.session_state.df is None:
    st.info("Configure settings in the sidebar and click **Fetch & Analyze** to get started.")
    st.stop()

df     = st.session_state.df
df_ind = st.session_state.df_ind

# ─── KPI Row ─────────────────────────────────────────────────────────────────
latest  = float(df["Close"].iloc[-1])
prev    = float(df["Close"].iloc[-2])
chg     = latest - prev
chg_pct = chg / prev * 100
high52  = float(df["High"].rolling(252).max().iloc[-1])
low52   = float(df["Low"].rolling(252).min().iloc[-1])
avg_vol = int(df["Volume"].rolling(20).mean().iloc[-1])

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Close Price",   f"${latest:.2f}",  f"{chg:+.2f} ({chg_pct:+.2f}%)")
k2.metric("52-Week High",  f"${high52:.2f}")
k3.metric("52-Week Low",   f"${low52:.2f}")
k4.metric("Avg Volume (20d)", f"{avg_vol:,}")
k5.metric("Data Points",   f"{len(df):,}")

st.markdown("---")

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Price Chart",
    "🔬 Technical Indicators",
    "🤖 LSTM Prediction",
    "🔮 Future Forecast",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRICE CHART
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    chart_type = st.radio("Chart Type", ["Candlestick", "Line", "OHLC"], horizontal=True)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.75, 0.25],
    )

    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name=ticker, increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        ), row=1, col=1)
    elif chart_type == "Line":
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"],
            name="Close", line=dict(color="#6366f1", width=1.5),
        ), row=1, col=1)
    else:
        fig.add_trace(go.Ohlc(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name=ticker,
        ), row=1, col=1)

    # EMAs overlay
    for ema, col in [("EMA_12", "#f59e0b"), ("EMA_26", "#ec4899"), ("EMA_50", "#06b6d4")]:
        fig.add_trace(go.Scatter(
            x=df_ind.index, y=df_ind[ema],
            name=ema, line=dict(color=col, width=1, dash="dot"),
            opacity=0.8,
        ), row=1, col=1)

    # Volume
    colors = ["#22c55e" if c >= o else "#ef4444"
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        name="Volume", marker_color=colors, opacity=0.6,
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=600,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=30, b=10),
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume",      row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Raw data expander
    with st.expander("View Raw Data"):
        st.dataframe(df.tail(50).sort_index(ascending=False), use_container_width=True)
        buf = io.BytesIO()
        df.to_csv(buf)
        st.download_button("Download Full CSV", buf.getvalue(),
                           file_name=f"{ticker}_data.csv", mime="text/csv")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — TECHNICAL INDICATORS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    fig2 = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Price + Bollinger Bands", "RSI (14)", "MACD"),
        row_heights=[0.5, 0.25, 0.25],
    )

    # Price + BB
    fig2.add_trace(go.Scatter(x=df_ind.index, y=df_ind["Close"],
                              name="Close", line=dict(color="#6366f1", width=1.5)), row=1, col=1)
    fig2.add_trace(go.Scatter(x=df_ind.index, y=df_ind["BB_Upper"],
                              name="BB Upper", line=dict(color="#94a3b8", width=1, dash="dash"),
                              opacity=0.6), row=1, col=1)
    fig2.add_trace(go.Scatter(x=df_ind.index, y=df_ind["BB_Lower"],
                              name="BB Lower", line=dict(color="#94a3b8", width=1, dash="dash"),
                              fill="tonexty", fillcolor="rgba(148,163,184,0.05)",
                              opacity=0.6), row=1, col=1)
    fig2.add_trace(go.Scatter(x=df_ind.index, y=df_ind["BB_Mid"],
                              name="BB Mid (SMA20)", line=dict(color="#f59e0b", width=1, dash="dot"),
                              opacity=0.7), row=1, col=1)

    # RSI
    fig2.add_trace(go.Scatter(x=df_ind.index, y=df_ind["RSI"],
                              name="RSI", line=dict(color="#f59e0b", width=1.5)), row=2, col=1)
    fig2.add_hline(y=70, line_color="#ef4444", line_dash="dash", opacity=0.5, row=2, col=1)
    fig2.add_hline(y=30, line_color="#22c55e", line_dash="dash", opacity=0.5, row=2, col=1)

    # MACD
    colors_macd = ["#22c55e" if v >= 0 else "#ef4444" for v in df_ind["MACD_Hist"].fillna(0)]
    fig2.add_trace(go.Bar(x=df_ind.index, y=df_ind["MACD_Hist"],
                          name="Histogram", marker_color=colors_macd, opacity=0.7), row=3, col=1)
    fig2.add_trace(go.Scatter(x=df_ind.index, y=df_ind["MACD"],
                              name="MACD", line=dict(color="#6366f1", width=1.5)), row=3, col=1)
    fig2.add_trace(go.Scatter(x=df_ind.index, y=df_ind["Signal"],
                              name="Signal", line=dict(color="#ec4899", width=1.5)), row=3, col=1)

    fig2.update_layout(
        template="plotly_dark",
        height=700,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60, b=10),
    )
    fig2.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig2.update_yaxes(title_text="RSI",         row=2, col=1, range=[0, 100])
    fig2.update_yaxes(title_text="MACD",        row=3, col=1)
    st.plotly_chart(fig2, use_container_width=True)

    # Indicator descriptions
    col_a, col_b = st.columns(2)
    with col_a:
        with st.expander("What is RSI?"):
            st.write("**Relative Strength Index** measures momentum on a 0–100 scale. "
                     "Above **70** = potentially overbought. Below **30** = potentially oversold.")
    with col_b:
        with st.expander("What is MACD?"):
            st.write("**Moving Average Convergence Divergence** shows the relationship between "
                     "two EMAs. A bullish signal occurs when MACD crosses above the Signal line.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — LSTM PREDICTION
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    if train_btn:
        st.markdown("### Training LSTM Model")
        close_arr = df["Close"].values.reshape(-1, 1)

        # Train/test split (last 20% as test)
        split = int(len(close_arr) * 0.80)
        train_raw = close_arr[:split]
        test_raw  = close_arr[split:]

        sc = MinMaxScaler((0, 1))
        train_scaled = sc.fit_transform(train_raw)
        test_scaled  = sc.transform(test_raw)

        X_tr, y_tr = build_sequences(train_scaled.flatten(), n_steps)
        X_tr = X_tr.reshape(-1, n_steps, 1)

        prog  = st.progress(0.0)
        stat  = st.empty()

        try:
            model, history, losses = train_lstm(X_tr, y_tr, n_steps, epochs, prog, stat)
            stat.success("Training complete!")

            # Test predictions
            X_te, y_te = build_sequences(test_scaled.flatten(), n_steps)
            X_te = X_te.reshape(-1, n_steps, 1)
            preds_scaled = model.predict(X_te, verbose=0)
            preds = sc.inverse_transform(preds_scaled).flatten()
            actuals = sc.inverse_transform(y_te.reshape(-1, 1)).flatten()

            # Metrics
            rmse = np.sqrt(mean_squared_error(actuals, preds))
            mae  = mean_absolute_error(actuals, preds)
            mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
            ss_res = np.sum((actuals - preds) ** 2)
            ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
            dir_acc = np.mean((np.diff(actuals) > 0) == (np.diff(preds) > 0)) * 100

            st.session_state.model   = model
            st.session_state.sc      = sc
            st.session_state.metrics = dict(rmse=rmse, mae=mae, mape=mape, r2=r2, dir_acc=dir_acc)

            # Prediction date index
            pred_dates = df.index[split + n_steps:]
            st.session_state.pred_df = pd.DataFrame({
                "Date":      pred_dates[:len(preds)],
                "Actual":    actuals,
                "Predicted": preds,
            }).set_index("Date")

        except ImportError:
            st.error("TensorFlow is not installed. Run: `pip install tensorflow`")

    # Show results if model exists
    if st.session_state.metrics:
        m = st.session_state.metrics
        st.markdown("#### Model Performance")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("RMSE",               f"${m['rmse']:.2f}")
        c2.metric("MAE",                f"${m['mae']:.2f}")
        c3.metric("MAPE",               f"{m['mape']:.2f}%")
        c4.metric("R²",                 f"{m['r2']:.4f}")
        c5.metric("Directional Acc.",   f"{m['dir_acc']:.1f}%")

    if st.session_state.pred_df is not None:
        pred_df = st.session_state.pred_df
        st.markdown("#### Actual vs Predicted")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=pred_df.index, y=pred_df["Actual"],
            name="Actual", line=dict(color="#6366f1", width=2),
        ))
        fig3.add_trace(go.Scatter(
            x=pred_df.index, y=pred_df["Predicted"],
            name="Predicted", line=dict(color="#f59e0b", width=2, dash="dash"),
        ))
        fig3.update_layout(
            template="plotly_dark", height=400,
            xaxis_title="Date", yaxis_title="Price (USD)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=30, b=10),
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Download
        buf = io.BytesIO()
        pred_df.to_csv(buf)
        st.download_button("Download Predictions CSV", buf.getvalue(),
                           file_name=f"{ticker}_predictions.csv", mime="text/csv")

    if st.session_state.model is None and not train_btn:
        st.info("Click **Train LSTM Model** in the sidebar to train and evaluate the model.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — FUTURE FORECAST
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    if st.session_state.model is None:
        st.info("Train the LSTM model first (sidebar → **Train LSTM Model**).")
    else:
        model = st.session_state.model
        sc    = st.session_state.sc

        close_arr    = df["Close"].values.reshape(-1, 1)
        scaled_all   = sc.transform(close_arr).flatten()

        # Seed from last n_steps
        seed = list(scaled_all[-n_steps:])
        future_preds = []
        for _ in range(future_days):
            x    = np.array(seed[-n_steps:]).reshape(1, n_steps, 1)
            pred = model.predict(x, verbose=0)[0, 0]
            future_preds.append(pred)
            seed.append(pred)

        future_prices = sc.inverse_transform(
            np.array(future_preds).reshape(-1, 1)
        ).flatten()

        last_date    = df.index[-1]
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1),
                                      periods=future_days)

        future_df = pd.DataFrame({
            "Date":  future_dates,
            "Price": future_prices,
        }).set_index("Date")
        st.session_state.future_df = future_df

        # ── Chart: history + forecast ──
        hist_tail = df["Close"].tail(120)
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=hist_tail.index, y=hist_tail.values,
            name="Historical", line=dict(color="#6366f1", width=2),
        ))
        fig4.add_trace(go.Scatter(
            x=future_df.index, y=future_df["Price"],
            name=f"{future_days}-Day Forecast",
            line=dict(color="#f59e0b", width=2.5, dash="dot"),
            mode="lines+markers",
            marker=dict(size=4),
        ))
        # Shaded forecast region
        fig4.add_vrect(
            x0=str(future_df.index[0].date()),
            x1=str(future_df.index[-1].date()),
            fillcolor="rgba(251,191,36,0.05)",
            layer="below", line_width=0,
        )
        fig4.update_layout(
            template="plotly_dark", height=420,
            xaxis_title="Date", yaxis_title="Price (USD)",
            title=f"{ticker} — {future_days}-Day Price Forecast",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=60, b=10),
        )
        st.plotly_chart(fig4, use_container_width=True)

        # KPIs
        start_p = float(future_prices[0])
        end_p   = float(future_prices[-1])
        change  = end_p - start_p
        change_pct = change / start_p * 100
        direction  = "Bullish 📈" if change >= 0 else "Bearish 📉"

        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric("Forecast Start",  f"${start_p:.2f}")
        fc2.metric("Forecast End",    f"${end_p:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
        fc3.metric("Min Forecast",    f"${float(future_prices.min()):.2f}")
        fc4.metric("Trend",           direction)

        st.markdown("#### Forecast Table")
        st.dataframe(
            future_df.reset_index().assign(
                Date=lambda d: d["Date"].dt.strftime("%Y-%m-%d"),
                Price=lambda d: d["Price"].round(2),
            ).rename(columns={"Price": "Forecast Price (USD)"}),
            use_container_width=True, hide_index=True,
        )

        buf = io.BytesIO()
        future_df.to_csv(buf)
        st.download_button("Download Forecast CSV", buf.getvalue(),
                           file_name=f"{ticker}_forecast.csv", mime="text/csv")

# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with Streamlit · Data from Yahoo Finance · Model: Stacked LSTM (TensorFlow/Keras)")
