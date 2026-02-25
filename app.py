import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import warnings
import time

warnings.filterwarnings('ignore')

st.set_page_config(page_title="GE Stock Predictor", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(4, 15, 34) 0%, rgb(20, 20, 48) 90.1%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    h1 {
        background: -webkit-linear-gradient(45deg, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0em;
    }
    h2, h3 { color: #e2e8f0; font-weight: 600; }
    
    [data-testid="stMetricValue"] {
        color: #00f2fe;
        font-weight: 700;
        font-size: 2.2rem;
    }
    [data-testid="stMetricLabel"] { color: #94a3b8; }
    [data-testid="stMetricDelta"] svg { fill: #10b981; }
    
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(5px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    [data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 40px rgba(0, 242, 254, 0.15);
        border: 1px solid rgba(0, 242, 254, 0.3);
    }

    .stButton>button {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 2rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        text-transform: uppercase;
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.4);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('GE.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

def plot_forecast(actual_dates, actual_values, forecast_dates, forecast_values, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_dates, y=actual_values, mode='lines', name='Actual', line=dict(color='#00f2fe', width=2)))
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_values, mode='lines', name='Forecast', line=dict(color='#fbbf24', width=2, dash='dash')))
    
    fig.update_layout(
        title=title,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

def run_prophet(df, horizon):
    from prophet import Prophet
    df_p = df[['Date', 'Close(t)']].rename(columns={'Date': 'ds', 'Close(t)': 'y'})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, seasonality_mode='multiplicative', changepoint_prior_scale=0.05)
    m.fit(df_p)
    future = m.make_future_dataframe(periods=horizon, freq='D')
    forecast = m.predict(future)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_p['ds'], y=df_p['y'], mode='lines', name='Actual', line=dict(color='#00f2fe')))
    fig.add_trace(go.Scatter(x=forecast['ds'].tail(horizon), y=forecast['yhat'].tail(horizon), mode='lines', name='Forecast', line=dict(color='#fbbf24', dash='dash')))
    fig.add_trace(go.Scatter(x=forecast['ds'].tail(horizon), y=forecast['yhat_upper'].tail(horizon), mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast['ds'].tail(horizon), y=forecast['yhat_lower'].tail(horizon), mode='lines', fill='tonexty', fillcolor='rgba(251, 191, 36, 0.2)', line=dict(width=0), name='Confidence Interval'))
    
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', title="Prophet Forecast")
    st.plotly_chart(fig, use_container_width=True)

def run_arima(df, horizon):
    from statsmodels.tsa.arima.model import ARIMA
    series = df['Close(t)'].values
    model = ARIMA(series, order=(2,1,3))
    result = model.fit()
    forecast = result.predict(start=len(series), end=len(series)+horizon-1)
    last_date = df['Date'].iloc[-1]
    forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, horizon + 1)]
    plot_forecast(df['Date'], df['Close(t)'], forecast_dates, forecast, "ARIMA Forecast")

def run_sarima(df, horizon):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    series = df['Close(t)'].values
    model = SARIMAX(series, order=(2,1,3), seasonal_order=(0,0,2,12))
    results = model.fit(disp=False)
    forecast = results.forecast(steps=horizon)
    last_date = df['Date'].iloc[-1]
    forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, horizon + 1)]
    plot_forecast(df['Date'], df['Close(t)'], forecast_dates, forecast, "SARIMA Forecast")

def run_es(df, horizon):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    series = df['Close(t)'].values
    model_es = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=12)
    result_es = model_es.fit()
    forecast = result_es.forecast(steps=horizon)
    last_date = df['Date'].iloc[-1]
    forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, horizon + 1)]
    plot_forecast(df['Date'], df['Close(t)'], forecast_dates, forecast, "Exponential Smoothing Forecast")

def run_lstm(df, horizon):
    from sklearn.preprocessing import StandardScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    import tensorflow as tf
    import numpy as np
    
    cols = ['Close(t)', 'Open', 'High', 'Low', 'Volume']
    df_selected = df[cols].astype(float)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_selected)
    
    n_past = 14
    X, y = [], []
    for i in range(n_past, len(scaled_data)):
        X.append(scaled_data[i - n_past:i, :])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    
    model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),
        LSTM(32, activation='tanh', return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    progress_text = "Training LSTM Model. Please wait..."
    my_bar = st.progress(0, text=progress_text)
    
    class ProgressBarCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            my_bar.progress((epoch + 1) / 10, text=f"Training LSTM Model... Epoch {epoch+1}/10")
            
    model.fit(X, y, epochs=10, batch_size=32, verbose=0, callbacks=[ProgressBarCallback()])
    
    last_window = scaled_data[-n_past:]
    forecast_scaled = []
    
    for _ in range(horizon):
        pred = model.predict(last_window[np.newaxis, :, :], verbose=0)[0][0]
        forecast_scaled.append(pred)
        new_step = np.copy(last_window[-1])
        new_step[0] = pred
        last_window = np.vstack([last_window[1:], new_step])
        
    forecast_scaled = np.array(forecast_scaled)
    prediction_copies = np.repeat(forecast_scaled[:, np.newaxis], df_selected.shape[1], axis=-1)
    y_pred = scaler.inverse_transform(prediction_copies)[:, 0]
    
    last_date = df['Date'].iloc[-1]
    forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, horizon + 1)]
    my_bar.empty()
    plot_forecast(df['Date'], df['Close(t)'], forecast_dates, y_pred, "LSTM (Multivariate) Forecast")

def main():
    st.title("📈 General Electric Stock Forecaster")
    st.markdown("Predict future stock prices using advanced Machine Learning & Deep Learning models.")
    
    with st.spinner('Loading data...'):
        df = load_data()
        
    if df is None: return

    st.sidebar.title("Setup Config")
    model_choice = st.sidebar.selectbox("Select Model", ["Prophet", "ARIMA", "SARIMA", "Exponential Smoothing", "LSTM (Multivariate)"])
    horizon = st.sidebar.slider("Forecast Horizon (Days)", min_value=10, max_value=365, value=30, step=10)
    
    col1, col2, col3, col4 = st.columns(4)
    latest_price = df.iloc[-1]['Close(t)']
    prev_price = df.iloc[-2]['Close(t)']
    price_change = latest_price - prev_price
    pct_change = (price_change / prev_price) * 100

    with col1: st.metric(label="Latest Close Price", value=f"${latest_price:.2f}", delta=f"{price_change:.2f} ({pct_change:.2f}%)")
    with col2: st.metric(label="7D High", value=f"${df.tail(7)['High'].max():.2f}")
    with col3: st.metric(label="7D Low", value=f"${df.tail(7)['Low'].min():.2f}")
    with col4: st.metric(label="Volume", value=f"{df.iloc[-1]['Volume']:,.0f}")

    st.subheader("Historical Stock Price")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close(t)'],
                                 increasing_line_color='#10b981', decreasing_line_color='#ef4444'))
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=0, r=0, t=30, b=0), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.sidebar.markdown("---")
    if st.sidebar.button("Run Forecast 🚀"):
        st.markdown(f"### {model_choice} Forecasting")
        with st.spinner(f'Training {model_choice} and forecasting {horizon} days ahead...'):
            start_time = time.time()
            if model_choice == "Prophet": run_prophet(df, horizon)
            elif model_choice == "ARIMA": run_arima(df, horizon)
            elif model_choice == "SARIMA": run_sarima(df, horizon)
            elif model_choice == "Exponential Smoothing": run_es(df, horizon)
            elif model_choice == "LSTM (Multivariate)": run_lstm(df, horizon)
            st.success(f"Forecast completed in {time.time() - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()
