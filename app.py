import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings

warnings.filterwarnings('ignore')

# --- Page Layout ---
st.set_page_config(page_title="Advanced Crypto Dashboard", layout="wide")
st.title("📈 Pro-Grade Crypto Forecasting Pipeline")

# --- Session State for Leaderboard ---
if 'leaderboard' not in st.session_state:
    st.session_state['leaderboard'] = pd.DataFrame(columns=['Model', 'RMSE', 'MAE', 'Directional Accuracy (%)'])

# --- 1. Data Collection Pipeline ---
with st.sidebar:
    st.header("1. Data Configuration")
    ticker = st.text_input("Crypto Symbol", "BTC-USD")
    start_date = st.date_input("Start Date", pd.to_datetime("2021-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("today"))
    
    st.header("2. Model Parameters")
    forecast_horizon = st.slider("Forecast Days", 7, 60, 30)
    
    if st.button("Clear Leaderboard"):
        st.session_state['leaderboard'] = pd.DataFrame(columns=['Model', 'RMSE', 'MAE', 'Directional Accuracy (%)'])
        st.rerun()

@st.cache_data
def load_data(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end, progress=False)
        # Fix for yfinance MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        # Ensure Date column
        if 'Date' not in df.columns:
             if pd.api.types.is_datetime64_any_dtype(df.index):
                 df['Date'] = df.index
                 df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

raw_data = load_data(ticker, start_date, end_date)

if raw_data is not None and not raw_data.empty:
    
    # --- 2. Data Preprocessing ---
    model_data = raw_data.copy()
    
    # Feature Engineering
    model_data['Returns'] = model_data['Close'].pct_change()
    model_data['Log_Returns'] = np.log(model_data['Close'] / model_data['Close'].shift(1))
    model_data['MA_20'] = model_data['Close'].rolling(window=20).mean()
    model_data['MA_50'] = model_data['Close'].rolling(window=50).mean()
    model_data['Volatility'] = model_data['Close'].rolling(window=20).std()
    
    # RSI
    delta = model_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    model_data['RSI'] = 100 - (100 / (1 + rs))
    
    model_data.dropna(inplace=True)
    model_data.reset_index(drop=True, inplace=True)

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Market Analysis", "🔮 Forecasting Studio", "🏆 Model Accuracy & Evaluation", "🧠 Feature Engineering"])

    # --- TAB 1: Market Analysis ---
    with tab1:
        st.subheader(f"Price Action: {ticker}")
        
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05, row_heights=[0.6, 0.2, 0.2],
                            subplot_titles=("Candlestick & MAs", "Volume", "RSI"))

        fig.add_trace(go.Candlestick(x=model_data['Date'],
                                     open=model_data['Open'], high=model_data['High'],
                                     low=model_data['Low'], close=model_data['Close'], name='OHLC'), row=1, col=1)
        fig.add_trace(go.Scatter(x=model_data['Date'], y=model_data['MA_20'], line=dict(color='orange', width=1), name='MA 20'), row=1, col=1)
        
        colors = ['green' if row['Open'] - row['Close'] >= 0 else 'red' for index, row in model_data.iterrows()]
        fig.add_trace(go.Bar(x=model_data['Date'], y=model_data['Volume'], marker_color=colors, name='Volume'), row=2, col=1)
        
        fig.add_trace(go.Scatter(x=model_data['Date'], y=model_data['RSI'], line=dict(color='purple'), name='RSI'), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", row=3, col=1, line_color="red")
        fig.add_hline(y=30, line_dash="dot", row=3, col=1, line_color="green")

        fig.update_layout(height=800, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Simulated Sentiment Heatmap**")
            sim_data = np.random.rand(10, 24)
            fig_heat = px.imshow(sim_data, labels=dict(x="Hour", y="Day", color="Sentiment"), color_continuous_scale='RdBu')
            st.plotly_chart(fig_heat, use_container_width=True)

        with col2:
            st.write("**Market Regimes (Clustering)**")
            X_cluster = model_data[['Returns', 'Volatility']].dropna()
            if not X_cluster.empty:
                kmeans = KMeans(n_clusters=3, random_state=42)
                model_data['Cluster'] = kmeans.fit_predict(X_cluster)
                fig_cluster = px.scatter(model_data, x='Returns', y='Volatility', color=model_data['Cluster'].astype(str),
                                         title="Clusters: Low Vol vs High Vol", color_discrete_sequence=px.colors.qualitative.Safe)
                st.plotly_chart(fig_cluster, use_container_width=True)

    # --- TAB 2: Forecasting ---
    with tab2:
        st.subheader("Time Series Forecasting")
        model_choice = st.selectbox("Select Model", ["ARIMA", "SARIMA", "PROPHET", "LSTM"])
        
        if st.button("Train & Forecast"):
            with st.spinner(f"Training {model_choice}..."):
                try:
                    # Prepare Data Splits (Standard for linear models)
                    train_size = int(len(model_data) * 0.9)
                    train = model_data.iloc[:train_size]
                    test = model_data.iloc[train_size:]
                    
                    forecast_vals = []
                    conf_lower = []
                    conf_upper = []
                    
                    # --- ARIMA ---
                    if model_choice == "ARIMA":
                        model = ARIMA(train['Close'], order=(5,1,0))
                        res = model.fit()
                        fc = res.get_forecast(steps=len(test) + forecast_horizon)
                        forecast_vals = fc.predicted_mean
                        conf = fc.conf_int()
                        conf_lower, conf_upper = conf.iloc[:, 0], conf.iloc[:, 1]
                    
                    # --- SARIMA ---
                    elif model_choice == "SARIMA":
                        model = SARIMAX(train['Close'], order=(1,1,1), seasonal_order=(0,0,0,0)) 
                        res = model.fit(disp=False)
                        fc = res.get_forecast(steps=len(test) + forecast_horizon)
                        forecast_vals = fc.predicted_mean
                        conf = fc.conf_int()
                        conf_lower, conf_upper = conf.iloc[:, 0], conf.iloc[:, 1]
                        
                    # --- PROPHET ---
                    elif model_choice == "PROPHET":
                        df_p = train[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
                        m = Prophet()
                        m.fit(df_p)
                        future = m.make_future_dataframe(periods=len(test) + forecast_horizon)
                        fc = m.predict(future)
                        forecast_vals = fc['yhat'].tail(len(test) + forecast_horizon)
                        conf_lower = fc['yhat_lower'].tail(len(test) + forecast_horizon)
                        conf_upper = fc['yhat_upper'].tail(len(test) + forecast_horizon)
                        
                    # --- LSTM (FIXED & ROBUST) ---
                    elif model_choice == "LSTM":
                        scaler = MinMaxScaler()
                        # Scale the entire Close column
                        scaled_data = scaler.fit_transform(model_data['Close'].values.reshape(-1,1))
                        
                        # DYNAMIC LOOK_BACK: Ensure look_back isn't larger than the dataset
                        # We need at least enough data for 1 sequence + 1 label
                        max_look_back = max(1, int(len(scaled_data) * 0.2)) 
                        look_back = min(30, max_look_back) 
                        
                        # Data Generator
                        def create_dataset(dataset, look_back=30):
                            X, Y = [], []
                            # Ensure we don't go out of bounds
                            if len(dataset) <= look_back:
                                return np.array([]), np.array([])
                                
                            for i in range(len(dataset) - look_back - 1):
                                a = dataset[i:(i + look_back), 0]
                                X.append(a)
                                Y.append(dataset[i + look_back, 0])
                            return np.array(X), np.array(Y)
                        
                        X, y = create_dataset(scaled_data, look_back)
                        
                        if len(X) == 0:
                            st.error(f"Dataset too small for LSTM. Need more than {look_back} points.")
                            st.stop()
                        
                        # Reshape for LSTM [samples, time steps, features]
                        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
                        
                        # Split
                        lstm_train_size = int(len(X) * 0.9)
                        if lstm_train_size == 0:
                             st.error("Not enough data to create a training split.")
                             st.stop()

                        X_train, X_test = X[:lstm_train_size], X[lstm_train_size:]
                        y_train, y_test = y[:lstm_train_size], y[lstm_train_size:]
                        
                        # Model Architecture
                        model = Sequential()
                        model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
                        model.add(LSTM(50))
                        model.add(Dense(1))
                        model.compile(optimizer='adam', loss='mean_squared_error')
                        
                        # Train (Verbose=0 prevents progress bar thread errors)
                        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
                        
                        # Predict
                        if len(X_test) > 0:
                            test_predict = model.predict(X_test, verbose=0)
                            
                            # Future Forecast Logic
                            future_preds = []
                            last_window = X_test[-1].copy() # Get last known sequence
                            
                            for _ in range(forecast_horizon):
                                # Predict next step
                                next_pred = model.predict(last_window.reshape(1, look_back, 1), verbose=0)
                                val = next_pred[0,0]
                                future_preds.append(val)
                                
                                # Update window: drop first, add new prediction
                                last_window = np.roll(last_window, -1)
                                last_window[-1] = val
                                
                            # Combine
                            combined_pred = np.concatenate((test_predict.flatten(), future_preds))
                            forecast_vals = scaler.inverse_transform(combined_pred.reshape(-1,1)).flatten()
                            
                            conf_lower = forecast_vals * 0.95
                            conf_upper = forecast_vals * 1.05
                            
                            # Adjust 'test' data for evaluation to match LSTM timeframe
                            split_idx_original = look_back + 1 + lstm_train_size
                            test = model_data.iloc[split_idx_original:]
                            
                        else:
                            st.warning("Not enough test data generated. Try a larger date range.")
                            forecast_vals = np.zeros(1)

                    # --- METRICS & LEADERBOARD ---
                    # Align actual vs pred
                    min_len = min(len(test), len(forecast_vals))
                    y_true_eval = test['Close'].values[:min_len]
                    y_pred_eval = forecast_vals[:min_len] if isinstance(forecast_vals, np.ndarray) else forecast_vals.values[:min_len]
                    
                    if min_len > 0:
                        rmse = np.sqrt(mean_squared_error(y_true_eval, y_pred_eval))
                        mae = mean_absolute_error(y_true_eval, y_pred_eval)
                        
                        diff_true = np.diff(y_true_eval)
                        diff_pred = np.diff(y_pred_eval)
                        if len(diff_true) > 0:
                            da = np.mean(np.sign(diff_true) == np.sign(diff_pred)) * 100
                        else:
                            da = 0.0
                        
                        new_row = pd.DataFrame([{
                            'Model': model_choice, 
                            'RMSE': round(rmse, 2), 
                            'MAE': round(mae, 2), 
                            'Directional Accuracy (%)': round(da, 2)
                        }])
                        st.session_state['leaderboard'] = pd.concat([st.session_state['leaderboard'], new_row], ignore_index=True).drop_duplicates(subset=['Model'], keep='last')

                    # Store for visualization
                    st.session_state['forecast_vals'] = forecast_vals
                    st.session_state['test_data'] = test
                    st.session_state['conf_lower'] = conf_lower
                    st.session_state['conf_upper'] = conf_upper
                    st.session_state['train_data'] = train
                    st.session_state['model_choice'] = model_choice
                    
                    st.success(f"{model_choice} Trained! Go to 'Model Accuracy' tab to see results.")
                    
                except Exception as e:
                    st.error(f"Training Error: {e}")

        # Visualization
        if 'forecast_vals' in st.session_state:
            vals = st.session_state['forecast_vals']
            test = st.session_state['test_data']
            train = st.session_state['train_data']
            low = st.session_state['conf_lower']
            high = st.session_state['conf_upper']
            
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=train['Date'], y=train['Close'], name="History"))
            fig_f.add_trace(go.Scatter(x=test['Date'], y=test['Close'], name="Actual Test"))
            
            start_future = test['Date'].iloc[0]
            future_dates = pd.date_range(start=start_future, periods=len(vals))
            
            if st.session_state['model_choice'] != "LSTM":
                 fig_f.add_trace(go.Scatter(x=future_dates, y=high, mode='lines', line=dict(width=0), showlegend=False))
                 fig_f.add_trace(go.Scatter(x=future_dates, y=low, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,100,255,0.1)', name='Confidence'))
            
            fig_f.add_trace(go.Scatter(x=future_dates, y=vals, name="Forecast", line=dict(color='red', width=2)))
            st.plotly_chart(fig_f, use_container_width=True)

    # --- TAB 3: Model Accuracy & Evaluation ---
    with tab3:
        st.subheader("🏆 Model Leaderboard")
        st.info("Run different models in the 'Forecasting Studio' tab to populate this table.")
        
        # 1. Comparison Table
        if not st.session_state['leaderboard'].empty:
            st.dataframe(st.session_state['leaderboard'].style.highlight_min(subset=['RMSE', 'MAE'], color='lightgreen').highlight_max(subset=['Directional Accuracy (%)'], color='lightgreen'), use_container_width=True)
        else:
            st.write("No models run yet.")

        st.divider()
        st.subheader("Detailed Error Analysis (Latest Run)")
        
        if 'forecast_vals' in st.session_state:
            vals = st.session_state['forecast_vals']
            test = st.session_state['test_data']
            
            length = min(len(vals), len(test))
            y_true = test['Close'].values[:length]
            y_pred = vals[:length]
            
            if length > 0:
                residuals = y_true - y_pred
                col1, col2 = st.columns(2)
                with col1:
                    fig_hist = px.histogram(x=residuals, nbins=20, title="Residual Distribution (Error Histogram)")
                    st.plotly_chart(fig_hist, use_container_width=True)
                with col2:
                    df_res = pd.DataFrame({'Res': residuals})
                    df_res['Rolling_RMSE'] = np.sqrt((df_res['Res']**2).rolling(window=10).mean())
                    fig_roll = px.line(df_res, y='Rolling_RMSE', title="Rolling RMSE (Stability over Time)")
                    st.plotly_chart(fig_roll, use_container_width=True)

    # --- TAB 4: Feature Importance ---
    with tab4:
        st.subheader("Random Forest Feature Importance")
        try:
            feat_df = model_data[['MA_20', 'MA_50', 'RSI', 'Volatility', 'Volume']].dropna()
            target = model_data['Close'].loc[feat_df.index]
            
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(feat_df, target)
            
            importances = pd.DataFrame({
                'Feature': feat_df.columns,
                'Importance': rf.feature_importances_
            }).sort_values(by='Importance', ascending=True)
            
            fig_feat = px.bar(importances, x='Importance', y='Feature', orientation='h')
            st.plotly_chart(fig_feat, use_container_width=True)
        except Exception as e:
            st.error(f"Analysis Error: {e}")

else:
    st.warning("No data loaded. Check ticker symbol or internet connection.")