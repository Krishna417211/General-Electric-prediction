# Comprehensive Project Report: General Electric (GE) Stock Price Prediction

## 1. Executive Summary
This project presents an end-to-end machine learning pipeline for forecasting the historical stock data of General Electric (`GE.csv`). The core objective was to analyze past price action and forecast future 'Close' prices by evaluating a suite of five distinct models—ranging from classical statistical time-series algorithms to advanced deep-learning neural networks. Finally, to make the predictive models accessible and interactive, a modern, data-driven web application was built and deployed locally using Streamlit.

## 2. Dataset & Exploratory Data Analysis (EDA)
### 2.1 Data Source
The foundational dataset used in this project is `GE.csv`, representing historical market records for General Electric.

### 2.2 Features Used
The dataset consists of several critical financial indicators:
*   `Date`: The chronological timestamp of the trading day.
*   `Close(t)`: The closing price of the stock, which serves as the primary **target variable** for all forecasting models.
*   `Open`, `High`, `Low`: Intraday price variations representing market volatility.
*   `Volume`: The total number of shares traded, indicating market activity and liquidity.

### 2.3 Data Preprocessing
To ensure data quality and integrity before modeling, the following preprocessing steps were undertaken:
*   **Chronological Sorting:** The data was verified to be strictly ordered by `Date` to maintain the temporal sequence essential for time-series forecasting.
*   **Missing Value Imputation:** The dataset was checked for null values in the `Close(t)` column to prevent computational errors during model training.
*   **Train-Test Split:** A standard **80/20 chronological split** was employed. The first 80% of the timeline served as the training environment, while the final 20% was held out specifically to evaluate the models against unseen, future data.
*   **Feature Scaling:** For the deep-learning model (LSTM), a `StandardScaler` was fit exclusively on the training set and applied to both train and test sets to prevent data leakage while normalizing the continuous variables (Open, High, Low, Close, Volume) for optimal gradient descent optimization.

## 3. Modeling Architectures & Techniques
To capture varying underlying patterns (trend, seasonality, and complex non-linear relationships), five distinct forecasting techniques were implemented:

### 3.1 Prophet (by Meta)
*   **Methodology:** An additive regression model that works exceptionally well with time-series data exhibiting strong seasonal effects and historical trend shifts.
*   **Configuration:** 
    *   Set with `seasonality_mode='multiplicative'` as stock prices theoretically grow exponentially rather than linearly.
    *   `yearly_seasonality=True` was enabled to capture annual market cycles.
    *   `changepoint_prior_scale=0.05` to balance the flexibility of automatic trend changepoint detection without overfitting.

### 3.2 ARIMA (AutoRegressive Integrated Moving Average)
*   **Methodology:** A classical and robust statistical approach that explains a given time series based on its own past values (AutoRegressive) and past errors (Moving Average), using differencing (Integrated) to make the data stationary.
*   **Configuration:** Hyperparameter tuning was automated using `pmdarima.auto_arima`, which identified the optimal order `(p, d, q) = (2, 1, 3)`.

### 3.3 SARIMA (Seasonal ARIMA)
*   **Methodology:** An extension of ARIMA that explicitly supports univariate time series data with a seasonal component.
*   **Configuration:** In addition to the base order `(2, 1, 3)`, a seasonal order of `(P, D, Q, m) = (0, 0, 2, 12)` was applied, utilizing a 12-period cycle to better capture recurring monthly or yearly trading floors and ceilings.

### 3.4 Exponential Smoothing (Holt-Winters)
*   **Methodology:** A rule-of-thumb technique for smoothing time series data using the exponential window function, applying decreasing weights to older observations.
*   **Configuration:** Configured with an `additive` trend and an `additive` seasonal component over a 12-period horizon, allowing the model to project steady linear growth with cyclical adjustments.

### 3.5 Multi-variate LSTM (Long Short-Term Memory)
*   **Methodology:** An advanced Recurrent Neural Network (RNN) architecture capable of learning long-term dependencies. Unlike the previous models, the LSTM was designed as a **multivariate** forecaster, utilizing all features (`Open`, `High`, `Low`, `Close`, `Volume`) to predict the target.
*   **Configuration:**
    *   **Sequence Window:** A `14-day` lookback window was used, meaning the model uses the past two weeks of full market data to predict the next day's closing price.
    *   **Network Layout:** Two consecutive LSTM layers consisting of 64 and 32 units respectively, utilizing the `tanh` activation function.
    *   **Regularization:** A `Dropout` layer (rate of 0.2) was integrated to randomly shut off neurons during training, heavily mitigating the risk of overfitting.
    *   **Compilation:** Trained using the `Adam` optimizer against Mean Squared Error (`MSE`) loss.
    *   **Early Stopping:** Deployed with a patience of 5 epochs on validation loss to restore the best weights automatically.

## 4. Model Evaluation & Performance Metrics
During the initial Jupyter Notebook research phase, each model was rigorously evaluated on the held-out 20% test set. The primary metrics tracked to evaluate performance included:
*   **Root Mean Squared Error (RMSE):** The core metric used to heavily penalize larger prediction errors.
*   **Mean Absolute Error (MAE):** Provided a linear measurement of the average error magnitude.
*   **Mean Squared Error (MSE):** The underlying variance metric utilized to calculate RMSE.
*   **AIC & BIC:** Used strictly for the statistical models (ARIMA, SARIMA, ES) to measure the relative quality of the models, penalizing complexity (number of parameters) to avoid overfitting.

## 5. Deployment, Web Interface, & Engineering Operations
To transform the research into a tangible product, the models were integrated into a production-ready web application stream using **Streamlit**.

### 5.1 System Architecture
*   **Backend:** Pure Python 3 environment utilizing numerical and statistical libraries (`numpy`, `pandas`, `statsmodels`, `tensorflow`, `prophet`).
*   **Frontend:** Streamlit, serving dynamic HTML/CSS/JS without requiring manual web development code.
*   **Optimization:** Employed Streamlit's `@st.cache_data` decorator to load the `GE.csv` data once into RAM, highly optimizing the application's responsiveness across different user sessions.

### 5.2 UI/UX Design (Visual Aesthetics)
*   **Theme:** The interface was built with a custom dark-mode aesthetic, utilizing deep navy-blue gradients.
*   **Glassmorphism Elements:** The sidebar and metric cards feature a modern frosted-glass blur effect, achieved through custom, injected CSS styling (`backdrop-filter`).
*   **Typography & Colors:** Integrated clean sans-serif fonts (`Inter`) alongside vibrant, neon-style color palettes (Cyan and Amber) to highlight critical interactions.

### 5.3 Interactive Capabilities
*   **User Controls:** The sidebar allows the client to seamlessly switch between any of the 5 implemented forecasting models and dynamically adjust the prediction horizon via a slider (ranging from 10 to 365 days into the future).
*   **Real-time Data Fetching:** Top-level metric cards automatically calculate and display the latest Close Price, 7-Day Highs/Lows, Volume, and daily percentage changes.
*   **Dynamic Visualizations:** Static `matplotlib` images were entirely replaced with **Plotly**. This integration provides fully responsive trading charts containing hover-tooltips, zoom capabilities, candlestick financial plotting, and visual confidence intervals (for the Prophet model).

### 5.4 Version Control
The full lifecycle of the project—spanning from the initial `.ipynb` notebook to the finalized `app.py`, `requirements.txt`, and data handling—was initialized and committed to a Git repository. It has been successfully pushed and is currently maintained on the central GitHub repository.

## 6. Conclusion and Future Work
This project successfully bridges the gap between raw data analysis and actionable insights. By benchmarking classical statistical models against modern deep-learning architectures, it provides a comprehensive overview of General Electric's historical price action while delivering a beautiful, localized tool for continuous future forecasting. 

**Future Iterations may include:**
*   Connecting the application to a live stock market API (e.g., Yahoo Finance or Alpaca) instead of relying on a static CSV file.
*   Implementing hyperparameter tuning controls directly inside the Streamlit UI.
*   Adding Explainable AI (XAI) overlays to determine which feature (e.g., Volume vs. Highs) most heavily influences the LSTM's predictions.
