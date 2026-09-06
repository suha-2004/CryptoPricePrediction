import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import plotly.graph_objects as go

from tensorflow.keras.models import load_model


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="CryptoPredict",
    page_icon="₿",
    layout="wide"
)


# ==========================================================
# CRYPTO INFORMATION
# ==========================================================

crypto_info = {

    "BTCUSDT": {
        "name": "Bitcoin",
        "short": "BTC",
        "image": "https://assets.coingecko.com/coins/images/1/large/bitcoin.png"
    },

    "ETHUSDT": {
        "name": "Ethereum",
        "short": "ETH",
        "image": "https://assets.coingecko.com/coins/images/279/large/ethereum.png"
    },

    "BNBUSDT": {
        "name": "BNB",
        "short": "BNB",
        "image": "https://assets.coingecko.com/coins/images/825/large/bnb-icon2_2x.png"
    },

    "LTCUSDT": {
        "name": "Litecoin",
        "short": "LTC",
        "image": "https://assets.coingecko.com/coins/images/2/large/litecoin.png"
    },

    "XRPUSDT": {
        "name": "XRP",
        "short": "XRP",
        "image": "https://assets.coingecko.com/coins/images/44/large/xrp-symbol-white-128.png"
    }
}


# ==========================================================
# MODEL METRICS
# ==========================================================

metrics = {

    "BTCUSDT": {
        "mae": 549.11,
        "rmse": 735.53,
        "mape": 0.70,
        "accuracy": 99.30
    },

    "ETHUSDT": {
        "mae": 21.23,
        "rmse": 27.43,
        "mape": 0.87,
        "accuracy": 99.13
    },

    "BNBUSDT": {
        "mae": 6.69,
        "rmse": 10.51,
        "mape": 0.92,
        "accuracy": 99.08
    },

    "LTCUSDT": {
        "mae": 0.60,
        "rmse": 0.82,
        "mape": 1.17,
        "accuracy": 98.83
    },

    "XRPUSDT": {
        "mae": 0.02,
        "rmse": 0.02,
        "mape": 1.16,
        "accuracy": 98.84
    }
}


# ==========================================================
# LIVE PRICE SETTINGS
# ==========================================================

LIVE_PRICE_CACHE_SECONDS = 30
USD_INR_CACHE_SECONDS = 300

FALLBACK_USD_TO_INR = 88.0


# ==========================================================
# SESSION STATE
# ==========================================================

if "selected_coin" not in st.session_state:
    st.session_state.selected_coin = None


# ==========================================================
# CSS
# ==========================================================

st.markdown(
    """
    <style>

    .title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #777777;
        margin-bottom: 25px;
    }

    .coin-name {
        font-size: 23px;
        font-weight: 600;
        margin-top: 12px;
    }

    .coin-symbol {
        color: #888888;
        font-size: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# LOAD HISTORICAL DATA
# ==========================================================

@st.cache_data
def load_data(symbol):

    df = pd.read_csv(
        f"data/{symbol}.csv"
    )

    df["open_time"] = pd.to_datetime(
        df["open_time"]
    )

    df = df.sort_values(
        "open_time"
    )

    return df


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_prediction_model(symbol):

    return load_model(
        f"models/{symbol}_lstm.keras"
    )


# ==========================================================
# LOAD SCALER
# ==========================================================

@st.cache_resource
def load_scaler(symbol):

    with open(
        f"models/{symbol}_scaler.pkl",
        "rb"
    ) as file:

        return pickle.load(file)


# ==========================================================
# GET LIVE CRYPTO PRICE FROM BINANCE
# ==========================================================

@st.cache_data(ttl=LIVE_PRICE_CACHE_SECONDS)
def get_live_price(symbol):

    url = "https://api.binance.com/api/v3/ticker/price"

    params = {
        "symbol": symbol
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        return float(
            data["price"]
        )

    except Exception:

        return None


# ==========================================================
# GET USD TO INR EXCHANGE RATE
# ==========================================================

@st.cache_data(ttl=USD_INR_CACHE_SECONDS)
def get_usd_to_inr():

    try:

        url = "https://api.frankfurter.app/latest"

        params = {
            "from": "USD",
            "to": "INR"
        }

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        return float(
            data["rates"]["INR"]
        )

    except Exception:

        return FALLBACK_USD_TO_INR


# ==========================================================
# PRICE FORMATTING
# ==========================================================

def usd_price(price):

    return f"${price:,.4f}"


def inr_price(price):

    usd_to_inr = get_usd_to_inr()

    return f"₹{price * usd_to_inr:,.2f}"


# ==========================================================
# PREDICT FUTURE PRICES
# ==========================================================

def predict_future_prices(
    df,
    model,
    scaler,
    hours
):

    prices = df[
        "close"
    ].values.reshape(
        -1,
        1
    )

    scaled_prices = scaler.transform(
        prices
    )

    sequence = list(
        scaled_prices[-24:].flatten()
    )

    predictions = []

    for _ in range(hours):

        X = np.array(
            sequence[-24:],
            dtype=np.float32
        ).reshape(
            1,
            24,
            1
        )

        prediction = model.predict(
            X,
            verbose=0
        )

        predicted_scaled = float(
            prediction[0][0]
        )

        predictions.append(
            predicted_scaled
        )

        sequence.append(
            predicted_scaled
        )

    predictions = np.array(
        predictions
    ).reshape(
        -1,
        1
    )

    predicted_prices = scaler.inverse_transform(
        predictions
    ).flatten()

    return predicted_prices


# ==========================================================
# NEXT HOUR PREDICTION
# ==========================================================

def predict_next_hour(
    symbol,
    df,
    model,
    scaler
):

    prediction = predict_future_prices(
        df,
        model,
        scaler,
        1
    )

    # Try live Binance price first
    current_price = get_live_price(
        symbol
    )

    # If live price unavailable,
    # use latest historical price
    if current_price is None:

        current_price = float(
            df["close"].iloc[-1]
        )

    predicted_price = float(
        prediction[0]
    )

    change = (
        (predicted_price - current_price)
        / current_price
    ) * 100

    return (
        current_price,
        predicted_price,
        change
    )


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title(
    "₿ CryptoPredict"
)

st.sidebar.divider()


if st.session_state.selected_coin is not None:

    if st.sidebar.button(
        "← Back to Home",
        use_container_width=True
    ):

        st.session_state.selected_coin = None

        st.rerun()


st.sidebar.info(
    "The LSTM model uses the previous "
    "24 hours of hourly price data."
)

st.sidebar.divider()

st.sidebar.write(
    "**Model:** LSTM"
)

st.sidebar.write(
    "**Frequency:** 1 Hour"
)

st.sidebar.write(
    "**Coins:** 5"
)

st.sidebar.divider()

st.sidebar.caption(
    "🟢 Current prices are fetched "
    "from Binance."
)

st.sidebar.caption(
    "🤖 Forecasts are generated "
    "by the trained LSTM models."
)


# ==========================================================
# HOME PAGE
# ==========================================================

if st.session_state.selected_coin is None:

    st.markdown(
        '<div class="title">'
        '₿ Cryptocurrency Price Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'AI-based cryptocurrency price forecasting '
        'using LSTM neural networks'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader(
        "🪙 Choose a Cryptocurrency"
    )

    st.write(
        "Current prices are fetched live from Binance. "
        "Click **View Details** to explore the forecast."
    )

    st.write("")

    symbols = list(
        crypto_info.keys()
    )


    # ======================================================
    # FIRST ROW
    # ======================================================

    columns = st.columns(3)

    for column, symbol in zip(
        columns,
        symbols[:3]
    ):

        info = crypto_info[symbol]

        with column:

            df = load_data(
                symbol
            )

            model = load_prediction_model(
                symbol
            )

            scaler = load_scaler(
                symbol
            )

            current_price, next_price, change = (
                predict_next_hour(
                    symbol,
                    df,
                    model,
                    scaler
                )
            )


            # ------------------------------------------------
            # IMAGE
            # ------------------------------------------------

            st.image(
                info["image"],
                width=110
            )


            # ------------------------------------------------
            # NAME
            # ------------------------------------------------

            st.markdown(
                f"### {info['name']}"
            )

            st.caption(
                info["short"]
            )


            # ------------------------------------------------
            # CURRENT PRICE
            # ------------------------------------------------

            st.metric(
                "🟢 Live Current Price",
                usd_price(
                    current_price
                )
            )

            st.caption(
                inr_price(
                    current_price
                )
            )


            # ------------------------------------------------
            # BUTTON
            # ------------------------------------------------

            if st.button(
                f"View {info['name']} Details →",
                key=f"view_{symbol}",
                use_container_width=True
            ):

                st.session_state.selected_coin = symbol

                st.rerun()


    # ======================================================
    # SECOND ROW
    # ======================================================

    st.write("")

    columns = st.columns(3)

    for column, symbol in zip(
        columns[:2],
        symbols[3:]
    ):

        info = crypto_info[symbol]

        with column:

            df = load_data(
                symbol
            )

            model = load_prediction_model(
                symbol
            )

            scaler = load_scaler(
                symbol
            )

            current_price, next_price, change = (
                predict_next_hour(
                    symbol,
                    df,
                    model,
                    scaler
                )
            )


            st.image(
                info["image"],
                width=110
            )

            st.markdown(
                f"### {info['name']}"
            )

            st.caption(
                info["short"]
            )

            st.metric(
                "🟢 Live Current Price",
                usd_price(
                    current_price
                )
            )

            st.caption(
                inr_price(
                    current_price
                )
            )


            if st.button(
                f"View {info['name']} Details →",
                key=f"view_{symbol}",
                use_container_width=True
            ):

                st.session_state.selected_coin = symbol

                st.rerun()


    # ======================================================
    # PROJECT INFORMATION
    # ======================================================

    st.divider()

    st.subheader(
        "🤖 About the Project"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            "**LSTM Neural Network**\n\n"
            "Deep learning model for time-series forecasting."
        )

    with col2:

        st.info(
            "**24-Hour Input Window**\n\n"
            "The previous 24 hourly prices are used as input."
        )

    with col3:

        st.info(
            "**5 Cryptocurrencies**\n\n"
            "BTC, ETH, BNB, LTC and XRP."
        )


    st.warning(
        "⚠️ Forecasts are model estimates for "
        "educational purposes and are not financial advice."
    )


# ==========================================================
# COIN DETAILS PAGE
# ==========================================================

else:

    symbol = st.session_state.selected_coin

    info = crypto_info[symbol]


    # ======================================================
    # LOAD
    # ======================================================

    df = load_data(
        symbol
    )

    model = load_prediction_model(
        symbol
    )

    scaler = load_scaler(
        symbol
    )


    # ======================================================
    # LIVE CURRENT PRICE
    # ======================================================

    live_price = get_live_price(
        symbol
    )

    if live_price is None:

        current_price = float(
            df["close"].iloc[-1]
        )

        live_available = False

    else:

        current_price = live_price

        live_available = True


    # ======================================================
    # PREDICTIONS
    # ======================================================

    next_hour_prediction = predict_future_prices(
        df,
        model,
        scaler,
        1
    )

    next_hour_price = float(
        next_hour_prediction[0]
    )


    five_day_predictions = predict_future_prices(
        df,
        model,
        scaler,
        24 * 5
    )


    ten_day_predictions = predict_future_prices(
        df,
        model,
        scaler,
        24 * 10
    )


    five_day_price = float(
        five_day_predictions[-1]
    )

    ten_day_price = float(
        ten_day_predictions[-1]
    )


    # ======================================================
    # PRICE CHANGES
    # ======================================================

    hourly_change = (
        (next_hour_price - current_price)
        / current_price
    ) * 100


    five_day_change = (
        (five_day_price - current_price)
        / current_price
    ) * 100


    ten_day_change = (
        (ten_day_price - current_price)
        / current_price
    ) * 100


    # ======================================================
    # HEADER
    # ======================================================

    col1, col2 = st.columns(
        [1, 5]
    )

    with col1:

        st.image(
            info["image"],
            width=100
        )

    with col2:

        st.title(
            f"{info['name']} ({info['short']})"
        )

        if live_available:

            st.success(
                "🟢 Live Binance market price"
            )

        else:

            st.warning(
                "⚠️ Binance price unavailable. "
                "Showing latest stored price."
            )


    st.write(
        "Complete cryptocurrency price forecast "
        "and model analysis."
    )

    st.divider()


    # ======================================================
    # CURRENT PRICE
    # ======================================================

    st.subheader(
        "💰 Current Market Price"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "USD",
            usd_price(
                current_price
            )
        )

    with col2:

        st.metric(
            "Indian Rupees",
            inr_price(
                current_price
            )
        )


    if live_available:

        st.caption(
            "Current price is fetched from Binance."
        )

    else:

        st.caption(
            "Live price could not be fetched. "
            "Showing the latest price in your dataset."
        )


    st.divider()


    # ======================================================
    # FORECASTS
    # ======================================================

    st.subheader(
        "🔮 Price Predictions"
    )

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Next 1 Hour",
            usd_price(
                next_hour_price
            ),
            f"{hourly_change:+.2f}%"
        )

        st.caption(
            inr_price(
                next_hour_price
            )
        )


    with col2:

        st.metric(
            "After 5 Days",
            usd_price(
                five_day_price
            ),
            f"{five_day_change:+.2f}%"
        )

        st.caption(
            inr_price(
                five_day_price
            )
        )


    with col3:

        st.metric(
            "After 10 Days",
            usd_price(
                ten_day_price
            ),
            f"{ten_day_change:+.2f}%"
        )

        st.caption(
            inr_price(
                ten_day_price
            )
        )


    st.caption(
        "⚠️ 5-day and 10-day values are recursive "
        "model projections, not guaranteed future prices."
    )


    # ======================================================
    # GRAPH
    # ======================================================

    st.divider()

    st.subheader(
        "📈 Historical & Forecast Graph"
    )


    forecast_option = st.radio(
        "Forecast Horizon",
        [
            "Next 1 Hour",
            "Next 5 Days",
            "Next 10 Days"
        ],
        horizontal=True
    )


    # Use current time as forecast starting point
    current_time = pd.Timestamp.now()


    if forecast_option == "Next 1 Hour":

        forecast_prices = np.array([
            next_hour_price
        ])

        periods = 1


    elif forecast_option == "Next 5 Days":

        forecast_prices = five_day_predictions

        periods = 120


    else:

        forecast_prices = ten_day_predictions

        periods = 240


    forecast_times = pd.date_range(
        start=current_time + pd.Timedelta(hours=1),
        periods=periods,
        freq="h"
    )


    # Historical data
    historical_data = df.tail(100)


    fig = go.Figure()


    # ------------------------------------------------------
    # HISTORICAL PRICE
    # ------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=historical_data["open_time"],
            y=historical_data["close"],
            mode="lines",
            name="Historical Price"
        )
    )


    # ------------------------------------------------------
    # FORECAST
    # ------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=forecast_times,
            y=forecast_prices,
            mode="lines",
            name="LSTM Forecast",
            line=dict(
                dash="dash"
            )
        )
    )


    # ------------------------------------------------------
    # FORECAST START
    # ------------------------------------------------------

    fig.add_vline(
        x=current_time,
        line_dash="dot",
        annotation_text="Forecast starts"
    )


    # ------------------------------------------------------
    # GRAPH LAYOUT
    # ------------------------------------------------------

    fig.update_layout(
        title=f"{info['name']} Price Forecast",
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        height=550
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ======================================================
    # MODEL PERFORMANCE
    # ======================================================

    st.divider()

    st.subheader(
        "🤖 Model Performance"
    )


    coin_metrics = metrics[symbol]

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "MAE",
            f"{coin_metrics['mae']:.2f}"
        )


    with col2:

        st.metric(
            "RMSE",
            f"{coin_metrics['rmse']:.2f}"
        )


    with col3:

        st.metric(
            "MAPE",
            f"{coin_metrics['mape']:.2f}%"
        )


    with col4:

        st.metric(
            "Approx. Accuracy",
            f"{coin_metrics['accuracy']:.2f}%"
        )


    st.caption(
        "MAE and RMSE measure prediction error. "
        "MAPE represents mean absolute percentage error. "
        "Approximate accuracy is calculated as 100 − MAPE."
    )


    # ======================================================
    # MARKET DATA
    # ======================================================

    st.divider()

    st.subheader(
        "📋 Latest Historical Market Data"
    )


    latest_data = df.tail(
        10
    ).copy()


    latest_data = latest_data[
        [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    ]


    st.dataframe(
        latest_data,
        use_container_width=True,
        hide_index=True
    )


    # ======================================================
    # DATA INFORMATION
    # ======================================================

    st.divider()

    st.subheader(
        "📊 Data Information"
    )

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Historical Records",
            f"{len(df):,}"
        )


    with col2:

        st.metric(
            "Frequency",
            "1 Hour"
        )


    with col3:

        st.metric(
            "Input Window",
            "24 Hours"
        )


    # ======================================================
    # DISCLAIMER
    # ======================================================

    st.divider()

    st.caption(
        "LSTM Model | 24-Hour Input Window | "
        "Hourly Data | 5 Cryptocurrency Models"
    )

    st.caption(
        "🟢 Current market price: Binance API"
    )

    st.caption(
        "🤖 Forecast: Trained LSTM model"
    )

    st.caption(
        "⚠️ Educational project only. "
        "Cryptocurrency forecasts are estimates "
        "and should not be considered financial advice."
    )