import pandas as pd
import numpy as np
import pickle

from tensorflow.keras.models import load_model


SEQUENCE_LENGTH = 24

symbols = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "LTCUSDT",
    "XRPUSDT"
]


def load_crypto_model(symbol):

    model = load_model(
        f"models/{symbol}_lstm.keras"
    )

    with open(
        f"models/{symbol}_scaler.pkl",
        "rb"
    ) as file:

        scaler = pickle.load(file)

    return model, scaler


def load_crypto_data(symbol):

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


def predict_future_prices(
    symbol,
    hours
):

    df = load_crypto_data(symbol)

    model, scaler = load_crypto_model(
        symbol
    )

    prices = df["close"].values.reshape(
        -1, 1
    )

    # Scale historical prices
    scaled_prices = scaler.transform(
        prices
    )

    # Start with the latest 24 hours
    sequence = list(
        scaled_prices[-SEQUENCE_LENGTH:].flatten()
    )

    predictions = []

    for _ in range(hours):

        # Prepare input
        X = np.array(
            sequence[-SEQUENCE_LENGTH:],
            dtype=np.float32
        ).reshape(
            1,
            SEQUENCE_LENGTH,
            1
        )

        # Predict one hour
        predicted_scaled = model(
            X,
            training=False
        ).numpy()[0][0]

        # Store prediction
        predictions.append(
            predicted_scaled
        )

        # Add prediction to sequence
        sequence.append(
            predicted_scaled
        )

    # Convert predictions back to actual prices
    predictions = np.array(
        predictions
    ).reshape(-1, 1)

    predicted_prices = scaler.inverse_transform(
        predictions
    ).flatten()

    return predicted_prices


def predict_next_price(symbol):

    df = load_crypto_data(symbol)

    model, scaler = load_crypto_model(
        symbol
    )

    prices = df["close"].values.reshape(
        -1, 1
    )

    scaled_prices = scaler.transform(
        prices
    )

    latest_sequence = scaled_prices[
        -SEQUENCE_LENGTH:
    ]

    X = np.array(
        latest_sequence,
        dtype=np.float32
    ).reshape(
        1,
        SEQUENCE_LENGTH,
        1
    )

    predicted_scaled = model(
        X,
        training=False
    ).numpy()

    predicted_price = scaler.inverse_transform(
        predicted_scaled
    )[0][0]

    current_price = prices[-1][0]

    percentage_change = (
        (predicted_price - current_price)
        / current_price
    ) * 100

    return (
        current_price,
        predicted_price,
        percentage_change
    )


def show_prediction(symbol):

    print("\n" + "=" * 60)
    print(f"CRYPTocurrency: {symbol}")
    print("=" * 60)

    # ---------------------------------------------
    # NEXT HOUR
    # ---------------------------------------------

    current_price, next_hour_price, change = (
        predict_next_price(symbol)
    )

    print("\nNEXT 1 HOUR")
    print("-" * 40)

    print(
        f"Current price:   ${current_price:.4f}"
    )

    print(
        f"Predicted price: ${next_hour_price:.4f}"
    )

    print(
        f"Expected change: {change:+.2f}%"
    )

    # ---------------------------------------------
    # 5 DAYS
    # ---------------------------------------------

    five_day_predictions = predict_future_prices(
        symbol,
        24 * 5
    )

    five_day_price = five_day_predictions[-1]

    five_day_change = (
        (five_day_price - current_price)
        / current_price
    ) * 100

    print("\nNEXT 5 DAYS")
    print("-" * 40)

    print(
        f"Predicted price after 5 days: "
        f"${five_day_price:.4f}"
    )

    print(
        f"Expected change: "
        f"{five_day_change:+.2f}%"
    )

    # ---------------------------------------------
    # 10 DAYS
    # ---------------------------------------------

    ten_day_predictions = predict_future_prices(
        symbol,
        24 * 10
    )

    ten_day_price = ten_day_predictions[-1]

    ten_day_change = (
        (ten_day_price - current_price)
        / current_price
    ) * 100

    print("\nNEXT 10 DAYS")
    print("-" * 40)

    print(
        f"Predicted price after 10 days: "
        f"${ten_day_price:.4f}"
    )

    print(
        f"Expected change: "
        f"{ten_day_change:+.2f}%"
    )


if __name__ == "__main__":

    for symbol in symbols:

        show_prediction(symbol)

    print("\n" + "=" * 60)
    print("✅ All predictions completed!")
    print("=" * 60)