import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import pickle


symbols = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "LTCUSDT",
    "XRPUSDT"
]

SEQUENCE_LENGTH = 24

os.makedirs("models", exist_ok=True)


def preprocess_data(symbol):

    print(f"\nProcessing {symbol}...")

    # Load data
    file_path = f"data/{symbol}.csv"
    df = pd.read_csv(file_path)

    # Sort by time
    df["open_time"] = pd.to_datetime(df["open_time"])
    df = df.sort_values("open_time")

    # Select closing price
    prices = df["close"].values.reshape(-1, 1)

    # Scale values between 0 and 1
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices)

    # Create sequences
    X = []
    y = []

    for i in range(SEQUENCE_LENGTH, len(scaled_prices)):
        X.append(scaled_prices[i - SEQUENCE_LENGTH:i])
        y.append(scaled_prices[i])

    X = np.array(X)
    y = np.array(y)

    # 80% training, 20% testing
    train_size = int(len(X) * 0.8)

    X_train = X[:train_size]
    X_test = X[train_size:]

    y_train = y[:train_size]
    y_test = y[train_size:]

    # Save processed data
    np.save(f"models/{symbol}_X_train.npy", X_train)
    np.save(f"models/{symbol}_X_test.npy", X_test)
    np.save(f"models/{symbol}_y_train.npy", y_train)
    np.save(f"models/{symbol}_y_test.npy", y_test)

    # Save scaler
    with open(f"models/{symbol}_scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    print("Total sequences:", len(X))
    print("Training sequences:", len(X_train))
    print("Testing sequences:", len(X_test))
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("Processed data saved.")


if __name__ == "__main__":

    for symbol in symbols:
        preprocess_data(symbol)

    print("\n✅ Preprocessing completed and saved for all cryptocurrencies!")