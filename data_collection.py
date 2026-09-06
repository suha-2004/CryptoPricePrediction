import requests
import pandas as pd
import os
import time

# Cryptocurrency symbols
symbols = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "LTCUSDT",
    "XRPUSDT"
]

# Binance API
url = "https://api.binance.com/api/v3/klines"

# Data settings
interval = "1h"
limit = 1000

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)


def collect_data(symbol):
    print(f"\nCollecting {symbol} data...")

    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"Error collecting {symbol}: {response.status_code}")
        return

    data = response.json()

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore"
    ]

    df = pd.DataFrame(data, columns=columns)

    # Convert timestamp to readable date/time
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")

    # Keep only useful columns
    df = df[
        [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote"
        ]
    ]

    # Convert numerical columns
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Save CSV
    file_path = f"data/{symbol}.csv"
    df.to_csv(file_path, index=False)

    print(f"Saved: {file_path}")
    print(f"Rows: {len(df)}")
    print(f"Date range: {df['open_time'].min()} → {df['open_time'].max()}")


# Collect data for all five cryptocurrencies
for symbol in symbols:
    collect_data(symbol)
    time.sleep(1)

print("\n✅ All cryptocurrency datasets collected successfully!")