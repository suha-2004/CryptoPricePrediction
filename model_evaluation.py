import numpy as np
import pandas as pd
import pickle

from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error


symbols = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "LTCUSDT",
    "XRPUSDT"
]


def evaluate_model(symbol):

    print("\n" + "=" * 50)
    print(f"Evaluating {symbol}")
    print("=" * 50)

    # Load test data
    X_test = np.load(f"models/{symbol}_X_test.npy")
    y_test = np.load(f"models/{symbol}_y_test.npy")

    # Load scaler
    with open(f"models/{symbol}_scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    # Load trained model
    model = load_model(f"models/{symbol}_lstm.keras")

    # Make predictions
    predictions = model.predict(X_test, verbose=0)

    # Convert scaled values back to actual prices
    actual_prices = scaler.inverse_transform(y_test)
    predicted_prices = scaler.inverse_transform(predictions)

    # Calculate MAE
    mae = mean_absolute_error(
        actual_prices,
        predicted_prices
    )

    # Calculate RMSE
    rmse = np.sqrt(
        mean_squared_error(
            actual_prices,
            predicted_prices
        )
    )

    # Calculate MAPE
    mape = np.mean(
        np.abs(
            (actual_prices - predicted_prices)
            / actual_prices
        )
    ) * 100

    # Calculate approximate accuracy
    accuracy = 100 - mape

    print(f"\nMAE      : {mae:.2f}")
    print(f"RMSE     : {rmse:.2f}")
    print(f"MAPE     : {mape:.2f}%")
    print(f"Accuracy : {accuracy:.2f}%")

    # Show sample predictions
    results = pd.DataFrame({
        "Actual Price": actual_prices.flatten(),
        "Predicted Price": predicted_prices.flatten()
    })

    print("\nSample predictions:")
    print(results.head(10))

    return mae, rmse, mape, accuracy


if __name__ == "__main__":

    print("\nCRYPTOCURRENCY MODEL EVALUATION")

    for symbol in symbols:
        evaluate_model(symbol)

    print("\n" + "=" * 50)
    print("✅ Model evaluation completed!")
    print("=" * 50)