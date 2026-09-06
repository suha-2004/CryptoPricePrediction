import numpy as np
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# Five cryptocurrencies
symbols = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "LTCUSDT",
    "XRPUSDT"
]


# Create models folder
os.makedirs("models", exist_ok=True)


def train_model(symbol):

    print("\n" + "=" * 50)
    print(f"Training model for {symbol}")
    print("=" * 50)

    # Load processed data
    X_train = np.load(f"models/{symbol}_X_train.npy")
    y_train = np.load(f"models/{symbol}_y_train.npy")

    X_test = np.load(f"models/{symbol}_X_test.npy")
    y_test = np.load(f"models/{symbol}_y_test.npy")

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test :", X_test.shape)
    print("y_test :", y_test.shape)

    # Build LSTM model
    model = Sequential([
        
        LSTM(
            64,
            return_sequences=True,
            input_shape=(X_train.shape[1], X_train.shape[2])
        ),

        Dropout(0.2),

        LSTM(32),

        Dropout(0.2),

        Dense(16, activation="relu"),

        Dense(1)
    ])

    # Compile model
    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    # Display model structure
    model.summary()

    # Stop training if validation loss stops improving
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    # Train
    history = model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate
    test_loss = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print(f"\nTest Loss: {test_loss}")

    # Save model
    model_path = f"models/{symbol}_lstm.keras"
    model.save(model_path)

    print(f"Model saved: {model_path}")


# Train all five models
if __name__ == "__main__":

    for symbol in symbols:
        train_model(symbol)

    print("\n" + "=" * 50)
    print("✅ All five LSTM models trained successfully!")
    print("=" * 50)