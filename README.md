# 📈 CryptoForecastAI — Cryptocurrency Price Prediction

An AI-powered cryptocurrency price prediction web application built using **Python, TensorFlow, LSTM neural networks, and Streamlit**.

🌐 **Live Demo:** https://cryptoforecastai.streamlit.app/

🐙 **GitHub Repository:** https://github.com/suha-2004/CryptoPricePrediction

---

## 📌 Overview

**CryptoForecastAI** is a machine learning-based web application designed to analyze historical cryptocurrency price data and generate future price predictions using **Long Short-Term Memory (LSTM)** neural networks.

Cryptocurrency prices are time-series data, meaning that the current value can be analyzed along with previous values to identify patterns and trends. LSTM networks are well suited for this type of sequential data because they can learn relationships between observations over time.

This project implements a separate LSTM model for each supported cryptocurrency and provides an interactive Streamlit interface where users can explore historical prices and view model-generated forecasts.

---

## 🪙 Supported Cryptocurrencies

The application currently supports five cryptocurrencies:

| Cryptocurrency | Trading Pair |
| -------------- | ------------ |
| Bitcoin        | BTC/USDT     |
| Ethereum       | ETH/USDT     |
| Binance Coin   | BNB/USDT     |
| Litecoin       | LTC/USDT     |
| XRP            | XRP/USDT     |

---

## ✨ Features

### 📊 Historical Price Analysis

The application uses historical cryptocurrency market data to analyze price movements and trends.

### 🤖 LSTM-Based Prediction

A dedicated LSTM neural network is trained for each cryptocurrency to learn patterns from historical price sequences.

### 🔮 Future Price Forecasting

The trained models generate projected future cryptocurrency prices based on previously observed price patterns.

### 📈 Interactive Charts

The Streamlit application provides interactive visualizations for understanding:

* Historical price movements
* Actual vs predicted prices
* Prediction trends
* Future forecasts

### 🪙 Multiple Cryptocurrency Models

Instead of using one model for every cryptocurrency, the project trains separate models for:

* BTC
* ETH
* BNB
* LTC
* XRP

This allows each model to learn patterns specific to the selected cryptocurrency.

### 🌐 Web-Based Interface

The trained models are integrated into a Streamlit web application, allowing users to interact with the system directly from a browser.

---

# 🧠 Machine Learning Approach

The project follows a complete machine learning pipeline:

```text
Historical Cryptocurrency Data
            ↓
       Data Cleaning
            ↓
     Data Preprocessing
            ↓
      Normalization
            ↓
   Sequence Generation
            ↓
      Train / Test Split
            ↓
     LSTM Model Training
            ↓
      Model Evaluation
            ↓
    Future Price Forecast
            ↓
      Streamlit Web App
```

---

## 1️⃣ Data Collection

Historical cryptocurrency market data is collected using cryptocurrency market data sources.

The project uses **hourly price data** for the supported cryptocurrencies.

The main feature used for prediction is the historical **closing price**.

---

## 2️⃣ Data Preprocessing

Before training the neural network, the collected data is processed and prepared.

The preprocessing steps include:

* Selecting the closing price
* Organizing the data chronologically
* Checking for missing values
* Normalizing the price values
* Creating sequential input data
* Splitting the dataset into training and testing data

---

## 3️⃣ Feature Scaling

Cryptocurrency prices can have very different numerical ranges.

To make the data suitable for neural network training, the closing prices are normalized using **MinMaxScaler**.

The values are transformed into a smaller numerical range before being passed to the LSTM model.

After prediction, the values are converted back to their original price scale using the inverse transformation.

---

## 4️⃣ Sequence Generation

The model uses the previous **24 hourly observations** as input to predict the next price.

For example:

```text
Hour 1
Hour 2
Hour 3
...
Hour 24
   ↓
 LSTM Model
   ↓
Next Predicted Price
```

This allows the model to learn patterns from a sequence of previous prices.

---

# 🤖 LSTM Model Architecture

A separate LSTM model is trained for each cryptocurrency.

The model architecture is approximately:

```text
Input Sequence
      ↓
LSTM Layer
64 Units
      ↓
Dropout
      ↓
LSTM Layer
32 Units
      ↓
Dropout
      ↓
Dense Layer
16 Units
ReLU
      ↓
Output Layer
      ↓
Predicted Price
```

### Model Configuration

| Parameter         | Value              |
| ----------------- | ------------------ |
| Model Type        | LSTM               |
| Input Sequence    | Previous 24 hours  |
| First LSTM Layer  | 64 units           |
| Second LSTM Layer | 32 units           |
| Dense Layer       | 16 units           |
| Optimizer         | Adam               |
| Loss Function     | Mean Squared Error |
| Regularization    | Dropout            |
| Training          | Early Stopping     |

---

# 📊 Model Evaluation

The models are evaluated using regression metrics.

### Metrics Used

#### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted prices.

A lower MAE indicates that predictions are closer to the actual values.

#### RMSE — Root Mean Squared Error

Measures prediction error while giving greater weight to larger errors.

A lower RMSE indicates better prediction performance.

#### MAPE — Mean Absolute Percentage Error

Measures the average prediction error as a percentage.

A lower MAPE generally indicates that the predictions are closer to the actual values.

---

## 📈 Evaluation Results

The trained models achieved the following approximate results on the historical test data:

| Cryptocurrency |    MAE |   RMSE |  MAPE |
| -------------- | -----: | -----: | ----: |
| BTC            | 549.11 | 735.53 | 0.70% |
| ETH            |  21.23 |  27.43 | 0.87% |
| BNB            |   6.69 |  10.51 | 0.92% |
| LTC            |   0.60 |   0.82 | 1.17% |
| XRP            |   0.02 |   0.02 | 1.16% |

> **Note:** These metrics represent performance on historical held-out test data. They do not guarantee the same performance on future cryptocurrency prices.

---

# 🛠️ Technologies Used

## Programming Language

* **Python**

## Machine Learning & Deep Learning

* **TensorFlow**
* **Keras**
* **Scikit-learn**

## Data Processing

* **Pandas**
* **NumPy**

## Visualization

* **Plotly**
* **Matplotlib**

## Web Application

* **Streamlit**

## Deployment

* **Streamlit Community Cloud**

---

# 📂 Project Structure

```text
CryptoPricePrediction/
│
├── app.py
│
├── models/
│   ├── btc_model.keras
│   ├── eth_model.keras
│   ├── bnb_model.keras
│   ├── ltc_model.keras
│   └── xrp_model.keras
│
├── data/
│   └── cryptocurrency datasets
│
├── scalers/
│   └── saved scalers
│
├── notebooks/
│   └── model development
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

> The exact structure may change as the project is further developed.

---

# 💻 Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/suha-2004/CryptoPricePrediction.git
```

## 2. Open the Project

```bash
cd CryptoPricePrediction
```

## 3. Create a Virtual Environment

```bash
python -m venv venv
```

## 4. Activate the Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 🌐 Live Demo

You can access the deployed application here:

## 🚀 https://cryptoforecastai.streamlit.app/

The deployed application allows users to interact with the cryptocurrency prediction system directly through a web browser.

---

# 🔄 End-to-End Workflow

```text
                ┌───────────────────────┐
                │ Historical Crypto Data│
                └───────────┬───────────┘
                            ↓
                ┌───────────────────────┐
                │ Data Preprocessing    │
                └───────────┬───────────┘
                            ↓
                ┌───────────────────────┐
                │ MinMax Normalization  │
                └───────────┬───────────┘
                            ↓
                ┌───────────────────────┐
                │ 24-Hour Sequences     │
                └───────────┬───────────┘
                            ↓
                ┌───────────────────────┐
                │ LSTM Model Training   │
                └───────────┬───────────┘
                            ↓
                ┌───────────────────────┐
                │ Model Evaluation      │
                └───────────┬───────────┘
                            ↓
                ┌───────────────────────┐
                │ Future Forecasting    │
                └───────────┬───────────┘
                            ↓
                ┌───────────────────────┐
                │ Streamlit Application │
                └───────────────────────┘
```

---

# 🎯 Project Objectives

The main objectives of CryptoForecastAI are:

* To understand cryptocurrency time-series data
* To implement deep learning for sequential data
* To build LSTM-based forecasting models
* To train individual models for multiple cryptocurrencies
* To evaluate models using regression metrics
* To visualize historical and predicted prices
* To deploy a machine learning model as an interactive web application

---

# 💡 Why LSTM?

Cryptocurrency prices are sequential data because their values are recorded over time.

Traditional machine learning models may not naturally capture relationships between previous observations.

LSTM networks are designed to handle sequential information and can learn patterns from previous time steps.

In this project:

```text
Previous 24 Hours
       ↓
LSTM
       ↓
Next Price
```

The model learns from historical sequences and uses those patterns to estimate the next price.

---

# 🔮 Future Improvements

The project can be further improved by:

* Adding more cryptocurrencies
* Including Open, High, Low and Volume features
* Adding technical indicators such as RSI and MACD
* Comparing LSTM with GRU models
* Experimenting with CNN-LSTM architectures
* Exploring Transformer-based time-series models
* Adding real-time market data
* Implementing automatic model retraining
* Adding prediction confidence intervals
* Improving long-term forecasting
* Adding a model performance dashboard
* Adding more interactive visualizations

---

# ⚠️ Disclaimer

**CryptoForecastAI is developed for educational and research purposes only.**

Cryptocurrency markets are highly volatile and unpredictable. Machine learning models cannot guarantee future cryptocurrency prices or investment returns.

The forecasts generated by this application should **not be considered financial or investment advice**.

Users should conduct their own research and consult qualified financial professionals before making financial decisions.

---

# 👩‍💻 Author

## Suha

**B.Tech Computer Science & Engineering**

### 🔗 Connect

* 💼 LinkedIn: https://www.linkedin.com/in/suha-i-a-a769ab358/
* 🐙 GitHub: https://github.com/suha-2004

---

# ⭐ Project Links

| Resource             | Link                                               |
| -------------------- | -------------------------------------------------- |
| 🌐 Live Application  | https://cryptoforecastai.streamlit.app/            |
| 🐙 GitHub Repository | https://github.com/suha-2004/CryptoPricePrediction |
| 💼 LinkedIn          | https://www.linkedin.com/in/suha-i-a-a769ab358/    |

---

# ⭐ Support

If you find this project useful or interesting, consider giving the repository a ⭐ on GitHub.

Thank you for checking out **CryptoForecastAI!** 🚀
