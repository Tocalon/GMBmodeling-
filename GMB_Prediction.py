import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Step 1: Data Preparation
symbol = 'AAPL'
data = yf.download(symbol, start='2021-01-01', end='2022-01-01')['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)
X_train, y_train = [], []
for i in range(60, len(data_scaled)):
    X_train.append(data_scaled[i-60:i, 0])
    y_train.append(data_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Step 2: Model Architecture
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(units=50, return_sequences=False),
    Dense(units=25),
    Dense(units=1)
])

# Step 3: Model Training
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Step 4: Model Evaluation (if testing data is available)

# Step 5: Prediction
test_data = yf.download(symbol, start='2022-01-01', end='2022-01-15')['Close'].values.reshape(-1, 1)
scaled_test_data = scaler.transform(test_data)
X_test = []
for i in range(60, len(scaled_test_data)):
    X_test.append(scaled_test_data[i-60:i, 0])

# Add print statement to inspect the shape of X_test
print("Shape of X_test before reshaping:", np.array(X_test).shape)

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(test_data[60:], label='True Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.title('Stock Price Prediction for ' + symbol)
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()
