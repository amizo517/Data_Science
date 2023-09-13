import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define the stock symbol and fetch historical data from Yahoo Finance
symbol = 'AAPL'
start_date = '2010-01-01'
end_date = '2021-09-01'

data = yf.download(symbol, start=start_date, end=end_date)

# Select the 'Close' column for prediction
data = data[['Close']]

# Plot the historical stock price data
plt.figure(figsize=(12, 6))
plt.title(f'{symbol} Stock Price')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.plot(data['Close'])
plt.show()

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]

# Function to create sequences for the LSTM model
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 10  # Number of past days to use for prediction
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Evaluate the model on the test data
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Make predictions on the test data
predictions = model.predict(X_test)

# Inverse transform the predictions to get actual stock prices
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test)

# Plot the predicted vs. actual stock prices
plt.figure(figsize=(12, 6))
plt.title(f'{symbol} Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.plot(data.index[train_size+seq_length:], y_test, label='Actual Prices')
plt.plot(data.index[train_size+seq_length:], predictions, label='Predicted Prices')
plt.legend()
plt.show()
