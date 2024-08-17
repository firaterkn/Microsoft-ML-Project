import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, GRU
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# Load the dataset
df = web.DataReader('MSFT', data_source='stooq', start='2012-01-01', end='2020-05-01')
df = df.iloc[::-1]
data = df.filter(["Close"])
dataset = data.values
training_data_length = math.ceil(len(dataset) * .8)

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Prepare training and testing data sets
train_data = scaled_data[0:training_data_length, :]
x_train = []
y_train = []

# Create the training dataset
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Convert data for LSTM
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Identify the GRU model
model_gru = Sequential()
model_gru.add(GRU(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model_gru.add(GRU(50, return_sequences=False))
model_gru.add(Dense(25))
model_gru.add(Dense(1))

# Compile the model
model_gru.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model_gru.fit(x_train, y_train, batch_size=1, epochs=1)

# Prepare the test data set
test_data = scaled_data[training_data_length - 60:, :]
x_test = []
y_test = dataset[training_data_length:, :]

# Convert the test data set to the appropriate format
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Make prediction
predictions_gru = model_gru.predict(x_test)
predictions_gru = scaler.inverse_transform(predictions_gru)

# RMSE calculation for comparison with original data
rmse = np.sqrt(np.mean(((predictions_gru - y_test)**2)))

# Visualize forecasts(guesses) and the original data
train = data[:training_data_length]
valid = data[training_data_length:]
valid['Predictions'] = predictions_gru

plt.figure(figsize=(10, 5))
plt.title('GRU Model')
plt.xlabel('Date', fontsize=15)
plt.ylabel('Closing Prices in USD ($)', fontsize=15)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Valid', 'Predictions'], loc='lower right')
plt.show()