import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import yfinance as yf

from pandas_datareader import data as pdr
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from git.csc522-project.src.pre_processing import preprocess

__author__ = "Rithik Jain"
__credits__ = ["Thomas Price"]
__license__ = "No License"
__version__ = "1.0.0"
__maintainer__ = "Rithik Jain"
__email__ = "isrithikta@gmail.com"
__status__ = "Production"

# Convert date column to datetime format
forex_data['date'] = pd.to_datetime(forex_data['date'])

# Sort data by date
forex_data = forex_data.sort_values('date')

# Split data into training and testing sets based on a date threshold
date_threshold = pd.to_datetime('2022-01-01')
train_data = forex_data[forex_data['date'] < date_threshold]
test_data = forex_data[forex_data['date'] >= date_threshold]

# Split data into features (X) and target (y)
X_train = train_data.drop('target', axis=1)
y_train = train_data['target']
X_test = test_data.drop('target', axis=1)
y_test = test_data['target']

# Normalize data using MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

##
# How many days to see in the past (8036 days)
# Increasing number of days to predict stock for long term
# Decrease number of days to predict stock for short term with high sensitivity
##
prediction_days     = 10

# Define two empty lists. Prepare the training data
x_train             = []
y_train             = []

##
# Start with 60 to length of scaled data (last index)
# Add a value to x_train through each iteration
# x_train upto 60 values
# y_train would be 61st value
##
for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days: x, 0])
    y_train.append(scaled_data[x, 0])

# Convert to numpy arrays. Reshape x_train to work with neural network.
x_train, y_train    = np.array(x_train), np.array(y_train)
x_train             = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train

# Build the model
model = Sequential()

##
# Have LSTM, Dropout layers as a set of 3. A dense layer for stock prediction.
# More units and layers would increase training time. Overfit if too many layers are used.
# return_sequences would be true as it would feed back info and not just feed forward like dense
# layer.
##
model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = 1))

##
# Compile data with optimizer 'adam' & loss function ''
# Epochs 30 means that model would see same data 30 times
# 20 batch size means that model would see 20 units at once
##
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
history = model.fit(x_train, y_train, epochs = 30, batch_size = 20)


''' Test the model accuracy on existing data '''

# Load test that model hasn't seen before in training
test_start          = dt.datetime(2023, 1, 1)
test_end            = dt.datetime.now()

test_data           = pdr.get_data_yahoo(company_ticker, test_start, test_end)
stock_prices        = test_data['Close'].values

total_dataset       = pd.concat((data['Close'], test_data['Close']), axis = 0)

model_inputs        = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs        = model_inputs.reshape(-1, 1)
model_inputs        = scaler.transform(model_inputs)

# Make predictions on test data
x_test              = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days: x, 0])

x_test              = np.array(x_test)
x_test              = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction_prices   = model.predict(x_test)
prediction_prices   = scaler.inverse_transform(prediction_prices)

# Calculate RMSE value
rmse = math.sqrt(mean_squared_error(stock_prices, prediction_prices))
print(f"RMSE value: {rmse:.2f}")

# Plot the test predictions
plt.plot(stock_prices, color = "black", label = f"Actual {company_ticker} Price")
plt.plot(prediction_prices, color = "green", label = f"Predicted {company_ticker} Price")
plt.title(f"{company_ticker} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company_ticker} Share Price')
plt.legend()
plt.show()

# # Plot the RMSE value
# plt.plot(rmse, color="red", label="RMSE value")
# plt.title(f"RMSE value for {company_ticker} Share Price Prediction")
# plt.xlabel('Time')
# plt.ylabel('RMSE value')
# plt.legend()
# plt.show()

# Predict next day
live_data           = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs + 1), 0]]
live_data           = np.array(live_data)
live_data           = np.reshape(live_data, (live_data.shape[0], live_data.shape[1], 1))

prediction          = model.predict(live_data)
prediction          = scaler.inverse_transform(prediction)
print(f"Predicted Price: {prediction}")