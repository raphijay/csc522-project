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
from src.pre_processing import preprocess

__author__ = "Rithik Jain"
__credits__ = ["Thomas Price"]
__license__ = "No License"
__version__ = "1.0.0"
__maintainer__ = "Rithik Jain"
__email__ = "isrithikta@gmail.com"
__status__ = "Production"

class LSTM():

    def LSTMModel():
        usdinr_df         = pd.read_csv('../data/USDINRX.csv')
        forex_df          = pd.read_csv('../data/forex.csv')

        # Preprocessing the USDINR & Forex dataset
        cleaned_usdinr_df = preprocess.pre_p_usdinr(usdinr_df)
        cleaned_forex_df  = preprocess.pre_p_forex(forex_df)

        # Merge Datasets: Forex & USDINR
        merged_dataset    = preprocess.merge(cleaned_forex_df, cleaned_usdinr_df)

        # Split data into features (X) and target (y)
        X_train, X_test, y_train, y_test = preprocess.split(merged_dataset)

        # Normalize data using StandardScaler
        X_train, X_test = preprocess.transform(merged_dataset)

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
        for x in range(prediction_days, len(merged_dataset)):
            x_train.append(merged_dataset[x - prediction_days: x, 0])
            y_train.append(merged_dataset[x, 0])

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