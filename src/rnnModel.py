from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

__author__ = "Rithik Jain"
__credits__ = ["Thomas Price"]
__license__ = "No License"
__version__ = "1.0.0"
__maintainer__ = "Rithik Jain"
__email__ = "isrithikta@gmail.com"
__status__ = "Production"

class LSTM():

    def __init__(self, input_shape, lstm_units, dense_units, output_shape, epochs = 10, batch_size = 32):
        self.model = None
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.output_shape = output_shape
        self.epochs = epochs
        self.batch_size = batch_size

    ##
    # Have LSTM, Dropout layers as a set of 3. A dense layer for stock prediction.
    # More units and layers would increase training time. Overfit if too many layers are used.
    # return_sequences would be true as it would feed back info and not just feed forward like dense
    # layer.
    ##
    def lstm_model(self):
        self.model = Sequential()
        self.model.add(LSTM(units = self.lstm_units, return_sequences = True, input_shape = self.input_shape))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units = self.lstm_units, return_sequences = True, input_shape = self.input_shape))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units = self.lstm_units, return_sequences = True, input_shape = self.input_shape))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units = self.dense_units, activation='relu'))
        self.model.add(Dense(units = self.output_shape, activation='linear'))
        self.model.compile(optimizer = 'adam', loss = 'mse')

    def train(self, x_train, y_train):
        if self.model is None:
            self.build_model()
        self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size)

    def predict(self, x_test):
        if self.model is None:
            raise Exception('You must train the LSTM model first before it can predict!')
        return self.model.predict(x_test)