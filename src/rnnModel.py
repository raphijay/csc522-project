from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

__author__ = "Rithik Jain"
__credits__ = ["Thomas Price"]
__license__ = "No License"
__version__ = "1.0.0"
__maintainer__ = "Rithik Jain"
__email__ = "isrithikta@gmail.com"
__status__ = "Production"

##
# LSTM Model
##
class LSTM():

    ##
    # Creates an LSTM model for sequential data. The input parameters are:
    # input_shape:  A tuple representing the shape of the input data, excluding the batch size. For
    #               example, (timesteps, features) for a sequence of timesteps with feature vectors of
    #               a certain size.
    # lstm_units:   An integer representing the number of units in the LSTM layer.
    # dense_units:  An integer representing the number of units in the dense (fully connected) output
    #               layer.
    # output_shape: A tuple representing the shape of the output data, excluding the batch size.
    # epochs:       An integer representing the number of epochs to train the model for.
    # batch_size:   An integer representing the batch size to use during training.
    ##
    def __init__(self, input_shape, lstm_units, dense_units, output_shape, epochs = 10, batch_size = 32):
        self.model = None
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.output_shape = output_shape
        self.epochs = epochs
        self.batch_size = batch_size

    ##
    # Have LSTM, Dropout layers as a set of 3. A dense layer for forex prediction.
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

    ##
    # Trains the LSTM model on the provided input and target data. It first checks if the model has
    # been built or not by checking if self.model is None. If the model has not been built, it builds
    # the model using build_model method. It then trains the model using the fit method with the
    # provided input and target data along with the number of epochs and batch size.
    ##
    def train(self, x_train, y_train):
        if self.model is None:
            self.build_model()
        self.model.fit(x_train, y_train, epochs = self.epochs, batch_size = self.batch_size)

    ##
    # Takes in a set of test data (x_test) and returns the predictions made by the LSTM model. If the
    # model has not been trained before, an exception will be raised.
    ##
    def predict(self, x_test):
        if self.model is None:
            raise Exception('You must train the LSTM model first before it can predict!')
        return self.model.predict(x_test)