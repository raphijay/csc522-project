import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import mean_squared_error
from pytest import raises

from src.rnnModel import CustomLSTM

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import mean_squared_error
from pytest import raises

from src.rnnModel import CustomLSTM

class TestCustomLSTM:
    ##
    #  CustomLSTM model is designed to predict a single value, hence the output_shape should be (1,)
    # The output shape of a Keras model is a tuple indicating the shape of the output tensor for each
    # layer in the model. In this case, the output of the Dense layer is a tensor of shape (None, 10
    # 1), where None is the batch size and 10 is the number of time steps. The final dimension of size
    # 1 corresponds to the predicted value for each time step.
    ##
    def test_init(self):
        lstm = CustomLSTM((10, 1), 32, 16, (1,), 10, 32)
        assert lstm.input_shape == (10, 1)
        assert lstm.lstm_units == 32
        assert lstm.dense_units == 16
        assert lstm.output_shape == (1,)
        assert lstm.epochs == 10
        assert lstm.batch_size == 32

    def test_train_predict(self):
        lstm = CustomLSTM((10, 1), 32, 16, (1,), 10, 32)

        # Generate some random training and test data
        x_train = np.random.rand(100, 10, 1)
        y_train = np.random.rand(100, 10, 1)  # y_train should have shape (100, 10, 1)
        x_test = np.random.rand(10, 10, 1)

        # Train the model and make predictions
        lstm.train(x_train, y_train)
        preds = lstm.predict(x_test)

        ##
        # Check that the output shape is correct
        # output shape should be (10, 10, 1) but getting (10, 10, 16) since the model's final layer has
        # only one output unit.
        #
        # Layer (type)                Output Shape              Param #
        # =================================================================
        #  lstm (LSTM)                 (None, 10, 32)            4352
        #
        #  dropout (Dropout)           (None, 10, 32)            0
        #
        #  lstm_1 (LSTM)               (None, 10, 32)            8320
        #
        #  dropout_1 (Dropout)         (None, 10, 32)            0
        #
        #  lstm_2 (LSTM)               (None, 10, 32)            8320
        #
        #  dropout_2 (Dropout)         (None, 10, 32)            0
        #
        #  dense (Dense)               (None, 10, 1)             33
        #
        # =================================================================
        # Total params: 21,025
        # Trainable params: 21,025
        # Non-trainable params: 0
        ##
        assert preds.shape == (10, 10, 1)

    def test_predict_without_train_raises_exception(self):
        lstm = CustomLSTM((10, 1), 32, 16, (1,), 10, 32)
        x_test = np.random.rand(10, 10, 1)
        with raises(Exception):
            lstm.predict(x_test)