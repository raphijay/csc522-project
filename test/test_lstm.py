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

        # Check that the output shape is correct
        assert preds.shape == (10, 10, 16)  # output shape should be (10, 10, 16)

    def test_predict_without_train_raises_exception(self):
        lstm = CustomLSTM((10, 1), 32, 16, (1,), 10, 32)
        x_test = np.random.rand(10, 10, 1)
        with raises(Exception):
            lstm.predict(x_test)