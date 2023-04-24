import math
import numpy as np
import sklearn
import pytest

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_wine
from pytest import raises
from src.rnnModel import CustomLSTM

RANDOM_SEED = 522

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

    @pytest.fixture
    def static_data(self):
        np.random.seed(RANDOM_SEED)
        return np.array([np.random.randint(1, 10 + 1, size = 3) for i in range(100)])

    @pytest.fixture
    def wine_data(self):
        wine_data = load_wine()
        wine_features = wine_data.data
        wine_target = wine_data.target

        wine_features_train, wine_features_test, wine_target_train, wine_target_test = train_test_split(wine_features, wine_target, test_size = 0.2, random_state = RANDOM_SEED)
        return {
            'training': { 'features': wine_features_train, 'target': wine_target_train },
            'testing': { 'features': wine_features_test, 'target': wine_target_test }
        }

    def test_train_predict(self):
        lstm = CustomLSTM((10, 1), 32, 16, (1,), 10, 32)

        # Generate some random training and test data
        x_train = np.random.rand(10, 1)
        y_train = np.random.rand(10, 1)
        x_test = np.random.rand(10, 1)

        # Train the model and make predictions
        lstm.train(x_train, y_train)
        preds = lstm.predict(x_test)

        ##
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

        assert preds.shape == (10, 1, 1)

    def test_train_predict_wine(self):
        lstm = CustomLSTM((10, 13), 32, 16, (1,), 10, 32)

        # Load the wine dataset
        X, y = load_wine(return_X_y=True)

        # Reshape the data to have 10 samples with 13 features (load_wine has 13 features)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False, random_state=25)


        X_reshaped_train = X_train[:10].reshape(1, 10, 13)
        X_reshaped_test = X_test[:10].reshape(1, 10, 13)
        y_reshaped_train = y_train[:10].reshape(1, 10, 1)
        y_reshaped_test = y_test[:10].reshape(1, 10, 1)


        # Train the model and make predictions
        lstm.train(X_reshaped_train, y_reshaped_train)
        preds = lstm.predict(X_reshaped_train)

        assert preds.shape == (1, 10, 1)

        preds = preds.reshape(10, 1)

        assert preds.shape == (10, 1)

        # currently gives 0.012754151676780703 (does that make sense?)
        #assert math.sqrt(sklearn.metrics.mean_squared_error(y_reshaped_train.reshape(-1), preds)) == 1
        preds = lstm.predict(X_reshaped_test)

        assert preds.shape == (1, 10, 1)

        preds = preds.reshape(10, 1)

        assert preds.shape == (10, 1)

        # current gives 2.002606222416647 (not sure if it makes sense)
        # assert math.sqrt(sklearn.metrics.mean_squared_error(y_reshaped_test.reshape(-1), preds)) == 0



    def test_predict_without_train_raises_exception(self):
        lstm = CustomLSTM((10, 1), 32, 16, (1,), 10, 32)
        x_test = np.random.rand(10, 10, 1)
        with raises(Exception):
            lstm.predict(x_test)