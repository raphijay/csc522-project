import pytest
import numpy as np

from src.rnnModel import LSTM
from sklearn.model_selection import train_test_split
import yfinance as yf

@pytest.fixture
def forex_data():
    forex_data = yf.download(tickers = 'USDINR=X' ,period ='1d', interval = '15m')

    forex_features = forex_data.drop('Close', axis=1).values
    forex_target   = forex_data['Close'].values

    forex_features_train, forex_features_test, forex_target_train, forex_target_test = train_test_split(forex_features, forex_target, test_size = 0.2)
    return {
        'training': { 'features': forex_features_train, 'target': forex_target_train },
        'testing': { 'features': forex_features_test, 'target': forex_target_test }
    }

def test_ltm_train(forex_data):
    lstm = LSTM(input_shape = (forex_data['training']['features'].shape[1], 1), lstm_units = 3, dense_units = 1, output_shape = 1, epochs = 10, batch_size = 32)

    lstm.train(forex_data['training']['features'], forex_data['training']['target'])

    assert(len(lstm.get_networks())) == 1

def test_lstm_predict(forex_data):
    lstm = LSTM(input_shape = (forex_data['training']['features'].shape[1], 1), lstm_units = 3, dense_units = 1, output_shape = 1, epochs = 10, batch_size = 32)

    lstm.train(forex_data['training']['features'], forex_data['training']['target'])

    predicted_target = lstm.predict(forex_data['testing']['features'])

    assert len(predicted_target) == len(forex_data['testing']['target'])
    assert round(lstm.calculate_rmse_of_predicted(forex_data['testing']['target']), 4) == 54.4082 # For random seed 522, this will always match

def test_lstm_no_train(forex_data):
    lstm = LSTM(input_shape = (forex_data['training']['features'].shape[1], 1), lstm_units = 3, dense_units = 1, output_shape = 1, epochs = 10, batch_size = 32)

    with pytest.raises(Exception):
        lstm.predict(forex_data['testing']['features'])
    with pytest.raises(Exception):
        lstm.predict(forex_data['testing']['target'])
