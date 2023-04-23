import pytest
import numpy as np
from sklearn.metrics import mean_squared_error

from src.rnnModel import CustomLSTM
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
    lstm = CustomLSTM(input_shape = (forex_data['training']['features'].shape[1], 1), lstm_units = 3, dense_units = 1, output_shape = 1, epochs = 10, batch_size = 32)

    lstm.train(forex_data['training']['features'], forex_data['training']['target'])

def test_lstm_predict(forex_data):
    lstm = CustomLSTM(input_shape = (forex_data['training']['features'].shape[1], 1), lstm_units = 3, dense_units = 1, output_shape = 1, epochs = 10, batch_size = 32)

    lstm.train(forex_data['training']['features'], forex_data['training']['target'])
    pred = []
    print(lstm.predict([1]))
    for i in forex_data['testing']['features']:
        print(lstm.predict([i]))
        pred.append(lstm.predict([i])[0][0][0])

    print(pred)
    assert len(pred) == len(forex_data['testing']['target'])
    assert mean_squared_error(forex_data['testing']['target'], pred) > 0

def test_lstm_no_train(forex_data):
    lstm = CustomLSTM(input_shape = (forex_data['training']['features'].shape[1], 1), lstm_units = 3, dense_units = 1, output_shape = 1, epochs = 10, batch_size = 32)

    with pytest.raises(Exception):
        lstm.predict(forex_data['testing']['features'])
    with pytest.raises(Exception):
        lstm.predict(forex_data['testing']['target'])
