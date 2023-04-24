import pytest
import numpy as np
from sklearn.neural_network import MLPRegressor

from src.rand_network_ensemble import RandomNetworkEnsemble
from src.lstm_rnn import CustomLSTM
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

RANDOM_SEED = 522

@pytest.fixture
def static_data():
    np.random.seed(RANDOM_SEED)
    return np.array([np.random.randint(1, 10 + 1, size = 3) for i in range(100)])

@pytest.fixture
def wine_data():
    wine_data = load_wine()
    wine_features = wine_data.data
    wine_target = wine_data.target

    wine_features_train, wine_features_test, wine_target_train, wine_target_test = train_test_split(wine_features, wine_target, test_size = 0.2, random_state = RANDOM_SEED)
    return {
        'training': { 'features': wine_features_train, 'target': wine_target_train },
        'testing': { 'features': wine_features_test, 'target': wine_target_test }
    }

@pytest.fixture
def wine_data_5050():
    wine_data = load_wine()
    wine_features = wine_data.data
    wine_target = wine_data.target

    wine_features_train, wine_features_test, wine_target_train, wine_target_test = train_test_split(wine_features, wine_target, test_size = 0.5, random_state = RANDOM_SEED)
    return {
        'training': { 'features': wine_features_train, 'target': wine_target_train },
        'testing': { 'features': wine_features_test, 'target': wine_target_test }
    }

@pytest.fixture
def sample_mlp_regressor_args():
    return { 'activation': 'identity', 'hidden_layer_sizes': (50,), 'learning_rate_init': 0.01, 'random_state': RANDOM_SEED } 

@pytest.fixture
def sample_lstm_rnn_args(wine_data_5050):
    training = wine_data_5050['training']['features']
    return { 'input_shape': (len(training), len(training[0])), 'lstm_units': 32, 'dense_units': 16, 'output_shape': (1,) }

def test_custom_rf_regressor_constructor_standard_baseline_rnn(sample_mlp_regressor_args):
    try:
        RandomNetworkEnsemble(num_networks = 50, base_nn_model = MLPRegressor, model_args = sample_mlp_regressor_args, random_seed = RANDOM_SEED)
    except ValueError:
        assert False, 'Valid constructor arguments with MLPRegressor still caused the constructor to fail' 

def test_custom_rf_regressor_constructor_standard_lstm_rnn(sample_lstm_rnn_args):
    try:
        RandomNetworkEnsemble(num_networks = 50, base_nn_model = CustomLSTM, model_args = sample_lstm_rnn_args, random_seed = RANDOM_SEED)
    except ValueError:
        assert False, 'Valid constructor arguments with LSTM RNN still caused the constructor to fail' 

def test_custom_rf_regressor_constructor_bad_num_networks(sample_mlp_regressor_args):
    with pytest.raises(ValueError):
        RandomNetworkEnsemble(num_networks = 1, base_nn_model = MLPRegressor, model_args = sample_mlp_regressor_args, random_seed = RANDOM_SEED)

def test_custom_rf_regressor_constructor_no_random_seed(sample_mlp_regressor_args):
    try:
        RandomNetworkEnsemble(num_networks = 50, base_nn_model = MLPRegressor, model_args = sample_mlp_regressor_args)
    except ValueError:
        assert False, 'No random seed constructor argument still caused the constructor to fail'

def test_custom_rf_regressor_make_bootstraps_baseline_rnn(static_data, sample_mlp_regressor_args):
    crfr = RandomNetworkEnsemble(num_networks = 3, base_nn_model = MLPRegressor, model_args = sample_mlp_regressor_args, random_seed = RANDOM_SEED)
    bootstrap_samples = crfr.make_bootstraps(static_data)
    print(bootstrap_samples)
    assert len(bootstrap_samples) == 3

def test_custom_rf_regressor_make_bootstraps_lstm_rnn(static_data, sample_lstm_rnn_args):
    crfr = RandomNetworkEnsemble(num_networks = 3, base_nn_model = CustomLSTM, model_args = sample_lstm_rnn_args, random_seed = RANDOM_SEED)
    bootstrap_samples = crfr.make_bootstraps(static_data)
    print(bootstrap_samples)
    assert len(bootstrap_samples) == 3

def test_custom_rf_regressor_train_baseline_rnn(wine_data, sample_mlp_regressor_args):
    crfr = RandomNetworkEnsemble(num_networks = 3, base_nn_model = MLPRegressor, model_args = sample_mlp_regressor_args, random_seed = RANDOM_SEED)
    crfr.train(wine_data['training']['features'], wine_data['training']['target'])
    assert(len(crfr.get_networks())) == 3

def test_custom_rf_regressor_train_lstm_rnn(wine_data_5050, sample_lstm_rnn_args):
    crfr = RandomNetworkEnsemble(num_networks = 3, base_nn_model = CustomLSTM, model_args = sample_lstm_rnn_args, random_seed = RANDOM_SEED)
    crfr.train(wine_data['training']['features'], wine_data_5050['training']['target'])
    assert(len(crfr.get_networks())) == 3

def test_custom_rf_regressor_predict_baseline_rnn(wine_data, sample_mlp_regressor_args):
    crfr = RandomNetworkEnsemble(num_networks = 3, base_nn_model = MLPRegressor, model_args = sample_mlp_regressor_args, random_seed = RANDOM_SEED)
    crfr.train(wine_data['training']['features'], wine_data['training']['target'])
    predicted_target = crfr.predict(wine_data['testing']['features'])
    assert len(predicted_target) == len(wine_data['testing']['target'])
    assert round(crfr.calculate_rmse_of_predicted(wine_data['testing']['target']), 4) == 19.2517 # For random seed 522 and the sample args, this will always match

def test_custom_rf_regressor_predict_lstm_rnn(wine_data_5050, sample_lstm_rnn_args):
    crfr = RandomNetworkEnsemble(num_networks = 3, base_nn_model = CustomLSTM, model_args = sample_lstm_rnn_args, random_seed = RANDOM_SEED)
    crfr.train(wine_data_5050['training']['features'], wine_data_5050['training']['target'])
    predicted_target = crfr.predict(wine_data_5050['testing']['features'])
    assert len(predicted_target) == len(wine_data_5050['testing']['target'])
    assert round(crfr.calculate_rmse_of_predicted(wine_data_5050['testing']['target']), 4) < 2

def test_custom_rf_regressor_predict_no_train(wine_data, sample_mlp_regressor_args):
    crfr = RandomNetworkEnsemble(num_networks = 3, base_nn_model = MLPRegressor, model_args = sample_mlp_regressor_args, random_seed = RANDOM_SEED)
    with pytest.raises(Exception):
        crfr.predict(wine_data['testing']['features'])
    with pytest.raises(Exception):
        crfr.predict(wine_data['testing']['target'])
