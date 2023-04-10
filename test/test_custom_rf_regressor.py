import pytest
import numpy as np

from src.custom_rf_regressor import CustomRandomForestRegressor
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

def test_custom_rf_regressor_constructor_standard():
    try:
        CustomRandomForestRegressor(num_networks = 50, activation = 'identity', hidden_layer_sizes = (50,), learning_rate_init = 0.01, random_seed = RANDOM_SEED)
    except ValueError:
        assert False, 'Valid constructor arguments still caused the constructor to fail' 

def test_custom_rf_regressor_constructor_bad_num_networks():
    with pytest.raises(ValueError):
        CustomRandomForestRegressor(num_networks = 1, activation = 'identity', hidden_layer_sizes = (50,), learning_rate_init = 0.01, random_seed = RANDOM_SEED)

def test_custom_rf_regressor_constructor_bad_activation():
    with pytest.raises(ValueError):
        CustomRandomForestRegressor(num_networks = 50, activation = 'invalid', hidden_layer_sizes = (50,), learning_rate_init = 0.01, random_seed = RANDOM_SEED)

def test_custom_rf_regressor_constructor_bad_hidden_layer_sizes():
    with pytest.raises(ValueError):
        CustomRandomForestRegressor(num_networks = 50, activation = 'identity', hidden_layer_sizes = (), learning_rate_init = 0.01, random_seed = RANDOM_SEED)
    with pytest.raises(ValueError):
        CustomRandomForestRegressor(num_networks = 50, activation = 'identity', hidden_layer_sizes = (1,), learning_rate_init = 0.01, random_seed = RANDOM_SEED)

def test_custom_rf_regressor_constructor_no_random_seed():
    try:
        CustomRandomForestRegressor(num_networks = 50, activation = 'identity', hidden_layer_sizes = (100,), learning_rate_init = 0.01)
    except ValueError:
        assert False, 'No random seed constructor argument still caused the constructor to fail'

def test_custom_rf_regressor_make_bootstraps(static_data):
    crfr = CustomRandomForestRegressor(num_networks = 3, activation = 'identity', hidden_layer_sizes = (3,), learning_rate_init = 0.01, random_seed = RANDOM_SEED)
    bootstrap_samples = crfr.make_bootstraps(static_data)
    print(bootstrap_samples)
    assert len(bootstrap_samples) == 3

def test_custom_rf_regressor_train(wine_data):
    crfr = CustomRandomForestRegressor(num_networks = 3, activation = 'identity', hidden_layer_sizes = (3,), learning_rate_init = 0.01, random_seed = RANDOM_SEED)
    crfr.train(wine_data['training']['features'], wine_data['training']['target'])
    assert(len(crfr.get_trees())) == 3

def test_custom_rf_regressor_predict(wine_data):
    crfr = CustomRandomForestRegressor(num_networks = 3, activation = 'identity', hidden_layer_sizes = (3,), learning_rate_init = 0.01, random_seed = RANDOM_SEED)
    crfr.train(wine_data['training']['features'], wine_data['training']['target'])
    predicted_target = crfr.predict(wine_data['testing']['features'])
    assert len(predicted_target) == len(wine_data['testing']['target'])
    assert round(crfr.calculate_rmse_of_predicted(wine_data['testing']['target']), 4) == 54.4082 # For random seed 522, this will always match

def test_custom_rf_regressor_predict_no_train(wine_data):
    crfr = CustomRandomForestRegressor(num_networks = 3, activation = 'identity', hidden_layer_sizes = (3,), learning_rate_init = 0.01, random_seed = RANDOM_SEED)
    with pytest.raises(Exception):
        crfr.predict(wine_data['testing']['features'])
    with pytest.raises(Exception):
        crfr.predict(wine_data['testing']['target'])
