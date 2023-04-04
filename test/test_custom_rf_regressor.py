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
        CustomRandomForestRegressor(n_trees = 50, max_depth = 3, min_samples_split = 2, random_seed = RANDOM_SEED)
    except ValueError:
        assert False, 'Valid constructor arguments still caused the constructor to fail' 

def test_custom_rf_regressor_constructor_bad_n_trees():
    with pytest.raises(ValueError):
        CustomRandomForestRegressor(n_trees = 1, max_depth = 3, min_samples_split = 2, random_seed = RANDOM_SEED)

def test_custom_rf_regressor_constructor_bad_max_depth():
    with pytest.raises(ValueError):
        CustomRandomForestRegressor(n_trees = 50, max_depth = 0, min_samples_split = 2, random_seed = RANDOM_SEED)

def test_custom_rf_regressor_constructor_bad_min_samples_split():
    with pytest.raises(ValueError):
        CustomRandomForestRegressor(n_trees = 1, max_depth = 3, min_samples_split = 1, random_seed = RANDOM_SEED)

def test_custom_rf_regressor_constructor_no_random_seed():
    try:
        CustomRandomForestRegressor(n_trees = 50, max_depth = 3, min_samples_split = 2)
    except ValueError:
        assert False, 'No random seed constructor argument still caused the constructor to fail'

def test_custom_rf_regressor_make_bootstraps(static_data):
    crfr = CustomRandomForestRegressor(n_trees = 3, max_depth = 2, min_samples_split = 2, random_seed = RANDOM_SEED)
    bootstrap_samples = crfr.make_bootstraps(static_data)
    print(bootstrap_samples)
    assert len(bootstrap_samples) == 3

def test_custom_rf_regressor_train(wine_data):
    crfr = CustomRandomForestRegressor(n_trees = 3, max_depth = 2, min_samples_split = 2, random_seed = RANDOM_SEED)
    crfr.train(wine_data['training']['features'], wine_data['training']['target'])
    assert(len(crfr.get_trees())) == 3

def test_custom_rf_regressor_predict(wine_data):
    crfr = CustomRandomForestRegressor(n_trees = 3, max_depth = 2, min_samples_split = 2, random_seed = RANDOM_SEED)
    crfr.train(wine_data['training']['features'], wine_data['training']['target'])
    predicted_target = crfr.predict(wine_data['testing']['features'])
    assert len(predicted_target) == len(wine_data['testing']['target'])
    assert round(crfr.calculate_rmse_of_predicted(wine_data['testing']['target']), 4) == 0.3332 # For random seed 522, this will always match

def test_custom_rf_regressor_predict_no_train(wine_data):
    crfr = CustomRandomForestRegressor(n_trees = 3, max_depth = 2, min_samples_split = 2, random_seed = RANDOM_SEED)
    with pytest.raises(Exception):
        crfr.predict(wine_data['testing']['features'])
    with pytest.raises(Exception):
        crfr.predict(wine_data['testing']['target'])
