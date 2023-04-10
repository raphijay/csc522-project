'''
    A representation for an ensemble of neural networks, custom to the needs of this project.
    This was originally based off of a custom random forest regressor, from the reference implmentations:
        * https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
        * https://insidelearningmachines.com/build-a-random-forest-in-python/
    
    The need to change to an ensemble of neural networks as opposed to decision trees
    was due to the application of neural networks in our project.
'''

import numpy as np

from sklearn.base import clone
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error


class RandomNetworkEnsemble():
    
    # Constructor
    def __init__(self, num_networks, activation: str = 'relu', hidden_layer_sizes = (100,), learning_rate_init: float = 0.001, random_seed: int = None):
        self.networks              = []
        self.resampler          = np.random
        self.last_predicted     = None
        self.seed               = None

        if (activation not in ['identity', 'logistic', 'tanh', 'relu']):
            raise ValueError('The provided activation function is not valid!')
        self.activation            = activation
        if (num_networks < 2):
            raise ValueError('Cannot have random network ensemble with less than 2 networks!')
        self.num_networks          = num_networks
        if (len(hidden_layer_sizes) < 1 or hidden_layer_sizes[0] < 2):
            raise ValueError('Hidden layer sizes argument is not valid!')
        self.hidden_layer_sizes    = hidden_layer_sizes
        if (learning_rate_init <= 0 or learning_rate_init > 1):
            raise ValueError('Not allowed to have a learning rate of 0 or less or greater than 1!')
        self.learning_rate_init    = learning_rate_init
        if (random_seed != None):
            # Recommended way to seed the resampler
            self.seed = random_seed
            self.resampler = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(random_seed)))

    # Getter for trees
    def get_trees(self):
        return self.networks

    # Public bootstrapping function
    def make_bootstraps(self, data: np):
        # Get list of indices from the np.array
        idx = [i for i in range(data.shape[0])]
        # Loop through the required number of bootstraps, mapping each bootstrapped sample to a key
        return { 'boot_' + str(network_num) : data[np.random.choice(idx, size=data.shape[0]), :] for network_num in range(self.num_networks)}
    
    def train(self, feature_train: list, target_train: list):
        # Create the training instances by combining the feature and target columns
        training_data = np.concatenate((feature_train, target_train.reshape(-1,1)), axis=1)
        # Bootstrap the training instances
        bootstraps = self.make_bootstraps(training_data)
        # The base unit for the Random Forest - MLPRegressor
        # TODO: Change this baseline neural network to be the custom RNN once that is ready
        nn_m = MLPRegressor(
            activation = self.activation,
            hidden_layer_sizes = self.hidden_layer_sizes,
            learning_rate_init = self.learning_rate_init,
            random_state = self.seed
        )
        # Fit each new tree to the features and target from each bootstrap sample
        # bootstraps[b][:, :-1] is the current bootstrap's features
        # bootstraps[b][:, -1] is the current bootstrap's target
        # Decision tree's fit method requires both to work
        self.networks = [clone(nn_m).fit(bootstraps[b][:, :-1], bootstraps[b][:, -1].reshape(-1, 1).ravel()) for b in bootstraps]
        return

    # Public function to predict from the ensemble
    def predict(self, data_to_predict: list):
        if len(self.networks) == 0:
            raise Exception('You must train the random network ensemble first before it can predict!')
        # Loop through each decision tree, doing its native predict for it and saving each prediction
        self.last_predicted = [(network.predict(data_to_predict)).reshape(-1, 1) for network in self.networks]
        # For each tree in the last_predicted set, compute the average predictions
        predicted_target = np.mean(np.concatenate(self.last_predicted, axis=1), axis=1)
        return predicted_target

    def calculate_rmse_of_predicted(self, validation_data: list):
        if not self.last_predicted:
            raise Exception('You cannot get the RMSE of predicted if predict has not been done beforehand!')
        mse = mean_squared_error(validation_data, np.mean(np.concatenate(self.last_predicted, axis=1), axis=1))
        return np.sqrt(mse) # The RMSE