'''
    A random forest implementation custom to the needs of this project.
    Based off of the following reference implmentations:
        * https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
        * https://insidelearningmachines.com/build-a-random-forest-in-python/
'''

import numpy as np

from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


class CustomRandomForestRegressor():
    
    # Constructor
    def __init__(self, n_trees: int = 100, max_depth = None, min_samples_split: int = 2, random_seed: int = None):
        self.trees              = []
        self.resampler          = np.random
        self.last_predicted     = None

        if (n_trees < 2):
            raise ValueError('Not allowed to have fewer than 2 trees in the random forest!')
        self.n_trees            = n_trees
        if (max_depth < 1):
            raise ValueError('Not allowed to have fewer than a max depth of 1!')
        self.max_depth          = max_depth
        if (min_samples_split < 2):
            raise ValueError('Not allowed to have fewer than a min samples split of 2!')
        self.min_samples_split  = min_samples_split
        if (random_seed != None):
            # Recommended way to seed the resampler
            self.resampler = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(random_seed)))

    # Getter for trees
    def get_trees(self):
        return self.trees

    # Public bootstrapping function
    def make_bootstraps(self, data: np):
        # Get list of indices from the np.array
        idx = [i for i in range(data.shape[0])]
        # Loop through the required number of bootstraps, mapping each bootstrapped sample to a key
        return { 'boot_' + str(tree_num) : data[np.random.choice(idx, size=data.shape[0]), :] for tree_num in range(self.n_trees)}
    
    def train(self, feature_train: list, target_train: list):
        # Create the training instances by combining the feature and target columns
        training_data = np.concatenate((feature_train, target_train.reshape(-1,1)), axis=1)
        # Bootstrap the training instances
        bootstraps = self.make_bootstraps(training_data)
        # The base unit for the Random Forest - DecisionTreeRegressor
        tree_m = DecisionTreeRegressor(
            max_depth = self.max_depth, 
            min_samples_split = self.min_samples_split
        )
        # Fit each new tree to the features and target from each bootstrap sample
        # bootstraps[b][:, :-1] is the current bootstrap's features
        # bootstraps[b][:, -1] is the current bootstrap's target
        # Decision tree's fit method requires both to work
        self.trees = [clone(tree_m).fit(bootstraps[b][:, :-1], bootstraps[b][:, -1].reshape(-1, 1)) for b in bootstraps]
        return

    # Public function to predict from the ensemble
    def predict(self, data_to_predict: list):
        if len(self.trees) == 0:
            raise Exception('You must train the random forest first before it can predict!')
        # Loop through each decision tree, doing its native predict for it and saving each prediction
        self.last_predicted = [(m.predict(data_to_predict)).reshape(-1, 1) for m in self.trees]
        # For each tree in the last_predicted set, compute the average predictions
        predicted_target = np.mean(np.concatenate(self.last_predicted, axis=1), axis=1)
        return predicted_target

    def calculate_rmse_of_predicted(self, validation_data: list):
        if not self.last_predicted:
            raise Exception('You cannot get the RMSE of predicted if predict has not been done beforehand!')
        mse = mean_squared_error(validation_data, np.mean(np.concatenate(self.last_predicted, axis=1), axis=1))
        return np.sqrt(mse) # The RMSE