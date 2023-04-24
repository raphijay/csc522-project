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
from sklearn.metrics import mean_squared_error


class RandomNetworkEnsemble():

    ##
    # Constructor
    # num_networks: an integer specifying the number of neural networks to include in the ensemble.
    # activation: a string specifying the activation function to use in the neural networks. The
    # options are 'identity', 'logistic', 'tanh', and 'relu'.
    # hidden_layer_sizes: a tuple specifying the number of nodes in each hidden layer of the neural
    # networks.
    # learning_rate_init: a float specifying the initial learning rate for the neural networks.
    # random_seed: an integer specifying the seed for the random number generator used by the ensemble.
    # self.networks: an empty list that will be populated with the neural networks in the ensemble.
    # self.resampler: a random number generator based on the numpy library.
    # self.last_predicted: a variable to store the last prediction made by the ensemble.
    # self.seed: a variable to store the random seed used by the ensemble.
    ##
    def __oldinit__(self, num_networks, activation: str = 'relu', hidden_layer_sizes = (100,), learning_rate_init: float = 0.001, random_seed: int = None):
        self.networks           = []
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

    ##
    # Constructor
    # num_networks: an integer specifying the number of neural networks to include in the ensemble.
    # base_nn_model: the input NN class type for populating the ensemble - must have .fit and .predict methods, similar to sklearn models
    # model_args: the argument map for constructing each base_nn_model
    # random_seed: an integer specifying the seed for the random number generator used by the ensemble.
    # self.networks: an empty list that will be populated with the neural networks in the ensemble.
    # self.resampler: a random number generator based on the numpy library.
    # self.last_predicted: a variable to store the last prediction made by the ensemble.
    # self.seed: a variable to store the random seed used by the ensemble.
    ##
    def __init__(self, num_networks, base_nn_model, model_args, random_seed: int = None):
        self.networks           = []
        self.resampler          = np.random
        self.last_predicted     = None
        self.seed               = None

        if (num_networks < 2):
            raise ValueError('Cannot have random network ensemble with less than 2 networks!')
        self.num_networks       = num_networks
        if (base_nn_model is None):
            raise ValueError('A base neural network model is needed for the ensemble to be populated!')
        if (not hasattr(base_nn_model, 'fit') or not hasattr(base_nn_model, 'predict')):
            raise ValueError('The base neural network model needs to have a .fit(x_train, y_train) and a predict(x_test) method to properly work in the ensemble!')
        self.model              = base_nn_model
        if (model_args is None or len(model_args) <= 0):
            raise ValueError('Arguments are needed for the base neural network to be used in the generation of networks')
        self.model_args         = model_args
        if (random_seed is not None):
            # Recommended way to seed the resampler
            self.seed = random_seed
            self.resampler = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(random_seed)))

    # Getter for networks
    def get_networks(self):
        return self.networks

    ##
    # Public bootstrapping function
    #
    # Bootstrapping is a statistical technique where multiple datasets are created by randomly sampling
    # with replacement from the original dataset. In this case, the function takes a numpy array as
    # input and creates a list of indices from the array. It then loops through the required number of
    # bootstraps, mapping each bootstrapped sample to a key. The key for each bootstrapped sample is
    # constructed by concatenating the string "boot_" with the index of the current network.
    ##
    def make_bootstraps(self, data: np):
        # Get list of indices from the np.array
        idx = [i for i in range(data.shape[0])]
        # Loop through the required number of bootstraps, mapping each bootstrapped sample to a key
        return { 'boot_' + str(network_num) : data[np.random.choice(idx, size=data.shape[0]), :] for network_num in range(self.num_networks)}

    ##
    # Trains the Random Network Ensemble model. It takes two arguments: feature_train and target_train,
    # which are the feature and target data used for training the model.
    #
    # First concatenates the feature_train and target_train data horizontally to create the training
    # data. It then creates a set of bootstrap samples from the training data using the make_bootstraps
    # method defined earlier.
    #
    # For each bootstrap sample, the function creates a new instance of the MLPRegressor class with the
    # specified parameters and fits it to the bootstrap sample using the fit method.
    # After fitting each new neural network to the features and target from each bootstrap sample, the
    # function saves the fitted neural networks in the networks attribute of the Random Network
    # Ensemble object.
    ##
    def train(self, feature_train: list, target_train: list):
        # Create the training instances by combining the feature and target columns
        training_data = np.concatenate((feature_train, target_train.reshape(-1,1)), axis=1)
        # Bootstrap the training instances
        bootstraps = self.make_bootstraps(training_data)
        # The base unit for the Random Forest - MLPRegressor
        # TODO: Change this baseline neural network to be the custom RNN once that is ready
        """
        nn_m = MLPRegressor(
            activation = self.activation,
            hidden_layer_sizes = self.hidden_layer_sizes,
            learning_rate_init = self.learning_rate_init,
            random_state = self.seed
        )
        """
        nn_m = self.model(**self.model_args)
        # Fit each new tree to the features and target from each bootstrap sample
        # bootstraps[b][:, :-1] is the current bootstrap's features
        # bootstraps[b][:, -1] is the current bootstrap's target
        # Neural network's fit method requires both to work
        try:
            self.networks = [clone(nn_m).fit(bootstraps[b][:, :-1], bootstraps[b][:, -1].reshape(-1, 1).ravel()) for b in bootstraps]
        except ValueError:
            # Some neural networks need the data to be in the constructor's layer shape, so for those scenarios, reshape as needed
            self.networks = [clone(nn_m).fit(bootstraps[b][:, :-1].reshape(1, len(bootstraps['boot_0'][:, :-1]), len(bootstraps['boot_0'][:, :-1][0])), bootstraps[b][:, -1].reshape(1, len(bootstraps['boot_0'][:, -1]), 1)) for b in bootstraps]
        return

    ##
    # Public function to predict from the ensemble
    #
    # Predict the target values for a given set of input data using the trained random network
    # ensemble. It takes as input a list of data points to be predicted and returns a numpy array of
    # predicted target values.
    #
    # First checks if the ensemble has been trained or not by checking the length of the networks list
    # If it is zero, meaning the ensemble has not been trained yet, it raises an exception. If it has
    # been trained, it loops through each neural network in the ensemble, performs its native predict()
    # method on the input data, and saves each prediction. The predictions for each neural network are
    # stored as a column in a 2D numpy array. Finally, it computes the average prediction across all
    # neural networks and returns the result as a numpy array of predicted target values. The average
    # prediction is computed using the np.mean() method along the second axis of the 2D numpy array
    # containing the predictions for each neural network.
    ##
    def predict(self, data_to_predict: list):
        if len(self.networks) == 0:
            raise Exception('You must train the random network ensemble first before it can predict!')
        # Loop through each decision tree, doing its native predict for it and saving each prediction
        self.last_predicted = [(network.predict(data_to_predict)).reshape(-1, 1) for network in self.networks]
        # For each tree in the last_predicted set, compute the average predictions
        predicted_target = np.mean(np.concatenate(self.last_predicted, axis=1), axis=1)
        return predicted_target

    ##
    # Calculates the root mean squared error (RMSE) of the ensemble's predictions on a validation
    # dataset. The method expects a list of validation data as input, and it checks if the predict
    # method has been called before. If predict has not been called before, the method raises an
    # exception.
    #
    # Calculates the RMSE by first concatenating the predictions of all networks in the ensemble along
    # the axis = 1, taking the mean along axis=1, and then calculating the mean squared error (MSE)
    # between the concatenated predictions and the actual validation data. Finally, it returns the
    # square root of the MSE as the RMSE.
    ##
    def calculate_rmse_of_predicted(self, validation_data: list):
        if not self.last_predicted:
            raise Exception('You cannot get the RMSE of predicted if predict has not been done beforehand!')
        mse = mean_squared_error(validation_data, np.mean(np.concatenate(self.last_predicted, axis=1), axis=1))
        return np.sqrt(mse) # The RMSE