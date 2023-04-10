from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

# Define the RNN model
def create_rnn(neurons=100, activation='tanh', dropout=0.0, optimizer='adam'):
    model = Sequential()
    model.add(SimpleRNN(neurons, input_shape=(10, 1), activation=activation, dropout=dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
def grid_search_test():
    # Load the data
    X_train = np.random.rand(1000, 10, 1)
    y_train = np.random.randint(0, 2, size=(1000,))

    # Create a Keras classifier with the RNN model
    rnn = KerasClassifier(build_fn=create_rnn)

    # Define the hyperparameter grid
    neurons = [50, 100, 150]
    activation = ['tanh', 'relu']
    dropout = [0.0, 0.1, 0.2]
    optimizer = ['adam', 'rmsprop']
    param_grid = dict(neurons=neurons, activation=activation, dropout=dropout, optimizer=optimizer)

    # Perform the grid search
    grid = GridSearchCV(estimator=rnn, param_grid=param_grid, cv=3)
    grid_result = grid.fit(X_train, y_train)

    # Print the results
    print(f"Best score: {grid_result.best_score_} using {grid_result.best_params_}")
