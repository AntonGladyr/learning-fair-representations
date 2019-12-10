import numpy as np
from sklearn.base import BaseEstimator


class CustomLogisticRegression(BaseEstimator):
    def __init__(self, learning_rate, condition, itr=True):
        self.learning_rate = learning_rate
        self.parameters = None
        self.condition = condition
        self.itr = itr

    def __logistic_function(self, features):
        a = np.dot(features, self.parameters)
        return 1.0 / (1 + np.exp(-a))

    def __update_step(self, features, targets):
        # step 1: sigmoid of all weights(dot)features representing probabilities dim(n, 1)
        probabilities = self.__logistic_function(features)

        # step 2: gradient. Sum[x_i( y_i - sigmoid(w*x_i) ), {i, 1, n}] =: X(dot)probabilities
        # dim(m, 1)
        gradient = np.dot(features.T, targets - probabilities)

        # step 3: multiply by learning rate. dim(m, 1)
        gradient *= self.learning_rate

        # step 4: update weights
        return self.parameters + gradient

    def __fit_itr(self, features, targets, maxitr):
        self.parameters = np.zeros((features.shape[1]))
        for k in range(maxitr):
            self.parameters = self.__update_step(features, targets)

    def __fit_threshold(self, features, targets, atol):
        self.parameters = np.zeros((features.shape[1]))
        prev_parameters = np.ones((features.shape[1]))
        while False in np.isclose(prev_parameters, self.parameters, atol=atol):
            prev_parameters = self.parameters
            self.parameters = self.__update_step(features, targets)

    def fit(self, features, targets):
        if (self.itr):
            self.__fit_itr(features, targets, self.condition)
        else:
            self.__fit_threshold(features, targets, self.condition)

    def predict(self, features):
        return np.round(self.__logistic_function(features))

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {"learning_rate": self.learning_rate, "condition": self.condition, "itr": self.itr}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
