from __future__ import annotations
from typing import NoReturn
from IMLearn.base import BaseEstimator ##TODO ...base
import numpy as np
from numpy.linalg import pinv
from IMLearn.metrics.loss_functions import mean_square_error

# import matplotlib.pyplot as plt
# import numpy as np
from sklearn import datasets, linear_model, metrics


class LinearRegression(BaseEstimator):
    """
    Linear Regression Estimator

    Solving Ordinary Least Squares optimization problem
    """

    def __init__(self, include_intercept: bool = True) -> LinearRegression:
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """
        super().__init__()
        self.include_intercept_, self.coefs_ = include_intercept, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        if not self.include_intercept_:
            X = X.reshape(-1, 1)
        else:
            X = np.c_[np.ones(X.shape[0]), X]
        self.coefs_ =np.linalg.pinv(X) @ y



    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if not self.include_intercept_:
            X = X.reshape(-1, 1)
        else:
            X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.coefs_

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        return mean_square_error(self.predict(X), y)


# if __name__ == '__main__':
#     x = np.linspace(1, 10, 3)
#     print(x)
#     y = 12 * x - 3
#     # y_ = y+np.random.normal(0,1,y.size)
#     print(y)
#     model = LinearRegression().fit(x, y)
#     print(model.coefs_)
#     # print(model._loss(x, y_))
#
