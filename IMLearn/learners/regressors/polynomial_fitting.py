from __future__ import annotations
from typing import NoReturn
from IMLearn.learners.regressors.linear_regression import LinearRegression
from IMLearn.base import BaseEstimator
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"
pio.renderers.default = "browser"


class PolynomialFitting(BaseEstimator):
    """
    Polynomial Fitting using Least Squares estimation
    """
    def __init__(self, k: int) -> PolynomialFitting:
        """
        Instantiate a polynomial fitting estimator

        Parameters
        ----------
        k : int
            Degree of polynomial to fit
        """
        super().__init__()
        self.degree_ = k
        self.linear_regression = LinearRegression()
        self.coefs_ = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to polynomial transformed samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        vander = self.__transform(X)
        self.linear_regression.coefs_ = self.linear_regression.fit(vander, y).coefs_


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
        vander = self.__transform(X)
        return self.linear_regression._predict(vander)

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
        vander = self.__transform(X)
        return self.linear_regression._loss(vander, y)

    def __transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform given input according to the univariate polynomial transformation

        Parameters
        ----------
        X: ndarray of shape (n_samples,)

        Returns
        -------
        transformed: ndarray of shape (n_samples, k+1)
            Vandermonde matrix of given samples up to degree k
        """
        return np.vander(X, N=self.degree_ + 1, increasing=True)


if __name__ == '__main__':
    x = np.linspace(1, 10, 1000)
    # x= np.array([0, 1.5, 4])
    # print(x)
    y = -12*x**3+ 32*x**2 - 5*x +14
    y_ = y+np.random.normal(0,1,y.size)
    # fig = px.scatter(pd.DataFrame({'x': x, 'y': y}), x="x", y="y",
    #                  trendline="ols",
    #                  title=f"Correlation Between",
    #                  labels={"x":" Values", "y": "Response Values"})
    # # fig.write_image("pearson.correlation.%s.png" % i)
    # fig.update_layout(title=dict(text='Estimation of the PDF')).show()
    # y = 12 * x - 3
    model = PolynomialFitting(3)
    model.fit(x, y)
    # y_ = y+np.random.normal(0,1,y.size)
    # model = LinearRegression().fit(x, y)
    # for i in model.coefs_:
    #     print(i)
    print(model.coefs_)
    print(model._loss(x, y_))



    # print(model._loss(x, y_))