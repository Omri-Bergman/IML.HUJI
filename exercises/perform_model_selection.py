from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    foo = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    eps = np.random.normal(0, noise, n_samples)
    X = np.linspace(-1.2, 2, n_samples)
    y= foo(X)
    X_train, y_train, X_test, y_test = split_train_test(pd.DataFrame(X), pd.Series(eps+y), (2 / 3))
    def flat(arr):
        return np.array(arr).flatten()

    X_train_flat, y_train_flat, X_test_flat, y_test_flat = flat(X_train), flat(y_train), flat(X_test), flat(y_test)


    fig1 = go.Figure()
    fig1.add_traces([go.Scatter(x=X, y=y, mode='lines', name='Real points',
                                marker=dict(color="green")),
                     go.Scatter(x=X_train_flat, y=y_train_flat, mode='markers',
                                name='Train points',
                                marker=dict(color="blue")),
                     go.Scatter(x=X_test_flat, y=y_test_flat, mode='markers',
                                name='Test points',
                                marker=dict(color="black"))])
    fig1.show()
    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_err = [-1]*11
    val_err = [-1]*11
    for k in range(11):
         train_err[k], val_err[k] = cross_validate(PolynomialFitting(k), X_train_flat, y_train_flat, mean_square_error, 5)

    fig2 = go.Figure()
    fig2.add_traces(
        [go.Scatter(x=list(range(11)), y=train_err, mode='lines', name='Train errors', marker=dict(color="green")),
         go.Scatter(x=list(range(11)), y=val_err, mode='lines', name='Validation errors',
                    marker=dict(color="blue"))])
    fig2.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    min_k = val_err.index(min(val_err))
    model = PolynomialFitting(int(min_k)).fit(X_train_flat, np.array(y_train))
    test_err = np.round(mean_square_error(np.array(y_test), model.predict(X_test_flat)), 2)
    print(f"Best k validation loss: {round(val_err[min_k], 2)}")
    print(
        f"Best k test loss: {np.round(mean_square_error(np.array(y_test), model.predict(X_test_flat)), 2)}")

    print("number of samples: " + str(n_samples) + ", noise: " + str(noise))
    print("value of k* = " + str(int(min_k)))
    print("test error of the fitted model = " + str(test_err))

def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    X_train, y_train, X_test, y_test = X[:n_samples], y[:n_samples], X[n_samples:], y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambdas = np.linspace(0, 2, n_evaluations)
    train_err_ridge = np.zeros(n_evaluations)
    val_err_ridge = np.zeros(n_evaluations)

    train_err_lasso = np.zeros(n_evaluations)
    val_err_lasso = np.zeros(n_evaluations)

    for i in range(n_evaluations):
        train_err_ridge[i], val_err_ridge[i] = cross_validate(
            RidgeRegression(lambdas[i]), np.array(X_train),
            np.array(y_train), mean_square_error)
        train_err_lasso[i], val_err_lasso[i] = cross_validate(
            Lasso(lambdas[i]), np.array(X_train),
            np.array(y_train), mean_square_error)

    fig3 = go.Figure()
    fig3.add_traces(
        [go.Scatter(x=list(range(n_evaluations)), y=train_err_ridge,
                    mode='lines', name='Ridge Train errors',
                    marker=dict(color="black")),
         go.Scatter(x=list(range(n_evaluations)), y=val_err_ridge,
                    mode='lines', name='Ridge Validation errors',
                    marker=dict(color="blue"))])

    fig3.show()

    fig4 = go.Figure()
    fig4.add_traces(
        [go.Scatter(x=list(range(n_evaluations)), y=train_err_lasso,
                    mode='lines', name='Lasso Train errors',
                    marker=dict(color="green")),
         go.Scatter(x=list(range(n_evaluations)), y=val_err_lasso,
                    mode='lines', name='Lasso Validation errors',
                    marker=dict(color="blue"))])

    fig4.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ridge = np.argmin(val_err_ridge)
    best_lasso = np.argmin(val_err_lasso)
    print(best_lasso, lambdas[best_lasso])
    print(f"Best ridge regularization is {lambdas[best_ridge]}")
    print(f"Best lasso regularization is {lambdas[best_lasso]}")

    lass_model = Lasso(lambdas[best_lasso]).fit(X_train, y_train)
    ridge_model = RidgeRegression(lambdas[best_ridge]).fit(X_train, y_train)
    LS_model = LinearRegression().fit(X_train, y_train)

    print(f"Ridge error: {ridge_model.loss(X_test, y_test)}")
    print(f"LS error: {LS_model.loss(X_test, y_test)}")

    y_lass = lass_model.predict(X_test)
    print(f"Lasso error: {mean_square_error(y_test, y_lass)}")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(100,0)
    select_polynomial_degree(1500,10)
    select_regularization_parameter()
