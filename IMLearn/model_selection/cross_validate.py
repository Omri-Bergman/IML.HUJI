from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator

from statistics import mean
import random


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    a = np.arange(X.shape[0])
    splits = np.array_split(a, cv)
    train_err = np.zeros(cv)
    val_err = np.zeros(cv)
    for k in range(cv):
        train_set = np.delete(X, splits[k], axis=0)
        val_set = X[splits[k]]
        test_set = np.delete(y, splits[k], axis=0)
        val_test = y[splits[k]]
        estimator.fit(train_set, test_set)
        pred_train = estimator.predict(train_set)
        train_err[k] = scoring(test_set, pred_train)
        pred_val = estimator.predict(val_set)
        val_err[k] = scoring(val_test, pred_val)
    return np.mean(train_err), np.mean(val_err)







# X = np.array(
# #     [[10, 20, 30, 40,1,1,1,1],
# # [10, 20, 30, 40,1,1,1,1],
# # [10, 20, 30, 40,1,1,1,1],
# # [10, 20, 30, 40,1,1,1,1],
# # [10, 20, 30, 40,1,1,1,1],
# # [10, 20, 30, 40,1,1,1,1],
# # [10, 20, 30, 40,1,1,1,1],
# # [10, 20, 30, 40,1,1,1,1],
# # [10, 20, 30, 40,1,1,1,1],
# # [10, 20, 30, 40,1,1,1,1],
# #      [100, 200, 300, 400,1,1,1,1],
# #      [1000, 2000, 3000, 4000,1,1,1,1,1]])
# #     [[1, 2, 3, 4],
# #     [5, 6, 7, 8],
# #     [9, 10, 11, 12],
# #     [13, 14, 15, 16],
# #     [17, 18, 19, 20],
# #      [21, 22, 23, 24],
# #      [1000, 2000, 3000, 4000]])
#     [[1, 1, 1, 1],
#      [2, 2, 2, 2],
#      [3, 3, 3, 3],
#      [4, 4, 4, 4],
#      [5, 5, 5, 5],
#      [6, 6, 6, 6],
#      [7, 7, 7, 7]])
#
# y =  np.array([100,200,300,400,500,600,700])
# cross_validate(None, X, y, None)


