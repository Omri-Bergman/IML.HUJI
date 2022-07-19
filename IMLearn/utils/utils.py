from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_train_test(X: pd.DataFrame, y: pd.Series, train_proportion: float = .75) \
        -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Randomly split given sample to a training- and testing sample

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Data frame of samples and feature values.

    y : Series of shape (n_samples, )
        Responses corresponding samples in data frame.

    train_proportion: Fraction of samples to be split as training set

    Returns
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples

    """
    # val = 'Temp'
    # val_2 = 'price'
    # df = pd.concat([X, y], axis=1)
    # train = df.sample(frac=train_proportion, random_state=1)
    # test = df.drop(train.index)
    # return train.drop(val,1), test.drop(val,1), train[val], test[val]
    # # return train.drop(val_2,1), test.drop(val_2,1), train[val_2], test[val_2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = train_proportion)
    return  X_train, y_train, X_test, y_test

def confusion_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute a confusion matrix between two sets of integer vectors

    Parameters
    ----------
    a: ndarray of shape (n_samples,)
        First vector of integers

    b: ndarray of shape (n_samples,)
        Second vector of integers

    Returns
    -------
    confusion_matrix: ndarray of shape (a_unique_values, b_unique_values)
        A confusion matrix where the value of the i,j index shows the number of times value `i` was found in vector `a`
        while value `j` vas found in vector `b`
    """
    raise NotImplementedError()

def train_test_split_check(X: pd.DataFrame, train_proportion: float = .75):
    return train_test_split(X, test_size=train_proportion)


