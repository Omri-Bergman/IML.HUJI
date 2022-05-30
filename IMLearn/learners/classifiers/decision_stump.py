from __future__ import annotations
from typing import Tuple, NoReturn
# from base import BaseEstimator
from IMLearn import BaseEstimator
import numpy as np
from itertools import product
#
#
class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None
        self._sign_mat = None
        self.m_ = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # self._sign_mat = np.ones(shape=(X.shape[1],X.shape[1]))
        # self._sign_mat = []
        # for i in range(X.shape[1]):
        #     row = []
        #     for j in range(X.shape[1]):
        #         if j >= i:
        #             row.append(1)
        #         else:
        #             row.append(-1)
        #     self._sign_mat.append(row)
        if self._sign_mat is None:
            self.m_ = X.shape[0]
            zero_num = np.arange(0, self.m_)
            self._sign_mat = np.where(
                np.arange(self.m_) >= zero_num[:, np.newaxis], 1, -1
            )

        def thres_func(col):
            pos = self._find_threshold(col, y, 1)
            neg = self._find_threshold(col, y, -1)
            return pos if pos[1] <= neg[1] else neg

        self.m_ = X.shape[0]
        thrs = np.apply_along_axis(lambda col: thres_func(col),
                                   0, X).T
        self.j_ = np.argmin(thrs[:, 1])
        self.threshold_, thr_err, self.sign_ = thrs[self.j_]
        return



    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.where(X[:,self.j_] >= self.threshold_, self.sign_,
                        -self.sign_)
    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """

        sort_idx = np.argsort(values)
        values, labels = values[sort_idx], labels[sort_idx]
        # calculating the loss according to sum(D) if y == y_pred
        temp_err = np.sum(np.abs(labels[np.sign(labels) != sign]))
        # putting the threshold in between each two values in order to find
        # the best threshold
        temp_thr = np.concatenate(
            [[-np.inf], (values[1:] + values[:-1]) / 2, [np.inf]])
        # making a loss loss according to different predictions
        losses = np.append(temp_err, temp_err + np.cumsum(labels * sign))
        minimal_loss = np.argmin(losses)
        return temp_thr[minimal_loss], losses[minimal_loss]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return ((self.predict(X) != y).sum()) / X.shape[0]

# if __name__ == '__main__':
#     # s = DecisionStump()
#     # X = np.array([[1, 2, 1, 2], [4, 1, 3, 4], [6, 2, 3, 4], [6, 2, 3, 4]])
#     # y = np.array([[5, 6, -1, 5]])
#     # s.fit(X,y)
#     # print(s._sign_mat)
#     high = 12; low=-5
#     size = high-low
#     X = np.arange(low,high)
#     y = np.where(X > 5, 1,-1)
#     # y[3] =
#     y[8] = 1
#     # y[9] = 1
#     print(X,y,sep="\n")
#     stump = DecisionStump()
#     stump.fit(X,y)
#

######################################################################################3


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        thr_err = np.inf
        for sign, j in product([-1, 1], range(X.shape[1])):
            thr, temp_err = self._find_threshold(X[:, j], y, sign)
            if temp_err < thr_err:
                self.sign_, self.threshold_, self.j_ = sign, thr, j
                thr_err = temp_err

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        y_pred = self.sign_ * ((X[:, self.j_] >= self.threshold_) * 2 - 1)
        return y_pred

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        sort_idx = np.argsort(values)
        values, labels = values[sort_idx], labels[sort_idx]
        # calculating the loss according to sum(D) if y == y_pred
        temp_err = np.sum(np.abs(labels[np.sign(labels) != sign]))
        # putting the threshold in between each two values in order to find
        # the best threshold
        temp_thr = np.concatenate([[-np.inf], (values[1:] + values[:-1]) / 2, [np.inf]])
        # making a loss loss according to different predictions
        losses = np.append(temp_err, temp_err + np.cumsum(labels * sign))
        minimal_loss = np.argmin(losses)
        return temp_thr[minimal_loss], losses[minimal_loss]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self.predict(X)
        return misclassification_error(y, y_pred)
