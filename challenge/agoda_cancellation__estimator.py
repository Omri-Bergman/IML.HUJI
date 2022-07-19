from __future__ import annotations

from typing import NoReturn

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from IMLearn.base import BaseEstimator


class AgodaCancellationEstimator(BaseEstimator):
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    def __init__(self, k:int) -> AgodaCancellationEstimator:
        """
        Instantiate an estimator for solving the Agoda Cancellation challenge
        Parameters
        ----------
        Attributes
        ----------
        """
        super().__init__()
        self.classification_model = make_pipeline(StandardScaler(), BaggingClassifier(n_estimators=k))

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an estimator for given samples
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to.
            responses: days to cancellation, or -1 if didn't cancel
        Notes
        -----
        """
        self.classification_model.fit(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for
        Returns
        -------
        responses : ndarray of shape (n_samples, 2)
            Predicted responses of given samples.
            first column is the classification prediction, second is the regression one.
        """
        # y1 = np.where(self.classification_model.predict_proba(X)[:, 1] < 0.3, 0, 1)
        # y2 = np.where(self.classification_model.predict_proba(X)[:, 1] < 0.32, 0, 1)
        # y3 = np.where(self.classification_model.predict_proba(X)[:, 1] < 0.34, 0, 1)
        # y4 = np.where(self.classification_model.predict_proba(X)[:, 1] < 0.36, 0, 1)
        # y5 = np.where(self.classification_model.predict_proba(X)[:, 1] < 0.38, 0, 1)
        # return y1, y2, y3, y4, y5
        return np.where(self.classification_model.predict_proba(X)[:, 1] < 0.38, 0, 1)
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for
        Returns
        ----
        """
        return self.classification_model.predict_proba(X)[:, 1]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples, )
            True labels of test samples
        Returns
        -------
        loss : float
            Performance under loss function
        """
        return f1_score(self._predict(X), y, average='macro')