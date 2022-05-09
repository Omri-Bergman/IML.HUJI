from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None
        self.d_,  self.k_, self.m_ =0,0,0
    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.d_ = X.shape[1]
        self.cov_ = np.zeros((self.d_, self.d_))
        self.classes_ = np.unique(y)
        self.k_ = len(self.classes_)
        self.mu_ = np.zeros((self.k_, self.d_))
        self.pi_ = np.zeros(self.k_)
        self.m_ = X.shape[0]

        for i, k in enumerate(self.classes_):
            self.mu_[i] = np.mean(X[y==k], axis=0)
            self.pi_[i] = X[y==k].shape[0]/self.m_
            self.cov_ += (X[y==k] - self.mu_[i]).T.dot(X[y==k] - self.mu_[i])
        self.cov_ /= (self.m_-self.k_)
        self._cov_inv = np.linalg.inv(self.cov_)
        self.fitted_ = True
        print("CLASESS NUM: ", self.k_)

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

        l_matrix = self.likelihood(X)
        return np.array([self.classes_[np.argmax(x_j)] for x_j in l_matrix])


    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")
        a = np.array([(self._cov_inv @ self.mu_[k]) for k in range(self.k_)])
        b = np.array([(np.log(self.pi_[k]) - 0.5 * self.mu_[k] @ self._cov_inv @ self.mu_[k]) for k in range(self.k_)])
        ll_matrix = np.zeros([X.shape[0], self.classes_.size])
        for i, x in enumerate(X):
            for k in range(self.k_):
                ll_matrix[i][k] = a[k] @ x + b[k]
        return ll_matrix

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
        from ...metrics import misclassification_error
        return misclassification_error(self._predict(X), y)
