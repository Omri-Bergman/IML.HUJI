from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None
        self.d_, self.k_, self.m_ = 0, 0, 0

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.fitted_ = True
        self.classes_ = np.unique(y)
        self.k_ = self.classes_.shape[0]
        self.d_ = X.shape[1]
        self.m_ = X.shape[0]

        # creating pi, vars, mu as an empty np.array in the fit size
        self.pi_ = np.zeros(self.k_)
        self.vars_ = np.zeros((self.k_, self.d_))
        self.mu_ = np.zeros((self.k_, self.d_))

        for indx, class_name in enumerate(self.classes_):
            x_class = X[y == class_name]
            self.vars_[indx] = np.var(x_class, axis=0, ddof=1)
            self.mu_[indx] = np.mean(x_class, axis=0)
            self.pi_[indx] = x_class.shape[0] / self.m_

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
        return np.argmax(self.likelihood(X), axis=1)

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
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        res = np.zeros((self.m_, self.k_))

        for j in range(self.k_):
            curr_cov = np.diag(self.vars_[j])
            inv_curr_cov = np.linalg.inv(curr_cov)
            z = np.sqrt((2 * np.pi) ** self.d_ * np.linalg.det(curr_cov))
            for i in range(self.m_):
                x_m = X[i] - self.mu_[j]
                x_ij_likelihood = (1 / z) * np.exp(
                    -0.5 * (x_m @ inv_curr_cov) @ x_m.T)
                res[i, j] = x_ij_likelihood * self.pi_[j]
        return res

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
        return misclassification_error(y, self._predict(X))
