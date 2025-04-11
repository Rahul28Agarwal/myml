from __future__ import annotations

import numpy as np
from scipy.stats import mode

from ..base import BaseEstimator, ClassifierMixin
from ..exception import ParameterError, ValidationError
from ..utils.validation import ArrayLike, Validator


class KNeighborsClassifier(BaseEstimator, ClassifierMixin):
    """Classification based on k-nearest neighbors."""
    
    def __init__(self, k: int = 3, weight: str = "uniform") -> None:
        self.k = k
        self.weight = weight
        self._check_params()

    def _check_params(self) -> None:
        """Validate model parameters."""
        if not isinstance(self.k, int) or self.k <= 0:
            msg = f"k must be positive integer, got {self.k}"
            raise ParameterError(msg)

        if self.weight not in ["uniform", "distance"]:
            msg = f"Invalid weights: {self.weight}"
            raise ParameterError(msg)
    
    def fit(self, X: ArrayLike, y: ArrayLike) -> KNeighborsClassifier:  # noqa: N803
        # Validate and convert inputs to numpy arrays
        self.X_, self.y_ = Validator.validate_X_y(X, y)
        self.n_features_in_ = self.X_.shape[1]
        return self

    def _euclidean_distance(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.sqrt(np.sum((self.X_ - X[:, np.newaxis])**2, axis=2))


    def predict(self, X: ArrayLike) -> np.ndarray:  # noqa: N803
        Validator.check_is_fitted(self, ["X_", "y_"])

        # validate and convert the numpy array
        X_array = Validator.validate_array(X)

        if X.shape[1] != self.n_features_in_:
            msg = f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            raise ValidationError(msg)

        distances = self._euclidean_distance(X_array)
        neighbor_indices = np.argpartition(distances, self.k)[:, :self.k]
        
        neighbor_labels = self.y_[neighbor_indices]
        return mode(neighbor_labels, axis=1).mode.flatten()