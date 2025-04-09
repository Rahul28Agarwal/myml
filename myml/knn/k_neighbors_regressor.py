from __future__ import annotations

import numpy as np

from ..base import BaseEstimator, RegressionMixin
from ..exception import NotFittedError, ParameterError
from ..utils.validation import ArrayLike, Validator


class KNeighborsRegressor(BaseEstimator, RegressionMixin):
    """Regression based on k-nearest neighbors."""

    def __init__(self, k: int = 3) -> None:
        self.k = k

        self.is_fitted = False

    def _check_params(self) -> None:
        """Validate model parameters."""
        if self.k <= 0:
            msg = f"K in the K-nearest neighbors must be postive, got {self.k}"
            raise ParameterError(msg)

    def fit(self, X: ArrayLike, y: ArrayLike) -> KNeighborsRegressor:  # noqa: N803
        # Validate and convert inputs to numpy arrays
        self.X_array, self.y_array = Validator.validate_X_y(X, y)
        self.is_fitted = True
        return self

    def _euclidean_distance(self, x1, x2) -> float:
        return np.sqrt(np.sum((x1 - x2)**2))

    def _get_neighbors(self, x_test: np.array) -> float:
        distances = []
        for i, x_train in enumerate(self.X_array):
            dist = self._euclidean_distance(x_test, x_train)
            distances.append((i, dist))
        distances.sort(key=lambda x:x[1])
        return distances[:self.k]


    def predict(self, X: ArrayLike):
        if not self.is_fitted:
            msg = "KNeighborsRegressor not fitted. Call 'Fit' first"
            raise NotFittedError(msg)

        # validate and convert the numpy array
        X_array = Validator.validate_array(X)

        predictions = []

        for x in X_array:
            k_neighbors = self._get_neighbors(x)
            neighbor_indices = [n[0] for n in k_neighbors]

            # Get target values of neighbors and calcualte mean
            neighbor_values = self.y_array[neighbor_indices]
            pred = np.mean(neighbor_values)

            predictions.append(pred)

        return np.array(predictions)



