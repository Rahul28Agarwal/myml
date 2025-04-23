from __future__ import annotations

import numpy as np

from ..exception import ParameterError
from ..utils.validation import ArrayLike, Validator


class PCA:

    def __init__(self, n_components: int | None = None):
        self.n_components = n_components

        self._check_params()

    def _check_params(self):
        """Validate model parameters."""
        if not isinstance(self.n_components, int) or self.n_components <= 0:
            msg = f"n_components must be positive integer, got {self.n_components}"
            raise ParameterError(msg)

    def fit(self, X: ArrayLike) -> PCA:
        self.X_array = Validator.validate_array(X)
        return self

    def transform(self, X: ArrayLike) -> ArrayLike:

        X_array = Validator.validate_array(X)
        # Covariance Matrix
        cov = np.cov(X_array.T)

        # Eigen decomposition
        eigenvalue, eigenvectors = np.linalg.eig(cov)

        # Sort components
        sorted_idx = np.argsort(eigenvalue)[::-1]
        components = eigenvectors[:, sorted_idx[:self.n_components]]

        # Project Data
        return X_array @ components
