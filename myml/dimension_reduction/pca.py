from __future__ import annotations

import numpy as np

from ..base import BaseEstimator, TransformerMixin
from ..exception import ParameterError
from ..utils.validation import ArrayLike, Validator


class PCA(BaseEstimator, TransformerMixin):
    """Principal Component Analysis (PCA) implementation.

    PCA performs linear dimensionality reduction using Singular Value Decomposition
    to project data to a lower-dimensional space.
    """

    def __init__(self, n_components: int | None = None) -> None:
        """Initialize PCA.

        Args:
            n_components (int | None, optional): Number of components to keep. If None, keeps all components.
            Defaults to None.

        Parameters
        ----------
            components_ (np.ndarray): Principal axes in feature space
            explained_variance_ (np.ndarray): Explained variance of each component
            explained_variance_ratio_ (np.ndarray): Percentage of variance explained by each component
            singular_values_ (np.ndarray): Singular values corresponding to each component
            mean_ (np.ndarray): Per-feature mean from training data
            n_components_ (int): Actual number of components used in the model

        """
        self.n_components = n_components

        self._check_params()

    def _check_params(self) -> None:
        """Validate model parameters."""
        if self.n_components is not None and (not isinstance(self.n_components, int) or self.n_components <= 0):
            msg = f"n_components must be positive integer, got {self.n_components}"
            raise ParameterError(msg)

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> PCA:
        """Fit the PCA model on the training data.

        Args:
            X: Training data of shape (n_samples, n_features)
            y: Ignored (present for API consistency)

        Returns:
            self: Returns the instance itself

        """
        X_array = Validator.validate_array(X)

        # Center the data by substracting the mean
        self.mean_ = np.mean(X_array, axis=0)
        X_centered = X_array - self.mean_

        # Use SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Store all components and singular values initially
        self.components_full_ = Vt
        self.singular_values_full_ = S

        # Calculate explained variance and ratio
        n_samples = X_array.shape[0]
        self.explained_variance_ = (S**2)/(n_samples - 1)
        self.explained_variance_ratio_ = self.explained_variance_/np.sum(self.explained_variance_)

        # Determin number of components
        if self.n_components is None:
            self.n_components_ = min(X_array.shape)
        else:
            self.n_components_ = min(self.n_components, *X_array.shape)

        # Store components (eigenvectors)
        self.components_ = Vt[:self.n_components_]
        self.singular_values_ = S[:self.n_components_]

        # Store number of features seen during fit
        self.n_features_in_ = X_array.shape[1]

        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Apply dimensionality reduction to X.

        Args:
            X: Data to transform of shape (n_samples, n_features)

        Returns:
            Transformed data of shape (n_samples, n_components_)

        """
        self._check_is_fitted()
        X_array = Validator.validate_array(X)

        # Check feature dimensions match
        if X_array.shape[1] != self.n_features_in_:
            msg = f"Expected {self.n_features_in_} features, got {X_array.shape[1]}"
            raise ParameterError(msg)

        # Center the data using the training mean
        X_centered = X_array - self.mean_

        # Project the data onto principal components
        return np.dot(X_centered, self.components_.T)

    def _check_is_fitted(self) -> None:
        """Check if the model has been fitted."""
        Validator.check_is_fitted(self, ["components_", "mean_"])

    def inverse_transform(self, X: ArrayLike) -> np.ndarray:
        """Transform data back to its original space.

        Args:
            X: Data in transformed space

        Returns:
            X_original: Data in original space

        """
        self._check_is_fitted()
        X_array = np.asarray(X)

        # Project back to original space
        X_original = np.dot(X_array, self.components_)

        # Add the mean back
        return X_original + self.mean_
