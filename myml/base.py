from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd

# Create a type alias for all the array-like types
ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series, list]
class BaseEstimator(ABC):
    """Base class for all estimators in the library.

    This class provides common functionality including parameter management,
    input validation, serialization, and core interface definitions that all
    machine learning algorithms should implement.
    """

    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike | None=None) -> BaseEstimator:  # noqa: N803
        """Train the model on data.

        Args:
            X (ArrayLike): Training data of shape (n_samples, n_features)
            y (ArrayLike | None, optional): Target values of shape (n_samples,).
                Defaults to None.

        Returns:
            BaseEstimator: Returns self for method chaining

        """

    @abstractmethod
    def predict(self, X: ArrayLike) -> np.ndarray:  # noqa: N803
        """Make predictions using the trained model.

        Args:
            X (ArrayLike): Data to make predictions on, of shape (n_samples, n_features)

        Returns:
            np.ndarray: Predicted values of shape (n_samples,)

        """

    def _validate_data(
        self,
        X: ArrayLike, # noqa: N803
        y: ArrayLike | None = None,
    ) -> np.ndarray | tuple:
        """Convert input data to a consistent numpy array format.

        Args:
            X (ArrayLike): Input Features
            y (ArrayLike | None, optional): Target values

        Returns:
            X_validated : numpy.ndarray
                Validated and converted feature array
            y_validated : numpy.ndarray, optional
                Validated and converted target array (if y was provided)

        """
        # Convert X to numpy array
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X_validated = X.to_numpy()  # noqa: N806
        else:
            X_validated = np.asarray(X)  # noqa: N806

        # Ensure X is 2D
        if X_validated.ndim == 1:
            X_validated = X_validated.reshape(-1, 1)  # noqa: N806

        # If y is provided, convert it too
        if y is not None:
            if isinstance(y, (pd.DataFrame, pd.Series)):
                y_validated = y.to_numpy()
            else:
                y_validated = np.asarray(y)

            # Ensure y is 1D
            if y_validated.ndim > 1 and y_validated.shape[1] == 1:
                y_validated = y_validated.ravel()

            return X_validated, y_validated

        return X_validated
