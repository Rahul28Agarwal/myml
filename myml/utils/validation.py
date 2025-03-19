from __future__ import annotations

from typing import TypeVar, Union

import numpy as np
import pandas as pd

from ..exception import NotFittedError, ValidationError

# Type variable for estimator classes
EstimatorT = TypeVar("EstimatorT")


# Define common array-like input types
ArrayLike = Union[list, np.ndarray, pd.Series, pd.DataFrame]

class Validator:
    """Provide validation utilities for machine learning components.

    This class centralizes data and parameter validation logic for all estimators,
    ensuring consistent handling of input data formats and parameter validation
    across the entire library.
    """

    @staticmethod
    def validate_array(X: ArrayLike) -> np.ndarray:  # noqa: N803
        """Convert input data to a consistent 2D numpy array format.

        Args:
            X (ArrayLike): Input data to validate.

        Returns:
            np.ndarray : Validated 2D numpy array.

        """
        if isinstance(X, pd.Series):
            return X.to_numpy().reshape(-1, 1)
        if isinstance(X, np.ndarray) and len(X.shape) == 1:
            return X.reshape(-1, 1)
        if isinstance(X, pd.DataFrame):
            return X.to_numpy()
        # For other array-like objects or types we don't recongnize
        array = np.asarray(X)
        return array.reshape(-1, 1) if array.ndim == 1 else array

    @staticmethod
    def validate_X_y(X: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:  # noqa: N802, N803
        """Validate X and y arrays and ensure compatibility.

        Args:
            X (ArrayLike): Features data
            y (ArrayLike): Target data

        Returns:
            tuple[np.ndarray, np.ndarray]:  tuple of validated numpy arrays.

        """
        X_array = Validator.validate_array(X)

        # Convert y to numpy array
        y_array = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)

        # Ensure y is 1D if it's a single column
        y_array = y_array.ravel() if y_array.ndim > 1 and y_array.shape[1] == 1 else y_array

        # check compatible lengths
        if X_array.shape[0]  != y_array.shape[0]:
            msg = (
                f"X and y have incompatible shape: X has {X_array.shape[0]} samples, "
                f"but y has {y_array.shape[0]} samples."
            )
            raise ValidationError(msg)

        return X_array, y_array

    @staticmethod
    def check_is_fitted(estimator: EstimatorT, attributes: list[str]|None =  None) -> None:
        """Check if the estimator has been fitted.

        Args:
            estimator (EstimatorT): The estimator to check.
            attributes (list[str] | None, optional): Attributes to check. Defaults to None.

        """
        if attributes is None:
            attributes = ["coef_", "interpect_"]

        if not any(hasattr(estimator, attr) for attr in attributes):
            estimator_name = estimator.__class__.__name__
            msg = (
                f"This {estimator_name} instance is not fitted yet. "
                f"Call 'fit' before using this estimator."
            )
            raise NotFittedError(msg)
