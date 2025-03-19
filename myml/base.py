from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar, Union

import numpy as np
import pandas as pd

from .exception import ParameterError
from .utils.validation import ArrayLike, Validator

# Type variable for self-returning methods
T = TypeVar("T", bound="BaseEstimator")

# Type for prediction output
PredictionType = Union[np.ndarray, pd.Series]

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
            X (ArrayLike): Training data
            y (ArrayLike | None, optional): Target values
                Defaults to None.

        Returns:
            BaseEstimator: Returns self for method chaining

        """

    @abstractmethod
    def predict(self, X: ArrayLike) ->PredictionType:  # noqa: N803
        """Make predictions using the trained model.

        Args:
            X (ArrayLike): Data to make predictions on

        Returns:
           PredictionType: Predicted values

        """

    @abstractmethod
    def _check_params(self) -> None:
        """Validate model parameters.

        This method should be implemented by all estimator subclasses to 
        validate their specific hyperparameters. It should raise ParameterError
        for any invalid parameters.
        """

    def get_params(self) -> dict[str, Any]:
        """Get parameters for this estimator.

        Returns:
            dict[str, Any]: Parameters names mapped with their values.

        """
        # Get all public attributes that don't end with underscore
        params = {
            key: value for key, value in self.__dict__.items()
            if not key.startswith("_") and not key.endswith("_")
        }
        return params

    def set_params(self, **params: Any) -> BaseEstimator:
        """Set the parameters of this estimator.

        Args:
            **params: Estimator parameters

        Returns:
            BaseEstimator: Estimator instance

        """
        valid_params = self.get_params()

        for key, value in params.items():
            if key not in valid_params:
                msg = f"Invalid parameter: {key}"
                raise ParameterError(msg)
            setattr(self, key, value)

        # validate parameters after setting
        self._check_params()

        return self

class RegressionMixin:
    """Mixin class for all regression estimators in myml.

    This mixin provides common functionality for regression models.
    """

    def score(self, X: ArrayLike, y: ArrayLike) -> float:  # noqa: N803
        """Return the coefficient of determination (R^2) of the prediction.

        Args:
            X (ArrayLike): Test Samples
            y (ArrayLike): rue target values for X

        Returns:
            float: R^2 score

        """
        # Check if model is fitted
        Validator.check_is_fitted(self)

        # Validate inputs
        X_validated, y_validated = Validator.validate_X_y(X, y)

        # calculate predictions
        y_pred = self.predict(X_validated)

        # Ensure y_pred is a numpy array
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.to_numpy()

        # Calculate R^2 score
        sum_squared_residuals = ((y_validated - y_pred) ** 2).sum()
        sum_squared_total = ((y_validated - y_validated.mean()) ** 2).sum()

        # If sum_squared_total is close to 0, all true values are nearly identical
        # and R² score is undefined (return 0 in this case)
        if np.isclose(sum_squared_total, 0.0):
            return 0.0

        r_squared = 1 - (sum_squared_residuals / sum_squared_total)
        return r_squared
class ClassifierMixin:
    """Mixin class for all classifiers in myml.

    This mixin provides common functionality for classification models.
    """

    def predict_proba(self, X: ArrayLike) -> np.ndarray:  # noqa: N803
        """Predict class probabilities for samples in X.

        This method is optional for classifiers. If implemented, it should
        return probability estimates for each class.

        Args:
            X: Samples

        Returns:
            array: Probability estimates

        """
        msg = f"{self.__class__.__name__} does not implement predict_proba."
        raise NotImplementedError(msg)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:  # noqa: N803
        """Return the mean accuracy on the given test data and labels.

        Args:
            X: Test samples
            y: True labels for X

        Returns:
            float: Mean accuracy score

        """
        # Check if model is fitted
        Validator.check_is_fitted(self)

        # Validate inputs
        X_validated, y_validated = Validator.validate_X_y(X, y)

        # Calculate predictions and accuracy
        y_pred = self.predict(X_validated)

        # Ensure y_pred is a numpy array
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.to_numpy()

        return np.mean(y_pred == y_validated)
