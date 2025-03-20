from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import BaseEstimator, RegressionMixin
from ..exception import ParameterError
from ..utils.optimization import GradientDescent, LinearModel
from ..utils.validation import ArrayLike, Validator


class LinearRegression(BaseEstimator, RegressionMixin):
    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iter: int = 10000,
        tol: float = 1e-4,
        fit_intercept: bool = True,  # noqa: FBT001
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

        # These will be set during fitting
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.n_iter_: int | None = None
        self.history_: dict[str, any] | None = None

        # Validate parameters upon initialization
        self._check_params()

    def _check_params(self) -> None:
        """Validate model parameters."""
        if self.learning_rate <= 0:
            msg = f"Learning rate must be postive, got {self.learning_rate}"
            raise ParameterError(msg)

        if self.max_iter <= 0:
            msg = f"max_iter must be positive, got {self.max_iter}"
            raise ParameterError(msg)

        if self.tol <= 0:
            msg = f"tol must be positive, got {self.tol}"
            raise ParameterError(msg)

    def fit(self, X: ArrayLike, y: ArrayLike) -> LinearRegression:
        # Validate and convert inputs to numpy arrays
        X_array, y_array = Validator.validate_X_y(X, y)

        # Initialize parameters
        n_features = X_array.shape[1]
        params_init = {
            "coef": np.zeros(n_features),
            "intercept": np.array([0.0]),
        }

        # Create optimizer
        optimizer = GradientDescent(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            tol=self.tol,
        )

        # Run optimization
        optimized_params, optimization_info = optimizer.optimize(
            X=X_array,
            y=y_array,
            params_init=params_init,
            compute_loss=LinearModel.mse_loss,
            compute_gradients=LinearModel.mse_gradients,
        )

        # Store optimized parameters
        self.coef_ = optimized_params["coef"]
        self.intercept_ = float(optimized_params["intercept"][0])
        self.n_iter_ = optimization_info["n_iter"]
        self.history_ = {"cost_history": optimization_info["loss_history"]}

        # Return self for method chaining
        return self


class LinearRegression:
    def __init__(self, learning_rate:float = 0.05, max_iter:int = 10000) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.coef_ = None
        self.intercept_ = None

    def _validate_data(self, x):
        """Convert input data to a consistent 2D numpy array format."""
        if isinstance(x, pd.Series):
            return x.to_numpy().reshape(-1, 1)
        if isinstance(x, np.ndarray) and len(x.shape) == 1:
            return x.reshape(-1, 1)
        if isinstance(x, pd.DataFrame):
            return x.to_numpy()
        # For other array-like objects or types we don't recognize
        return np.asarray(x).reshape(-1, 1) if np.asarray(x).ndim == 1 else np.asarray(x)

    def fit(self, x: pd.DataFrame, y: pd.Series) -> LinearRegression:

        # Convert input to standard format
        x = self._validate_data(x)

        w = np.zeros(x.shape[1])
        b = 0.0  # Single intercept

        # Run gradient descent
        optimizer = GradientDescent(x, y, w, b, self.learning_rate, self.max_iter)
        self.coef_, self.intercept_, self.history_ = optimizer.compute()

        # Return self for method chaining
        return self

    def predict(self, x: pd.DataFrame) -> pd.Series:
         # Check if model is fitted
        if self.coef_ is None or self.intercept_ is None:
            msg = "Model not fitted. Call 'fit' first."
            raise ValueError(msg)
        # Convert input to standard format
        x = self._validate_data(x)

        # Make predictions
        return np.dot(x, self.coef_) + self.intercept_
