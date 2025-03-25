from __future__ import annotations

import numpy as np

from ..base import BaseEstimator, ClassifierMixin
from ..exception import NotFittedError, ParameterError, ValidationError
from ..utils.optimization import GradientDescent, LinearModel
from ..utils.validation import ArrayLike, Validator


class LogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iter: int = 10000,
        tol: float = 1e-4,
        fit_intercept: bool = True,  # noqa: FBT001
         threshold: float = 0.5,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.threshold = threshold

        # These will set during fitting
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

        if not 0 < self.threshold < 1:
            msg = f"threshold must be between 0 and 1, got {self.threshold}"
            raise ParameterError(msg)

    def fit(self, X: ArrayLike, y: ArrayLike) -> LogisticRegression:  # noqa: N803
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

        # Run Optimizer
        optimizer_params, optimization_info = optimizer.optimize(
            X=X_array,
            y=y_array,
            params_init=params_init,
            compute_loss= LinearModel.log_loss,
            compute_gradients=LinearModel.log_gradients,
        )

        # Store optimized parameters
        self.coef_ = optimizer_params["coef"]
        self.intercept_ = float(optimizer_params["intercept"][0])
        self.n_iter_ = optimization_info["n_iter"]
        self.history_ = {"cost_history": optimization_info["loss_history"]}

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:  # noqa: N803
        # Check if model is fitted
        if self.coef_ is None or self.intercept_ is None:
            msg = "LogisticRegression not fitted. Call 'fit' first."
            raise NotFittedError(msg)

        # Validate and convert input to numpy array
        X_array = Validator.validate_array(X)

        # Check if input has correct number of features
        if X_array.shape[1] != len(self.coef_):
            msg = (
                f"X has {X_array.shape[1]} features, but LogisticRegression "
                f"was trained with {len(self.coef_)} features."
            )
            raise ValidationError(msg)

        # Calculate predictions
        z = LinearModel.predict(X=X_array, params={"coef": self.coef_, "intercept": self.intercept_})
        y_pred = LinearModel.sigmoid(z)
        y_pred = np.where(y_pred > self.threshold, 1, 0)

        return y_pred

    def predict_proba(self, X: ArrayLike) -> np.ndarray:  # noqa: N803
         # Check if model is fitted
        if self.coef_ is None or self.intercept_ is None:
            msg = "LogisticRegression not fitted. Call 'fit' first."
            raise NotFittedError(msg)

        # Validate and convert input to numpy array
        X_array = Validator.validate_array(X)

        # Check if input has correct number of features
        if X_array.shape[1] != len(self.coef_):
            msg = (
                f"X has {X_array.shape[1]} features, but LogisticRegression "
                f"was trained with {len(self.coef_)} features."
            )
            raise ValidationError(msg)

        # Calculate probabilities
        z = LinearModel.predict(X=X_array, params={"coef": self.coef_, "intercept": self.intercept_})
        proba = LinearModel.sigmoid(z)

        # For binary classification, return probabilities for both classes
        return np.vstack([1 - proba, proba]).T
