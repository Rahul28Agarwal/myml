from __future__ import annotations

from typing import Callable

import numpy as np


class GradientDescent:
    """Generic gradient descent optimizer for machine learning models."""

    def __init__(
        self,
        learning_rate: float,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ) -> None:
        """Initialize GradientDescent.

        Args:
            learning_rate (float): Step size for parameter updates
            max_iter (int, optional): Maximum number of iterations.. Defaults to 1000.
            tol (float, optional): Convergence tolerance based on improvement in loss. Defaults to 1e-4.

        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol


    def optimize(
        self,
        X: np.ndarray,  # noqa: N803
        y: np.ndarray,
        params_init: dict[str, np.ndarray],
        compute_loss: Callable,
        compute_gradients: Callable,
    ) -> tuple[dict[str, np.ndarray], dict[str, any]]:
        """Run gradient descent optimization.

        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features)
            y (np.ndarray): Target values of shape (n_samples,)
            params_init (dict[str, np.ndarray]): Initial parameter values.
                Each key is a parameter name, and each value is a numpy array.
            compute_loss (Callable): Function to compute the loss value
            compute_gradients (Callable):  Function to compute gradients

        Returns:
            tuple[dict[str, np.ndarray], dict[str, any]]: A tuple containing:
                - params: Dictionary of optimized parameter values
                - optimization_info: Dictionary with optimization information

        """
        # Initialize parameters
        params = {k: v.copy() for k, v in params_init.items()}
        loss_history = []

        # Run optimization
        for iteration in range(self.max_iter):
            # Compute current loss
            current_loss = compute_loss(X, y, params)
            loss_history.append(current_loss)

            # Compute gradients
            gradients = compute_gradients(X, y, params)

            # Update parameters
            for param_name in params:
                params[param_name] -= self.learning_rate * gradients[param_name]

            # Check for convergence
            if iteration > 0 and abs(loss_history[iteration -1 ] - current_loss) < self.tol:
                break

        # Create result info
        optimization_info = {
            "loss_history": loss_history,
            "n_iter": len(loss_history),
        }

        return params, optimization_info

class LinearModel:
    """Helper class for linear model optimization.

    This class provides methods for computing predictions, 
    loss, and gradients for linear models including linear regression.

    """

    @staticmethod
    def predict(X: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:  # noqa: N803
        """Compute linear model predictions.

        Args:
            X (np.ndarray): Input features
            params (dict[str, np.ndarray]): Parameters with 'coef' and 'intercept' keys

        Returns:
            np.ndarray: Predicted values

        """
        return np.dot(X, params["coef"]) + params["intercept"]

    @staticmethod
    def mse_loss(X: np.ndarray, y: np.ndarray, params: dict[str, np.ndarray]) -> float:  # noqa: N803
        """Compute mean squared error loss.

        Args:
            X (np.ndarray): Input features
            y (np.ndarray): True target values
            params (dict[str, np.ndarray]): Parameters with 'coef' and 'intercept' keys

        Returns:
            float: Mean squared error

        """
        predict = LinearModel.predict(X, params)
        return np.mean((predict - y) ** 2)
    
    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1/(1 + np.exp(-z))
    
    @staticmethod
    def log_loss(X: np.ndarray, y: np.ndarray, params: dict[str, np.ndarray]) -> float:  # noqa: N803
        """Compute log loss.

        Args:
            X (np.ndarray):  Input features
            y (np.ndarray): True target values
            params (dict[str, np.ndarray]): Parameters with 'coef' and 'intercept' keys


        Returns:
            float: Log loss

        """
        linear_predict = LinearModel.predict(X, params)
        predict = LinearModel.sigmoid(linear_predict)
        loss = np.mean(-y * np.log(predict) - (1-y)* np.log(1-predict))
        return loss
    
    @staticmethod
    def log_gradients(X: np.ndarray, y: np.ndarray, params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:  # noqa: N803
        n_samples = X.shape[0]
        linear_predict = LinearModel.predict(X, params)
        predictions = LinearModel.sigmoid(linear_predict)
        error = predictions - y

        return {
            "coef": (1/n_samples) * np.dot(X.T, error),
            "intercept": np.array([np.mean(error)]),
        }

    @staticmethod
    def mse_gradients(X: np.ndarray, y: np.ndarray, params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:  # noqa: N803
        """Compute gradients for MSE loss.

        Args:
            X (np.ndarray): Input features
            y (np.ndarray): True target values
            params (Dict[str, np.ndarray]): Parameters with 'coef' and 'intercept' keys

        Returns:
            Dict[str, np.ndarray]: Gradients for each parameter

        """
        n_samples = X.shape[0]
        predictions = LinearModel.predict(X, params)
        error = predictions - y

        return {
            "coef": (1/n_samples) * np.dot(X.T, error),
            "intercept": np.array([np.mean(error)]),
        }
