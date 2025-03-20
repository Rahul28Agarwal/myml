from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


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
