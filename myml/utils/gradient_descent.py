from __future__ import annotations

import numpy as np
import pandas as pd

from ..metrics.regression import RegressionMetrics  # noqa: TID252


class GradientDescent:
    def __init__(
        self,
        x: pd.Series,
        y: pd.Series,
        w_init: pd.Series,
        b_init: pd.Series,
        learning_rate: float,
        max_iter: int = 1000,
        tol:float = 1e-4,
    ) -> None:
        self.X = x
        self.y = y
        self.w_init = w_init
        self.b_init = b_init
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol


    def compute(self) -> None:
        w = self.w_init.copy()
        b = self.b_init
        cost_history = []

        x = self.X.to_numpy() if isinstance(self.X, pd.DataFrame) else self.X
        y = self.y.to_numpy() if isinstance(self.y, pd.Series) else self.y

        for i in range(self.max_iter):
            # Make prediction
            y_pred = np.dot(x, w) + b

            # Compute cost using RegressionMetrics
            metrics = RegressionMetrics(y_pred, y)
            cost = metrics.mean_square_error()
            cost_history.append(cost)

            # Compute gradients
            m = len(x)
            dw = (1/m) * np.dot(x.T, (y_pred - y))
            db = (1/m) * np.sum(y_pred - y)

            # update parameters
            w = w - self.learning_rate * dw
            b = b - self.learning_rate * db

            # Check for convergence
            if i > 0 and abs(cost_history[i-1] - cost) < self.tol:
                break
        return w, b, {"cost_history": cost_history}
