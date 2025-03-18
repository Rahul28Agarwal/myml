from __future__ import annotations

import numpy as np
import pandas as pd

from ..metrics.classification import ClassificationMetric

class LogisticRegression:
    def __init__(self, learning_rate: float = 0.05, max_iter:int = 10000) -> None:
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.coef_ = None
        self.intercept_ = None

    def _validate_data(self, x):
        """Convert input data to a consistent 2D numpy array format."""
        if isinstance(x, pd.Series):
            return x.to_numpy().reshape(-1, 1)
        elif isinstance(x, np.ndarray) and len(x.shape) == 1:
            return x.reshape(-1, 1)
        elif isinstance(x, pd.DataFrame):
            return x.to_numpy()
        else:
            # For other array-like objects or types we don't recognize
            return np.asarray(x).reshape(-1, 1) if np.asarray(x).ndim == 1 else np.asarray(x)
        
    def fit(self, x: pd.DataFrame, y:pd.Series) -> LogisticRegression:
        # Convert input to standard format
        x = self._validate_data(x)
        
        w = np.zeros(x.shape[1])
        b = 0.0
        
        # Run gradient descent
        optimizer = ClassificationMetric(x, y, w, b, self.learning_rate, self.max_iter)
        self.coef_ , self.intercept_ = optimizer.gradient_descent()
        
        return self
    
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1/(1 + np.exp(-z))
    
    def predict(self, x: pd.DataFrame) -> pd.Series:

        # Check if model is fitted
        if self.coef_ is None or self.intercept_ is None:
            msg = "Model not fitted. Call 'fit' first."
            raise ValueError(msg)

         # Convert input to standard format
        x = self._validate_data(x)

        # Make predictions
        z = np.dot(x, self.coef_) + self.intercept_
        return self.sigmoid(z)
