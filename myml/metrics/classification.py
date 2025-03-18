from __future__ import annotations
import pandas as pd
import numpy as np

class ClassificationMetric:
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
        self.x = x
        self.y = y
        self.w_init = w_init
        self.b_init = b_init
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1/(1 + np.exp(-z))
        
    def compute_logistic_gradient(self, x, y, w, b) -> float:
        m = len(x)
        z = self.sigmoid(np.dot(self.x, w) + b)
        # Reshape y and z to ensure they're both column vectors (m,1) if needed
        y_reshaped = y.reshape(-1, 1) if len(y.shape) == 1 else y
        z_reshaped = z.reshape(-1, 1) if len(z.shape) == 1 else z
    
        dw = (1/m) * np.dot(self.x.T, (z_reshaped - y_reshaped))
        db = (1/m) * np.sum(z_reshaped - y_reshaped)
        return dw, db

    def gradient_descent(self):
        
        w = self.w_init.copy()
        b = self.b_init
        cost_history = []
        
        x = self.x.to_numpy() if isinstance(self.x, pd.DataFrame) else self.x
        y = self.y.to_numpy() if isinstance(self.y, pd.Series) else self.y
        
        for i in range(self.max_iter):
            dw, db = self.compute_logistic_gradient(x, y, w, b)
            
            # update parameters
            w = w  - self.learning_rate * dw
            b = b - self.learning_rate * db
            
        return w, b