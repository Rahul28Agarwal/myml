from __future__ import annotations
import pandas as pd
import numpy as np
from ...utils.gradient_descent import GradientDescent

class LinearRegression:
    def __init__(self) -> None:
        pass
    
    def fit(self, x: pd.Series | pd.DataFrame, y: pd.Series) -> LinearRegression:
        self.x = x
        self.y = y
    
    def predict(self) -> pd.Series:
        length = len(self.x)
        w = np.zeros(self.x.shape)
        b = np.zeros(length)
        w, b, _, _ = GradientDescent(self.x, self.y, w, b, 0.5, 10000).compute()
        
        return self.x * w + b
        