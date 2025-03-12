from __future__ import annotations
import pandas as pd
import numpy as np

class RegressionMetrics:
    def __init__(self, pred: pd.Series, actual: pd.Series) -> None:
        self.pred = pred
        self.actual = actual
        
    def mean_square_error(self) -> float:
        m = len(self.pred)
        mse = (np.sum(self.pred - self.actual)**2)/m
        return mse