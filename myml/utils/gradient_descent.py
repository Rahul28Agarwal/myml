from __future__ import annotations
import pandas as pd
import numpy as np
from ..metrics.regression import RegressionMetrics

class GradientDescent:
    def __init__(self, x: pd.Series, y: pd.Series, w_in: pd.Series, b_in: pd.Series, alpha: float, num_iters=1000) -> None:
        self.x = x
        self.y = y
        self.w_in = w_in
        self.b_in = b_in
        self.alpha = alpha
        self.num_iters = num_iters
        
        
    def predict(self, w, x, b):
        return w*x + b
    
    def compute(self) -> None:
        w = self.w_in.copy()
        b = self.b_in.copy()
        j_history = []
        p_history = [[w, b]]
        
        for i in range(self.num_iters):
            predict = self.predict(w, self.x, b)
            j_old = RegressionMetrics(predict, self.y).mean_square_error()
            j_history.append(j_old)
            m = len(self.x)
            w_temp = w - (self.alpha * np.sum(predict - self.y) * self.x)/m
            b_temp = b - (self.alpha * np.sum(predict - self.y))/m
            w = w_temp
            b = b_temp
            
            predict = self.predict(w, self.x, b)
            j_new = RegressionMetrics(predict, self.y).mean_square_error()
            j_history.append(j_new)
            p_history.append([w,b])
            if j_new > j_old:
                return w, b, j_history, p_history
            
        return w, b, j_history, p_history
            
        