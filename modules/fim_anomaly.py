import math
import numpy as np
from collections import deque

class FastFIMDetector:
    """
    Fisher Information Measure (FIM) Anomaly Detector.
    يكتشف ضرب ستوبات المتداولين (Stop Hunts) عبر قياس التغير المفاجئ 
    في "كمية المعلومات" وتشتت السعر.
    """
    def __init__(self, window_size: int = 20, threshold_multiplier: float = 1.5):
        self.window     = window_size
        self.multiplier = threshold_multiplier
        self.prices     = deque(maxlen=2)
        self.log_returns= deque(maxlen=window_size)
        self.scores     = deque(maxlen=window_size)
        self.fims       = deque(maxlen=window_size * 2)

    def detect_stop_hunts(self, price: float) -> int:
        self.prices.append(price)
        if len(self.prices) < 2:
            return 0

        prev_price = self.prices[-2]
        # درع حماية: منع الانهيار لو البورصة بعتت سعر صفري أو سالب بالغلط (Data Glitch)
        if prev_price <= 0 or price <= 0:
            return 0

        lr = math.log(price / prev_price)
        self.log_returns.append(lr)
        
        if len(self.log_returns) < self.window:
            return 0

        # تحديد النوع بـ float64 لتسريع عمليات Numpy وتجنب الـ Type Casting اللحظي
        lr_arr = np.array(self.log_returns, dtype=np.float64)
        lr_var = np.var(lr_arr, ddof=1)
        
        # حماية من الـ Zero Variance
        if lr_var <= 1e-8:
            lr_var = 1e-8

        score = (lr - np.mean(lr_arr)) / lr_var
        self.scores.append(score)
        
        if len(self.scores) < self.window:
            return 0

        fim = float(np.var(self.scores, ddof=1))
        self.fims.append(fim)
        
        if len(self.fims) < self.window * 2:
            return 0

        fims_arr  = np.array(self.fims, dtype=np.float64)
        threshold = np.mean(fims_arr) + (self.multiplier * np.std(fims_arr, ddof=1))
        
        return 1 if fim > threshold else 0