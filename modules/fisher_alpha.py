import math
from collections import deque

class FastFisherAlpha:
    """
    يحول حركة السعر إلى توزيع طبيعي (Gaussian) لاصطياد نقاط الانعكاس الحادة.
    تم تصحيح المعادلة الرياضية لـ John Ehlers وحل مشكلة الـ Noise.
    """

    def __init__(self, lookback_period=10, smoothing_period=9, threshold=0.3):
        self.lookback    = lookback_period
        self.alpha       = 2.0 / (smoothing_period + 1)
        self.threshold   = threshold
        
        self.prices      = deque(maxlen=lookback_period)
        
        # متغيرات Fisher الصحيحة
        self.prev_value  = 0.0
        self.prev_fisher = 0.0
        self.prev_ema    = None
        self.prev_cvd    = None

    def update_and_get_signal(self, price, cvd):
        price = float(price)
        cvd = float(cvd)
        self.prices.append(price)
        if len(self.prices) < self.lookback:
            self.prev_cvd = cvd
            return 0

        roll_max = max(self.prices)
        roll_min = min(self.prices)
        
        # حماية من القسمة على صفر (توقف السعر)
        denom = (roll_max - roll_min)
        if denom == 0:
            denom = 0.0001
            
        # 1. تطبيع السعر (Normalization) في نطاق [-1, 1]
        normalized_price = 2.0 * ((price - roll_min) / denom) - 1.0

        # 2. التنعيم الأولي (Smoothing) قبل تحويلة فيشر
        value = 0.5 * normalized_price + 0.5 * self.prev_value

        # تقييد القيمة لمنع أخطاء اللوغاريتم (Math Domain Error)
        value = max(-0.999, min(0.999, value))

        # 3. تحويلة فيشر (Fisher Transform)
        raw_fisher = 0.5 * math.log((1.0 + value) / (1.0 - value)) + 0.5 * self.prev_fisher

        # 4. حساب الـ Trigger (الإشارة) باستخدام EMA
        current_ema = raw_fisher if self.prev_ema is None else (raw_fisher * self.alpha) + (self.prev_ema * (1 - self.alpha))

        # 5. توليد الإشارة عند الخروج من مناطق التطرف.
        # الـ CVD هنا يستخدم كعامل ترجيح خفيف لأن السعر قد ينعكس قبل أن تتغير إشارته الصريحة.
        signal = 0
        if self.prev_ema is not None:
            was_above = self.prev_fisher > self.prev_ema
            is_above = raw_fisher > current_ema
            cvd_delta = 0.0 if self.prev_cvd is None else (cvd - self.prev_cvd)

            long_turn = (not was_above) and is_above and self.prev_fisher < -self.threshold
            short_turn = was_above and (not is_above) and self.prev_fisher > self.threshold

            if long_turn and (cvd_delta >= 0 or (raw_fisher - self.prev_fisher) >= self.threshold):
                signal = 1
            elif short_turn and (cvd_delta <= 0 or (self.prev_fisher - raw_fisher) >= self.threshold):
                signal = -1

        self.prev_value = value
        self.prev_fisher = raw_fisher
        self.prev_ema = current_ema
        self.prev_cvd = cvd
        return signal
