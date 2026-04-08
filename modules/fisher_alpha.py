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

    def update_and_get_signal(self, price, cvd):
        self.prices.append(price)
        if len(self.prices) < self.lookback:
            return 0

        roll_max = max(self.prices)
        roll_min = min(self.prices)
        
        # حماية من القسمة على صفر (توقف السعر)
        denom = (roll_max - roll_min)
        if denom == 0:
            denom = 0.0001
            
        # 1. تطبيع السعر (Normalization) في نطاق [-1, 1]
        normalized_price = 2.0 * ((price - roll_min) / denom) - 1.0

        # 2. التنعيم الأولي (Smoothing) قبل تحويلة فيشر (مهم جداً لتقليل الـ Noise)
        value = 0.5 * 2.0 * normalized_price + 0.5 * self.prev_value
        
        # تقييد القيمة لمنع أخطاء اللوغاريتم (Math Domain Error)
        value = max(-0.999, min(0.999, value))
        self.prev_value = value

        # 3. تحويلة فيشر (Fisher Transform)
        raw_fisher = 0.5 * math.log((1.0 + value) / (1.0 - value)) + 0.5 * self.prev_fisher
        self.prev_fisher = raw_fisher

        # 4. حساب الـ Trigger (الإشارة) باستخدام EMA
        current_ema = raw_fisher if self.prev_ema is None else (raw_fisher * self.alpha) + (self.prev_ema * (1 - self.alpha))

        # 5. توليد الإشارة بناءً على التقاطع والـ CVD
        signal = 0
        if self.prev_ema is not None:
            # تقاطع إيجابي من مناطق تشبع بيعي + تأكيد من السيولة الشرائية
            if raw_fisher > current_ema and self.prev_ema < -self.threshold and cvd > 0:
                signal = 1
            # تقاطع سلبي من مناطق تشبع شرائي + تأكيد من السيولة البيعية
            elif raw_fisher < current_ema and self.prev_ema > self.threshold and cvd < 0:
                signal = -1

        self.prev_ema = current_ema
        return signal