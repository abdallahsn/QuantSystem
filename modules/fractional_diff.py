import numpy as np
import pandas as pd
from collections import deque

def _get_weights_ffd(d: float, threshold: float = 1e-5, max_window: int = 2000) -> np.ndarray:
    """
    يحسب أوزان الـ fractional diff (Fixed-width Window)
    بدون عكس المصفوفة، لتجهيزها للعمليات الموجهة (Vectorized Convolution)
    """
    w = [1.0]
    for k in range(1, max_window):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
    # لا نقوم بعكس المصفوفة هنا لأن دالة Convolution تقوم بالضرب العكسي رياضياً
    return np.array(w)

def frac_diff_series(series: pd.Series,
                     d: float = 0.4,
                     threshold: float = 1e-5) -> pd.Series:
    """
    يطبق Fractional Differentiation بسرعة الضوء (Vectorized)
    """
    # الحصول على الأوزان
    weights = _get_weights_ffd(d, threshold=threshold)
    
    # حماية من القيم الفارغة قبل المعالجة
    arr = series.ffill().fillna(0.0).values.astype(np.float64)

    if len(weights) <= 1:
        return series.copy()

    # ══════════════════════════════════════════════════════════
    # LOOK-AHEAD FIX: mode='full' كان يُدخل معلومات مستقبلية
    # الحل الصحيح: padding يسار فقط (causal convolution)
    #   كل نقطة في الناتج تعتمد فقط على نقاط في الماضي
    # ══════════════════════════════════════════════════════════
    w_len = len(weights)
    # نضيف zeros على اليسار (الماضي) بدلاً من المستقبل
    padded = np.concatenate([np.zeros(w_len - 1), arr])
    frac_diffed = np.convolve(padded, weights[::-1], mode='valid')
    # الآن frac_diffed[t] يعتمد فقط على arr[t-w_len+1 : t+1]  (causal ✅)
    frac_diffed = frac_diffed[:len(arr)]

    result = pd.Series(frac_diffed, index=series.index)
    return result

def apply_fractional_diff(df: pd.DataFrame,
                           d: float = 0.4,
                           threshold: float = 1e-5) -> pd.DataFrame:
    """
    يطبق Fractional Differentiation على الـ features التراكمية (فقط).
    تم تصحيح قائمة الـ Features المستهدفة لمنع تدمير البيانات المستقرة.
    """
    df = df.copy()

    # يتم تطبيق التمايز الكسري *فقط* على المتغيرات التراكمية المفتوحة (Non-stationary)
    # مثل السعر والـ CVD التراكمي.
    CUMULATIVE_FEATURES = {
        'cvd_cumulative':    'cvd_frac',
        'close':             'close_frac',
        'cvd':               'cvd_frac_fallback',
        # ══ فيتشرز Quant — مطلوبة في FEATURE_COLS ══
        'kyle_lambda':       'kyle_frac',
        'hawkes_intensity':  'hawkes_frac',
        'vnet':              'vnet_frac',
    }

    added = []
    for src, dst in CUMULATIVE_FEATURES.items():
        if src not in df.columns:
            continue
            
        print(f"  ⏳ جاري حساب Fractional Diff لـ {src}...")
        df[dst] = frac_diff_series(df[src], d=d, threshold=threshold)
        added.append(dst)

    if added:
        print(f"  ✅ Fractional Diff (d={d}): {added}")

    return df, added