"""
dynamic_labels.py — الأهداف الديناميكية من دفتر الأوامر
═══════════════════════════════════════════════════════════════════════
بدل TP/SL ثابت → نبحث في الـ Order Book عن:
  SL: أقرب "حائط أوامر" يحمي ظهرك
  TP: أقرب "بركة سيولة" مكشوفة يجذب السعر
═══════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from collections import deque


# ── ثوابت ────────────────────────────────────────────────────────
LABEL_WIN    =  1   # TP ضُرب أولاً
LABEL_LOSE   =  0   # SL ضُرب أولاً
LABEL_CANCEL = -1   # انتهى الوقت


class OrderWallScanner:
    """
    يمسح الـ MBP10 ويكتشف:
    1. حوائط الأوامر (Order Walls) = SL
    2. فراغات السيولة (Liquidity Gaps) = TP
    """

    def __init__(self,
                 wall_mult:  float = 3.0,
                 gap_mult:   float = 2.0,
                 levels:     int   = 10):
        self.wall_mult  = wall_mult
        self.gap_mult   = gap_mult
        self.levels     = levels
        self._size_hist = deque(maxlen=100)

    def scan(self, row: dict, tick_size: float = 0.0001) -> dict:
        bid_px = []; bid_sz = []
        ask_px = []; ask_sz = []

        for i in range(self.levels):
            bp = float(row.get(f'bid_px_{i:02d}', 0) or 0)
            bs = float(row.get(f'bid_sz_{i:02d}', 0) or 0)
            ap = float(row.get(f'ask_px_{i:02d}', 0) or 0)
            as_ = float(row.get(f'ask_sz_{i:02d}', 0) or 0)
            if bp > 0 and bs > 0:
                bid_px.append(bp); bid_sz.append(bs)
            if ap > 0 and as_ > 0:
                ask_px.append(ap); ask_sz.append(as_)

        if not bid_sz or not ask_sz:
            return self._empty_result()

        # متوسط الحجم
        all_sz = bid_sz + ask_sz
        self._size_hist.extend(all_sz)
        mean_sz = np.mean(self._size_hist) if self._size_hist else np.mean(all_sz)
        wall_threshold = mean_sz * self.wall_mult

        # ── Bid Walls (حوائط الشراء) ─────────────────────────
        bid_wall_px = None
        for px, sz in zip(bid_px, bid_sz):
            if sz >= wall_threshold:
                bid_wall_px = px
                break  

        # ── Ask Walls (حوائط البيع) ──────────────────────────
        ask_wall_px = None
        for px, sz in zip(ask_px, ask_sz):
            if sz >= wall_threshold:
                ask_wall_px = px
                break

        # ── Bid Gaps (فراغات للأسفل) ─────────────────────────
        bid_gap_px = None
        for i in range(len(bid_px) - 1):
            gap = abs(bid_px[i] - bid_px[i+1])
            if gap > tick_size * self.gap_mult:
                bid_gap_px = bid_px[i+1]  
                break

        # ── Ask Gaps (فراغات للأعلى) ─────────────────────────
        ask_gap_px = None
        for i in range(len(ask_px) - 1):
            gap = abs(ask_px[i] - ask_px[i+1])
            if gap > tick_size * self.gap_mult:
                ask_gap_px = ask_px[i+1]
                break

        mid = (bid_px[0] + ask_px[0]) / 2 if bid_px and ask_px else 0

        return {
            'mid_price':   round(mid, 6),
            'bid_wall_px': bid_wall_px,
            'ask_wall_px': ask_wall_px,
            'bid_gap_px':  bid_gap_px,
            'ask_gap_px':  ask_gap_px,
        }

    def _empty_result(self) -> dict:
        return {'mid_price':0,'bid_wall_px':None,'ask_wall_px':None,
                'bid_gap_px':None,'ask_gap_px':None}


def compute_dynamic_levels(scan_result: dict,
                            current_price: float,
                            tick_size: float = 0.0001,
                            min_tp_pips: float = 15.0,
                            max_sl_pips: float = 20.0) -> dict:
    mid = current_price
    pip = tick_size

    # LONG levels
    if scan_result['ask_gap_px'] and scan_result['ask_gap_px'] > mid:
        long_tp = scan_result['ask_gap_px'] - mid
    else:
        long_tp = pip * min_tp_pips

    if scan_result['bid_wall_px'] and scan_result['bid_wall_px'] < mid:
        long_sl = mid - scan_result['bid_wall_px']
    else:
        long_sl = pip * (min_tp_pips * 0.5)

    # SHORT levels
    if scan_result['bid_gap_px'] and scan_result['bid_gap_px'] < mid:
        short_tp = mid - scan_result['bid_gap_px']
    else:
        short_tp = pip * min_tp_pips

    if scan_result['ask_wall_px'] and scan_result['ask_wall_px'] > mid:
        short_sl = scan_result['ask_wall_px'] - mid
    else:
        short_sl = pip * (min_tp_pips * 0.5)

    long_sl  = min(long_sl,  pip * max_sl_pips)
    short_sl = min(short_sl, pip * max_sl_pips)
    long_tp  = max(long_tp,  pip * min_tp_pips * 0.5)
    short_tp = max(short_tp, pip * min_tp_pips * 0.5)

    return {
        'long_tp':  round(long_tp, 6),
        'long_sl':  round(long_sl, 6),
        'short_tp': round(short_tp, 6),
        'short_sl': round(short_sl, 6),
        'long_rr':  round(long_tp / max(long_sl, 1e-8), 3),
        'short_rr': round(short_tp / max(short_sl, 1e-8), 3),
    }


def label_with_forward_scan(df_trades: pd.DataFrame,
                              df_levels: pd.DataFrame,
                              max_bars_forward: int = 50,
                              tick_size: float = 0.0001) -> pd.DataFrame:
    """
    ══════════════════════════════════════════════════════════════
    الفصل الزمني الصارم — ZERO LOOK-AHEAD BIAS (نسخة مسرعة بالـ Vectorization)
    ══════════════════════════════════════════════════════════════
    """
    prices     = df_trades['close'].values
    n          = len(prices)

    long_labels  = np.full(n, LABEL_CANCEL, dtype=np.int8)
    short_labels = np.full(n, LABEL_CANCEL, dtype=np.int8)
    
    # 🔴 تصحيح الجراحة: تفادي بطء הـ Loop باستخدام Numpy بالكامل
    ltp_arr = df_levels['long_tp'].fillna(tick_size * 15).values if 'long_tp' in df_levels else np.full(n, tick_size * 15)
    lsl_arr = df_levels['long_sl'].fillna(tick_size * 10).values if 'long_sl' in df_levels else np.full(n, tick_size * 10)
    stp_arr = df_levels['short_tp'].fillna(tick_size * 15).values if 'short_tp' in df_levels else np.full(n, tick_size * 15)
    ssl_arr = df_levels['short_sl'].fillna(tick_size * 10).values if 'short_sl' in df_levels else np.full(n, tick_size * 10)

    for t in range(n - 1):
        entry = prices[t]
        end = min(t + 1 + max_bars_forward, n)

        long_target  = entry + float(ltp_arr[t])
        long_stop    = entry - float(lsl_arr[t])
        short_target = entry - float(stp_arr[t])
        short_stop   = entry + float(ssl_arr[t])

        # ── Forward Loop Vectorized ───────────────────────────
        future_prices = prices[t+1:end]
        
        if len(future_prices) > 0:
            # LONG label
            l_win_idx = np.where(future_prices >= long_target)[0]
            l_lose_idx = np.where(future_prices <= long_stop)[0]
            lw = l_win_idx[0] if len(l_win_idx) > 0 else float('inf')
            ll = l_lose_idx[0] if len(l_lose_idx) > 0 else float('inf')
            
            if lw < ll and lw != float('inf'):
                long_labels[t] = LABEL_WIN
            elif ll < lw and ll != float('inf'):
                long_labels[t] = LABEL_LOSE
                
            # SHORT label
            s_win_idx = np.where(future_prices <= short_target)[0]
            s_lose_idx = np.where(future_prices >= short_stop)[0]
            sw = s_win_idx[0] if len(s_win_idx) > 0 else float('inf')
            sl = s_lose_idx[0] if len(s_lose_idx) > 0 else float('inf')
            
            if sw < sl and sw != float('inf'):
                short_labels[t] = LABEL_WIN
            elif sl < sw and sl != float('inf'):
                short_labels[t] = LABEL_LOSE

    result = df_trades.copy()
    result['long_label']  = long_labels
    result['short_label'] = short_labels

    def _bias(row):
        if row['long_label'] == LABEL_WIN and row['short_label'] != LABEL_WIN:
            return 0  # LONG
        if row['short_label'] == LABEL_WIN and row['long_label'] != LABEL_WIN:
            return 1  # SHORT
        return 2      # NEUTRAL / CANCEL

    result['bias_label'] = result.apply(_bias, axis=1)

    # إحصائيات
    total = len(result)
    lw = int((result['long_label']==LABEL_WIN).sum())
    ll = int((result['long_label']==LABEL_LOSE).sum())
    lc = int((result['long_label']==LABEL_CANCEL).sum())
    sw = int((result['short_label']==LABEL_WIN).sum())
    print(f"  Labels (Long):  WIN={lw}({lw/total:.0%}) LOSE={ll}({ll/total:.0%}) CANCEL={lc}({lc/total:.0%})")
    print(f"  Labels (Short): WIN={sw}({sw/total:.0%})")

    bc = result['bias_label'].value_counts()
    lp = bc.get(0,0)/total; sp = bc.get(1,0)/total; np_ = bc.get(2,0)/total
    print(f"  Bias: LONG={bc.get(0,0)}({lp:.0%}) SHORT={bc.get(1,0)}({sp:.0%}) NEUTRAL={bc.get(2,0)}({np_:.0%})")

    return result