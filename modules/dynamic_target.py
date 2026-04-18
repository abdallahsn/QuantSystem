"""
dynamic_target.py — V18: Regime-Aware Execution + Pipeline Integration
═══════════════════════════════════════════════════════════════════════════
التحديثات (V18):
  - يستقبل الآن scan_result كامل من OrderWallScanner
  - يحسب wall_consumption_rate كـ feature للنموذج
  - يدعم ثلاث أهداف (TP1/TP2/TP3) مع SL تتبعي ديناميكي
  - رادار الانعكاس: VWAP Z-Score + CVD consistency
  - يُرجع context موحد يمكن استخدامه كـ features في التدريب
═══════════════════════════════════════════════════════════════════════════
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


# ══════════════════════════════════════════════════════════════════════════
# Enums & Dataclasses
# ══════════════════════════════════════════════════════════════════════════

class TradeState(Enum):
    OPEN      = 'open'
    PARTIAL   = 'partial'     # TP1 اتضرب
    EXTENDED  = 'extended'    # تمدد للـ TP2/TP3
    CLOSED    = 'closed'


@dataclass
class ActiveTrade:
    direction:        str
    entry_price:      float
    entry_cvd:        float
    tp1:              float
    tp2:              Optional[float]
    tp3:              Optional[float]
    sl:               float
    sl_initial:       float          # الـ SL الأصلي (لا يتغير)
    wall_px:          float
    wall_size_orig:   float
    bid_wall_strength_orig: float    # قوة الحائط عند الدخول
    ask_wall_strength_orig: float
    total_size:       int
    state:            TradeState = TradeState.OPEN
    remaining_size:   int        = 0
    realized_pips:    float      = 0.0
    tp1_hit:          bool       = False
    # تتبع استهلاك الحائط
    wall_consumption_rate: float = 0.0   # feature للنموذج

    def __post_init__(self):
        self.remaining_size = self.total_size


# ══════════════════════════════════════════════════════════════════════════
# DynamicTargetManager
# ══════════════════════════════════════════════════════════════════════════

class DynamicTargetManager:
    """
    يدير الأهداف الديناميكية في السوق الحي.
    V18: متكامل مع Pipeline — يستقبل scan_result الكامل.
    """

    def __init__(self,
                 wall_consumption_threshold: float = 0.30,
                 cvd_consistency_min:        float = 0.55,
                 tp1_close_ratio:            float = 0.50,
                 max_vwap_zscore:            float = 3.0):
        self.wall_thr   = wall_consumption_threshold
        self.cvd_min    = cvd_consistency_min
        self.tp1_ratio  = tp1_close_ratio
        self.max_zscore = max_vwap_zscore

        self._trade:    Optional[ActiveTrade] = None
        self._cvd_hist: List[float] = []

    # ── فتح الصفقة ────────────────────────────────────────────────────────
    def open_trade(self,
                   signal:      dict,
                   levels:      dict,
                   scan_result: dict,
                   context:     dict,
                   total_size:  int = 4) -> ActiveTrade:
        """
        يفتح صفقة جديدة مع استخدام scan_result الكامل.

        Parameters
        ----------
        signal      : {'bias': 'LONG'/'SHORT', 'price': float, 'cvd_delta': float}
        levels      : من compute_dynamic_levels
        scan_result : من OrderWallScanner.scan
        context     : {'remaining_fuel': float, ...}
        """
        direction = signal.get('bias', 'LONG').upper()
        price     = signal.get('price', 0.0)
        cvd       = signal.get('cvd_delta', 0.0)
        pip       = context.get('tick_size', 0.0001)
        min_sl    = 10.0 * pip

        if direction == 'LONG':
            # SL: مسافة الحائط + تعديل بقوة الحائط
            dist_wall    = scan_result.get('dist_to_bid_wall') or context.get('dist_to_bid_wall', 0.0)
            wall_str     = scan_result.get('bid_wall_strength', 1.0)
            wall_size    = scan_result.get('bid_wall_size_raw', float(levels.get('long_wall_size', 1)))
            sl_dist      = max(float(dist_wall) * max(0.5, 1.0 - wall_str * 0.1),
                               min_sl) if dist_wall else min_sl
            sl           = price - sl_dist
            wall_px      = price - (float(dist_wall) if dist_wall else sl_dist)
            bid_wstr     = wall_str
            ask_wstr     = scan_result.get('ask_wall_strength', 0.0)
        else:
            dist_wall    = scan_result.get('dist_to_ask_wall') or context.get('dist_to_ask_wall', 0.0)
            wall_str     = scan_result.get('ask_wall_strength', 1.0)
            wall_size    = scan_result.get('ask_wall_size_raw', float(levels.get('short_wall_size', 1)))
            sl_dist      = max(float(dist_wall) * max(0.5, 1.0 - wall_str * 0.1),
                               min_sl) if dist_wall else min_sl
            sl           = price + sl_dist
            wall_px      = price + (float(dist_wall) if dist_wall else sl_dist)
            bid_wstr     = scan_result.get('bid_wall_strength', 0.0)
            ask_wstr     = wall_str

        # 1:2 RR إجباري للـ TP1
        tp1_dist = sl_dist * 2.0
        remaining_fuel = context.get('remaining_fuel', 80.0 * pip)
        tp2_dist = min(tp1_dist * 2.5, remaining_fuel * 0.5)
        tp3_dist = min(tp1_dist * 5.0, remaining_fuel * 0.9)

        if direction == 'LONG':
            tp1 = price + tp1_dist
            tp2 = price + tp2_dist if tp2_dist > tp1_dist else None
            tp3 = price + tp3_dist if tp3_dist > (tp2_dist or tp1_dist) else None
        else:
            tp1 = price - tp1_dist
            tp2 = price - tp2_dist if tp2_dist > tp1_dist else None
            tp3 = price - tp3_dist if tp3_dist > (tp2_dist or tp1_dist) else None

        self._trade = ActiveTrade(
            direction=direction,
            entry_price=price,
            entry_cvd=cvd,
            tp1=tp1, tp2=tp2, tp3=tp3,
            sl=sl, sl_initial=sl,
            wall_px=wall_px,
            wall_size_orig=wall_size,
            bid_wall_strength_orig=bid_wstr,
            ask_wall_strength_orig=ask_wstr,
            total_size=total_size,
        )
        self._cvd_hist = [cvd]
        return self._trade

    # ── وظائف مساعدة ─────────────────────────────────────────────────────
    def _wall_consumed(self, current_wall_size: float) -> bool:
        orig = self._trade.wall_size_orig
        if orig <= 0:
            return True
        rate = current_wall_size / orig
        self._trade.wall_consumption_rate = round(1.0 - rate, 4)
        return rate < self.wall_thr

    def _cvd_consistent(self, direction: str) -> bool:
        if len(self._cvd_hist) < 2:
            return True
        recent = np.mean(self._cvd_hist[-5:]) if len(self._cvd_hist) >= 5 else self._cvd_hist[-1]
        start  = self._cvd_hist[0]
        if direction == 'LONG':
            return recent > start * self.cvd_min
        else:
            return recent < start * self.cvd_min

    def _close_all(self, reason: str, price: float, tick_size: float) -> dict:
        t = self._trade
        if t is None:
            return {'action': 'none'}
        pips = ((price - t.entry_price) / tick_size
                if t.direction == 'LONG'
                else (t.entry_price - price) / tick_size)
        t.realized_pips += pips * t.remaining_size
        t.state = TradeState.CLOSED

        actions = {
            'action':     f'close_all_{reason}',
            'close_size': t.remaining_size,
            'pips':       round(pips, 1),
            'reason':     reason,
            'trade':      t,
            'context_features': self._get_context_features(price, tick_size),
        }
        t.remaining_size = 0
        return actions

    def _get_context_features(self, price: float, tick_size: float) -> dict:
        """
        يرجع features من حالة الصفقة الحالية لاستخدامها في النموذج.
        """
        t = self._trade
        if t is None:
            return {}
        pip = tick_size
        return {
            'tp_distance':           round(abs(t.tp1 - t.entry_price) / pip, 2),
            'sl_distance':           round(abs(t.sl  - t.entry_price) / pip, 2),
            'expected_rr':           round(abs(t.tp1 - t.entry_price) / max(abs(t.sl - t.entry_price), pip), 3),
            'wall_consumption_rate': t.wall_consumption_rate,
            'realized_pips':         round(t.realized_pips, 2),
            'remaining_size':        t.remaining_size,
        }

    # ── تحديث الصفقة ─────────────────────────────────────────────────────
    def update(self,
               current_price:     float,
               current_cvd_delta: float,
               current_wall_size: float,
               vwap_zscore:       float,
               regime_tradeable:  bool  = True,
               tick_size:         float = 0.0001) -> dict:
        """
        يُستدعى كل tick لتحديث الصفقة.
        يرجع dict بـ action + context_features.
        """
        if self._trade is None or self._trade.state == TradeState.CLOSED:
            return {'action': 'none', 'trade': None, 'context_features': {}}

        t = self._trade
        self._cvd_hist.append(current_cvd_delta)
        ctx = self._get_context_features(current_price, tick_size)

        # ── 🚨 رادار الانعكاس (Emergency Exit) ──────────────────────────
        if abs(vwap_zscore) >= self.max_zscore:
            if (t.direction == 'LONG'  and vwap_zscore > 0) or \
               (t.direction == 'SHORT' and vwap_zscore < 0):
                result = self._close_all('emergency_exhaustion', current_price, tick_size)
                result['context_features'] = ctx
                return result

        # ── SL ───────────────────────────────────────────────────────────
        sl_hit = (t.direction == 'LONG'  and current_price <= t.sl) or \
                 (t.direction == 'SHORT' and current_price >= t.sl)
        if sl_hit:
            result = self._close_all('sl_hit', current_price, tick_size)
            result['context_features'] = ctx
            return result

        # ── TP1 ──────────────────────────────────────────────────────────
        if not t.tp1_hit:
            tp1_reached = (t.direction == 'LONG'  and current_price >= t.tp1) or \
                          (t.direction == 'SHORT' and current_price <= t.tp1)

            if tp1_reached:
                t.tp1_hit = True
                close_now  = max(1, min(int(t.total_size * self.tp1_ratio), t.remaining_size))
                pips_tp1   = abs(t.tp1 - t.entry_price) / tick_size
                t.realized_pips  += pips_tp1 * close_now
                t.remaining_size -= close_now
                t.state           = TradeState.PARTIAL
                t.sl              = t.entry_price   # Break Even

                can_extend = (
                    t.tp2 is not None
                    and t.remaining_size > 0
                    and regime_tradeable
                    and not self._wall_consumed(current_wall_size)
                    and self._cvd_consistent(t.direction)
                    and abs(vwap_zscore) < (self.max_zscore - 0.5)
                )

                if can_extend:
                    t.state = TradeState.EXTENDED
                    return {
                        'action':          'partial_close_extend',
                        'close_size':      close_now,
                        'remain_size':     t.remaining_size,
                        'next_target':     t.tp2,
                        'reason':          'context_supports_extension',
                        'pips_locked':     round(pips_tp1, 1),
                        'trade':           t,
                        'context_features': ctx,
                    }
                else:
                    t.state          = TradeState.CLOSED
                    t.remaining_size = 0
                    return {
                        'action':          'close_all_tp1',
                        'close_size':      t.total_size,
                        'pips':            round(pips_tp1, 1),
                        'reason':          'tp1_exhaustion_detected',
                        'trade':           t,
                        'context_features': ctx,
                    }

        # ── TP2 ──────────────────────────────────────────────────────────
        if t.state == TradeState.EXTENDED and t.tp2:
            tp2_reached = (t.direction == 'LONG'  and current_price >= t.tp2) or \
                          (t.direction == 'SHORT' and current_price <= t.tp2)

            if tp2_reached:
                pips_tp2  = abs(t.tp2 - t.entry_price) / tick_size
                can_tp3   = (t.tp3 is not None
                             and self._cvd_consistent(t.direction)
                             and abs(vwap_zscore) < self.max_zscore)

                if can_tp3:
                    close_tp2 = max(1, t.remaining_size - 1)
                    t.realized_pips  += pips_tp2 * close_tp2
                    t.remaining_size -= close_tp2
                    t.sl = t.tp1   # تحريك الوقف لـ TP1

                    if t.remaining_size <= 0:
                        t.state = TradeState.CLOSED
                        return {
                            'action':          'close_all_tp2',
                            'close_size':      close_tp2,
                            'pips':            round(pips_tp2, 1),
                            'trade':           t,
                            'context_features': ctx,
                        }
                    return {
                        'action':          'partial_close_extend_tp3',
                        'close_size':      close_tp2,
                        'remain_size':     t.remaining_size,
                        'next_target':     t.tp3,
                        'pips_locked':     round(pips_tp2, 1),
                        'trade':           t,
                        'context_features': ctx,
                    }
                else:
                    t.realized_pips  += pips_tp2 * t.remaining_size
                    t.state           = TradeState.CLOSED
                    t.remaining_size  = 0
                    return {
                        'action':          'close_all_tp2',
                        'close_size':      t.total_size,
                        'pips':            round(pips_tp2, 1),
                        'trade':           t,
                        'context_features': ctx,
                    }

        # ── TP3 ──────────────────────────────────────────────────────────
        if t.tp3 and t.remaining_size > 0:
            tp3_reached = (t.direction == 'LONG'  and current_price >= t.tp3) or \
                          (t.direction == 'SHORT' and current_price <= t.tp3)
            if tp3_reached:
                result = self._close_all('tp3_hit', t.tp3, tick_size)
                result['context_features'] = ctx
                return result

        return {'action': 'hold', 'trade': t, 'context_features': ctx}

    # ── Getters ───────────────────────────────────────────────────────────
    @property
    def trade(self) -> Optional[ActiveTrade]:
        return self._trade

    @property
    def is_open(self) -> bool:
        return self._trade is not None and self._trade.state != TradeState.CLOSED
