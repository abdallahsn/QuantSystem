import numpy as np
import pandas as pd
from collections import deque

class SlippageModel:
    """
    يحسب تكلفة التنفيذ الحقيقية من الـ Order Book
    تم تصحيح الحسابات الرياضية للانزلاق والعمولات.
    """

    def __init__(self,
                 tick_size:    float = 0.0001,
                 tick_value:   float = 10.0,   # قيمة النقطة (الـ Pip/Tick Value) بالدولار
                 commission:   float = 2.50,   # دولار / عقد (للاتجاه الواحد)
                 slippage_hist: int = 500):

        self.tick_size    = tick_size
        self.tick_value   = tick_value
        self.commission   = commission
        self._slip_hist   = deque(maxlen=slippage_hist)
        self._avg_slip    = 0.0

    def compute_fill(self, row: dict, size: int, direction: str) -> dict:
        direction = direction.lower()
        remaining = size
        pv_sum    = 0.0   
        filled    = 0

        # تحديد نقطة الأساس (Mid Price) للحساب الصحيح للانزلاق
        best_bid = float(row.get('bid_px_00', 0) or 0)
        best_ask = float(row.get('ask_px_00', 0) or 0)
        
        # حماية من الداتا الفاسدة
        if best_bid <= 0 or best_ask <= 0:
            return self._empty_result(size)
            
        mid_price = (best_bid + best_ask) / 2.0

        if direction == 'long':
            for i in range(10):
                px = float(row.get(f'ask_px_{i:02d}', 0) or 0)
                sz = float(row.get(f'ask_sz_{i:02d}', 0) or 0)
                if px <= 0 or sz <= 0: continue
                take   = min(remaining, sz)
                pv_sum += px * take
                filled += take
                remaining -= take
                if remaining <= 0: break
        else:  # short
            for i in range(10):
                px = float(row.get(f'bid_px_{i:02d}', 0) or 0)
                sz = float(row.get(f'bid_sz_{i:02d}', 0) or 0)
                if px <= 0 or sz <= 0: continue
                take   = min(remaining, sz)
                pv_sum += px * take
                filled += take
                remaining -= take
                if remaining <= 0: break

        if filled == 0:
            return self._empty_result(size)

        fill_price = pv_sum / filled

        # حساب الانزلاق مقارنة بالسعر المثالي (Mid Price) وليس أفضل عرض/طلب
        if direction == 'long':
            slippage_raw = fill_price - mid_price
        else:
            slippage_raw = mid_price - fill_price

        # تحويل الانزلاق الخام إلى Ticks/Pips
        slippage_pips = max(0.0, slippage_raw / self.tick_size)
        fill_ratio    = filled / max(size, 1)

        market_impact = slippage_pips * 0.5 
        
        # العمولة تحسب بالدولار، يجب تحويلها لـ Pips لخصمها من الـ PnL لاحقاً
        # Commission in Pips = (Total Commission in $) / (Filled Size * Tick Value in $)
        commission_in_pips = (filled * self.commission) / (filled * self.tick_value)

        self._slip_hist.append(slippage_pips)
        self._avg_slip = float(np.mean(self._slip_hist))

        return {
            'fill_price':    round(fill_price, 6),
            'slippage_pips': round(slippage_pips, 4),
            'filled':        filled,
            'unfilled':      remaining,
            'fill_ratio':    round(fill_ratio, 4),
            'market_impact': round(market_impact, 4),
            'commission_pips': round(commission_in_pips, 4), # العمولة محسوبة بالنقاط
            'total_cost_pips': round(slippage_pips + commission_in_pips, 4),
        }

    def adjust_pnl(self, raw_pnl_pips: float, size: int, direction: str, row: dict = None) -> dict:
        if row is not None:
            fill_entry = self.compute_fill(row, size, direction)
            # نفترض انزلاق وعمولة مماثلة للخروج
            total_slip_pips = fill_entry['slippage_pips'] * 2
            total_comm_pips = fill_entry['commission_pips'] * 2 
        else:
            total_slip_pips = self._avg_slip * 2
            total_comm_pips = ((self.commission * 2) / self.tick_value)

        # الربح الصافي = الربح النظري - (الانزلاق ذهاب وعودة) - (العمولة ذهاب وعودة بالنقاط)
        real_pnl = raw_pnl_pips - total_slip_pips - total_comm_pips

        return {
            'raw_pnl_pips':  round(raw_pnl_pips, 2),
            'total_slip_pips': round(total_slip_pips, 2),
            'total_comm_pips': round(total_comm_pips, 2),
            'real_pnl_pips': round(real_pnl, 2),
            'profitable':    real_pnl > 0,
        }

    @property
    def avg_slippage(self) -> float:
        return round(self._avg_slip, 4)

    def _empty_result(self, size: int) -> dict:
        return {
            'fill_price': 0.0, 'slippage_pips': 0.0,
            'filled': 0, 'unfilled': size,
            'fill_ratio': 0.0, 'market_impact': 0.0,
            'commission_pips': 0.0, 'total_cost_pips': 0.0,
        }


def kelly_bet_size(confidence: float, win_rate: float, avg_win_pips: float, avg_loss_pips: float, fraction: float = 0.25, max_size: int = 10) -> int:
    """
    تم تصحيح معادلة Kelly لحماية الحساب من الأحجام السالبة أو المبالغ فيها.
    """
    if avg_loss_pips <= 0 or avg_win_pips <= 0 or win_rate <= 0:
        return 1

    # Ratio of win size to loss size
    b = avg_win_pips / avg_loss_pips 
    
    # Kelly Formula: f = W - [ (1 - W) / R ]
    # حيث W = win_rate و R = b
    f = win_rate - ((1.0 - win_rate) / b)

    # إذا كانت النتيجة سالبة (النظام خاسر رياضياً)، لا تدخل الصفقة (أو ادخل بأقل حجم)
    if f <= 0:
        return 1

    f_adj = f * fraction * confidence

    # تحويل نسبة كيلي لعدد عقود بناءً على أقصى حجم مسموح
    size = int(round(f_adj * max_size))
    
    return max(1, min(size, max_size))

def confidence_bet_size(confidence: float, base_size: int = 1, max_size: int = 5) -> int:
    if confidence >= 0.80: return max_size
    elif confidence >= 0.70: return max(base_size, max_size - 1)
    elif confidence >= 0.60: return max(base_size, max_size // 2)
    else: return base_size


def fractional_kelly_bet_size(
    probability: float,
    avg_win_pips: float,
    avg_loss_pips: float,
    *,
    uncertainty: float = 0.0,
    coverage_ratio: float = 1.0,
    regime_entropy: float = 0.0,
    runtime_penalty: float = 1.0,
    fraction: float = 0.25,
    max_size: int = 5,
    min_size: int = 1,
    allow_zero: bool = True,
) -> int:
    p = float(np.clip(probability, 0.0, 1.0))
    win = max(float(avg_win_pips), 1e-8)
    loss = max(float(avg_loss_pips), 1e-8)
    q = 1.0 - p
    b = win / loss
    kelly_fraction = p - (q / max(b, 1e-8))
    if kelly_fraction <= 0.0:
        return 0 if allow_zero else max(int(min_size), 1)

    penalty = float(
        np.clip(1.0 - float(uncertainty), 0.0, 1.0)
        * np.clip(float(coverage_ratio), 0.0, 1.0)
        * np.clip(1.0 - float(regime_entropy), 0.0, 1.0)
        * np.clip(float(runtime_penalty), 0.0, 1.0)
    )
    adjusted_fraction = max(float(kelly_fraction), 0.0) * max(float(fraction), 0.0) * penalty
    size = int(round(adjusted_fraction * max(int(max_size), 1)))
    if size <= 0:
        return 0 if allow_zero else max(int(min_size), 1)
    if allow_zero:
        return min(size, max(int(max_size), 1))
    return min(max(size, max(int(min_size), 1)), max(int(max_size), 1))


def position_size_from_prediction(
    prediction: dict,
    *,
    base_size: int = 1,
    max_size: int = 5,
    fraction: float = 0.25,
) -> int:
    explicit_size = prediction.get('position_size', None)
    if explicit_size is not None:
        try:
            return max(0, min(int(explicit_size), max(int(max_size), 1)))
        except Exception:
            pass

    if prediction.get('tradeable', False) and prediction.get('bias') in ('LONG', 'SHORT'):
        edge_prob = prediction.get('edge_prob')
        avg_win_pips = prediction.get('selected_avg_win_pips')
        avg_loss_pips = prediction.get('selected_avg_loss_pips')
        if edge_prob is not None and avg_win_pips is not None and avg_loss_pips is not None:
            # Do not pass decisionPolicy `sizing_penalty` here: it already blends
            # (1-uncertainty), (1-regime_entropy), and runtime_penalty — using it again
            # as FK's runtime_penalty double-counts and drives size→0 despite tradeable.
            size = fractional_kelly_bet_size(
                float(edge_prob),
                float(avg_win_pips),
                float(avg_loss_pips),
                uncertainty=float(prediction.get('uncertainty', 0.0) or 0.0),
                coverage_ratio=float(prediction.get('policy_coverage_ratio', 1.0) or 0.0),
                regime_entropy=float(prediction.get('regime_entropy', 0.0) or 0.0),
                runtime_penalty=1.0,
                fraction=fraction,
                max_size=max_size,
                min_size=base_size,
                allow_zero=True,
            )
            if size > 0:
                fk = float(np.clip(prediction.get('sizing_penalty', 1.0) or 1.0, 0.0, 1.0))
                scaled = max(1, int(round(float(size) * fk))) if fk < 1.0 else int(size)
                return min(scaled, max(int(max_size), 1))
            # marginal edges + FK rounding → zero; fallback so backtest/paper can execute
            return confidence_bet_size(
                float(prediction.get('confidence', 0.0) or 0.0),
                base_size=int(base_size),
                max_size=int(max_size),
            )

    return confidence_bet_size(
        float(prediction.get('confidence', 0.0) or 0.0),
        base_size=int(base_size),
        max_size=int(max_size),
    )

# ══════════════════════════════════════════════════════════════════
# DailyLossGuard — وقف التداول عند تجاوز الخسارة اليومية
# ══════════════════════════════════════════════════════════════════
class DailyLossGuard:
    """
    يوقف التداول تلقائياً عند تجاوز حد الخسارة اليومية.
    
    المنطق:
    - max_daily_loss_pct: أقصى خسارة مسموحة في اليوم (نسبة من الـ equity)
    - max_daily_trades: أقصى عدد صفقات في اليوم
    - يُصفَّر تلقائياً مع بداية يوم تداول جديد
    """
    
    def __init__(self,
                 max_daily_loss_pct: float = 0.02,   # 2% من الـ equity
                 max_daily_trades:   int   = 20,
                 max_drawdown_pct:   float = 0.05):  # 5% drawdown = وقف كامل
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_daily_trades   = max_daily_trades
        self.max_drawdown_pct   = max_drawdown_pct
        
        self._current_day    = None
        self._daily_pnl      = 0.0
        self._daily_trades   = 0
        self._equity         = 100_000.0  # افتراضي، يُحدَّث من الـ broker
        self._peak_equity    = 100_000.0
        self._blocked        = False
        self._block_reason   = ''

    def update_equity(self, equity: float):
        """تحديث الـ equity من الـ broker"""
        self._equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity

    def reset_daily(self):
        """تصفير العدادات اليومية"""
        self._daily_pnl    = 0.0
        self._daily_trades = 0
        self._blocked      = False
        self._block_reason = ''

    def record_trade(self, pnl: float, ts=None) -> bool:
        """يُسجَّل بعد إغلاق كل صفقة. يُعيد True إذا التداول لا يزال مسموحاً."""
        import pandas as pd
        if ts is not None:
            day = pd.Timestamp(ts).date()
            if self._current_day != day:
                self._current_day = day
                self.reset_daily()

        self._daily_pnl    += pnl
        self._daily_trades += 1

        # فحص الحدود
        loss_pct = abs(min(self._daily_pnl, 0)) / max(self._equity, 1)
        drawdown_pct = (self._peak_equity - self._equity) / max(self._peak_equity, 1)

        if loss_pct >= self.max_daily_loss_pct:
            self._blocked = True
            self._block_reason = f'Daily loss limit: {loss_pct:.1%} >= {self.max_daily_loss_pct:.1%}'
        elif drawdown_pct >= self.max_drawdown_pct:
            self._blocked = True
            self._block_reason = f'Max drawdown: {drawdown_pct:.1%} >= {self.max_drawdown_pct:.1%}'
        elif self._daily_trades >= self.max_daily_trades:
            self._blocked = True
            self._block_reason = f'Max daily trades: {self._daily_trades} >= {self.max_daily_trades}'

        return not self._blocked

    def can_trade(self, ts=None) -> tuple:
        """يُستدعى قبل كل صفقة. يُعيد (bool, reason)"""
        import pandas as pd
        if ts is not None:
            day = pd.Timestamp(ts).date()
            if self._current_day != day:
                self._current_day = day
                self.reset_daily()

        if self._blocked:
            return False, self._block_reason
        return True, 'OK'

    def status(self) -> dict:
        return {
            'blocked':       self._blocked,
            'reason':        self._block_reason,
            'daily_pnl':     round(self._daily_pnl, 2),
            'daily_trades':  self._daily_trades,
            'equity':        round(self._equity, 2),
            'drawdown_pct':  round((self._peak_equity - self._equity) / max(self._peak_equity, 1), 4),
        }
