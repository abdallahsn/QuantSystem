"""
catboost_5m_report.py
=====================
Generates a 5-minute interactive CatBoost dashboard using the same visual
style as the label visualizer.
"""

from __future__ import annotations

import json
import os

try:
    from modules.range_state_machine import apply_range_filter_to_dataframe
    RSM_AVAILABLE = True
except ImportError:
    try:
        from range_state_machine import apply_range_filter_to_dataframe
        RSM_AVAILABLE = True
    except ImportError:
        RSM_AVAILABLE = False
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.oof_stacking import align_probability_columns
from prepare_training_data import CATBOOST_ADVISOR_FEATURES
from train_v19 import _apply_scaler_to_stat_frame, _raw_stat_frame, load_training_csv

try:
    from catboost import CatBoostClassifier

    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False

BIAS_LABELS = {0: "LONG", 1: "SHORT"}
DIRECTION_PROB_COLS = ["cb_prob_long", "cb_prob_short"]
REGIME_LABELS = {
    0: "Trending",
    1: "Ranging",
    2: "Volatile",
    3: "Low_Liquidity",
}
REGIME_COLORS = {
    "Trending": "rgba(0,232,150,0.28)",
    "Ranging": "rgba(90,122,150,0.18)",
    "Volatile": "rgba(255,77,109,0.24)",
    "Low_Liquidity": "rgba(156,163,175,0.18)",
}


def _normalize_freq(freq: str) -> str:
    value = str(freq or "5min").strip()
    if not value:
        return "5min"
    return value.replace("T", "min").replace("H", "h")


def _load_scaler_params(models_dir: str) -> dict:
    path = os.path.join(models_dir, "scaler_params.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing scaler params: {path}")
    with open(path, "r") as f:
        return json.load(f)


def _load_catboost_classes(models_dir: str) -> list[int] | None:
    path = os.path.join(models_dir, "catboost_classes_v19.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        payload = json.load(f)
    classes = payload.get("classes")
    if not isinstance(classes, list):
        return None
    return [int(x) for x in classes]


def _load_optional_market_csv(path: str, t_min: pd.Timestamp, t_max: pd.Timestamp) -> pd.DataFrame | None:
    if not path or not Path(path).exists():
        return None
    df = pd.read_csv(path, engine="python", low_memory=False)
    if "ts_event" not in df.columns:
        return None
    df["ts_event"] = pd.to_datetime(df["ts_event"], errors="coerce")
    df = df[df["ts_event"].notna()].sort_values("ts_event").reset_index(drop=True)
    df = df[(df["ts_event"] >= t_min) & (df["ts_event"] <= t_max)].copy()
    return df


def predict_catboost_frame(csv_path: str, models_dir: str) -> pd.DataFrame:
    df = load_training_csv(csv_path).copy()
    if "price" not in df.columns:
        raise ValueError("Column 'price' is required for CatBoost visualization")
    if "ts_event" not in df.columns:
        raise ValueError("Column 'ts_event' is required for CatBoost visualization")

    if not CB_AVAILABLE:
        raise RuntimeError("CatBoost is not installed in this environment")

    model_path = os.path.join(models_dir, "catboost_advisor_v19.cbm")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing CatBoost model: {model_path}")

    scaler_params = _load_scaler_params(models_dir)
    raw_stat = _raw_stat_frame(df, CATBOOST_ADVISOR_FEATURES)
    X_stat = _apply_scaler_to_stat_frame(raw_stat, scaler_params).values.astype(np.float32)

    model = CatBoostClassifier()
    model.load_model(model_path)
    classes = _load_catboost_classes(models_dir)
    probs = align_probability_columns(
        model.predict_proba(X_stat),
        2,
        classes=classes or getattr(model, "classes_", None),
    )

    out = df.copy()
    out["cb_prob_long"] = probs[:, 0]
    out["cb_prob_short"] = probs[:, 1]
    out["cb_direction_idx"] = np.argmax(probs, axis=1).astype(np.int8)
    out["cb_direction"] = pd.Series(out["cb_direction_idx"]).map(BIAS_LABELS).fillna("UNKNOWN").values
    out["cb_confidence"] = probs.max(axis=1).astype(np.float32)
    return out


def _mode_or_default(series: pd.Series, default_value) -> object:
    s = series.dropna()
    if s.empty:
        return default_value
    mode = s.mode(dropna=True)
    if len(mode) == 0:
        return default_value
    return mode.iloc[0]


def _regime_to_name(value) -> str:
    try:
        iv = int(value)
        return REGIME_LABELS.get(iv, str(value))
    except Exception:
        return str(value)


def _resample_catboost_bars(pred_df: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    freq = _normalize_freq(freq)
    offset = pd.tseries.frequencies.to_offset(freq)

    frame = pred_df.copy()
    frame["ts_event"] = pd.to_datetime(frame["ts_event"], errors="coerce")
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame = frame.dropna(subset=["ts_event", "price"]).sort_values("ts_event")
    frame = frame.set_index("ts_event")

    if "size" not in frame.columns:
        frame["size"] = 0.0
    frame["size"] = pd.to_numeric(frame["size"], errors="coerce").fillna(0.0)

    ohlc = frame["price"].resample(freq).ohlc()
    volume = frame["size"].resample(freq).sum().rename("volume")
    event_count = frame["price"].resample(freq).size().rename("event_count")
    cvd_last = pd.to_numeric(frame.get("cvd", 0.0), errors="coerce").resample(freq).last().rename("cvd")
    cvd_first = pd.to_numeric(frame.get("cvd", 0.0), errors="coerce").resample(freq).first().rename("cvd_first")
    cvd_delta = (cvd_last - cvd_first).rename("cvd_delta")
    obi = pd.to_numeric(frame.get("obi", 0.0), errors="coerce").resample(freq).mean().rename("obi")
    absorption = pd.to_numeric(frame.get("absorption_intensity", 0.0), errors="coerce").resample(freq).mean().rename("absorption_intensity")
    kyle = pd.to_numeric(frame.get("kyle_lambda", 0.0), errors="coerce").resample(freq).mean().rename("kyle_lambda")
    hawkes = pd.to_numeric(frame.get("hawkes_intensity", 0.0), errors="coerce").resample(freq).mean().rename("hawkes_intensity")
    regime = frame.get("regime_label", pd.Series(1, index=frame.index)).resample(freq).apply(lambda s: _mode_or_default(s, 1)).rename("regime_label")

    cb_probs = frame[DIRECTION_PROB_COLS].resample(freq).mean()

    bars = pd.concat(
        [ohlc, volume, event_count, cvd_last, cvd_delta, obi, absorption, kyle, hawkes, regime, cb_probs],
        axis=1,
    ).dropna(subset=["open", "high", "low", "close"])

    prob_matrix = bars[DIRECTION_PROB_COLS].fillna(0.0).values
    bars["cb_direction_idx"] = np.argmax(prob_matrix, axis=1).astype(int)
    bars["cb_direction"] = bars["cb_direction_idx"].map(BIAS_LABELS).fillna("UNKNOWN")
    bars["cb_confidence"] = prob_matrix.max(axis=1).astype(float)
    bars["cb_change_flag"] = (bars["cb_direction"] != bars["cb_direction"].shift(1)).astype(int)
    bars["signal_time"] = pd.to_datetime(bars.index) + offset
    bars["regime_label"] = bars["regime_label"].apply(_regime_to_name)
    return bars.reset_index()


def _compute_signal_stats(bars: pd.DataFrame, future_bars: int = 4) -> dict:
    total = len(bars)
    bc = bars["cb_direction_idx"].value_counts()

    n_long = int(bc.get(0, 0))
    n_short = int(bc.get(1, 0))

    correct_long = correct_short = total_long = total_short = 0
    closes = bars["close"].values.astype(float)
    labels = bars["cb_direction_idx"].values.astype(int)
    for i in range(len(bars) - future_bars):
        lbl = labels[i]
        future_return = closes[i + future_bars] - closes[i]
        if lbl == 0:
            total_long += 1
            if future_return > 0:
                correct_long += 1
        elif lbl == 1:
            total_short += 1
            if future_return < 0:
                correct_short += 1

    return {
        "total": total,
        "n_long": n_long,
        "n_short": n_short,
        "pct_long": n_long / max(total, 1) * 100,
        "pct_short": n_short / max(total, 1) * 100,
        "long_hit_rate": correct_long / max(total_long, 1) * 100,
        "short_hit_rate": correct_short / max(total_short, 1) * 100,
        "note": "5m CatBoost predictions, not Step 4 raw causal labels",
        "regime_dist": bars["regime_label"].value_counts().to_dict() if "regime_label" in bars.columns else {},
    }


def _build_price_hover_trace(bars: pd.DataFrame) -> go.Scatter:
    hover_y = ((bars["high"] + bars["low"]) / 2.0).fillna(bars["close"]).astype(float)
    customdata = np.column_stack(
        [
            bars["open"].astype(float).values,
            bars["high"].astype(float).values,
            bars["low"].astype(float).values,
            bars["close"].astype(float).values,
            bars["event_count"].fillna(0).astype(int).values,
            bars["volume"].fillna(0.0).astype(float).values,
            bars["cb_direction"].fillna("UNKNOWN").astype(str).values,
            bars["cb_confidence"].fillna(0.0).astype(float).values,
            bars["cb_prob_long"].fillna(0.0).astype(float).values,
            bars["cb_prob_short"].fillna(0.0).astype(float).values,
            bars["cvd_delta"].fillna(0.0).astype(float).values,
            bars["obi"].fillna(0.0).astype(float).values,
            bars["absorption_intensity"].fillna(0.0).astype(float).values,
            bars["kyle_lambda"].fillna(0.0).astype(float).values,
            bars["hawkes_intensity"].fillna(0.0).astype(float).values,
            bars["regime_label"].fillna("Ranging").astype(str).values,
        ]
    )
    return go.Scatter(
        x=bars["ts_event"],
        y=hover_y,
        mode="markers",
        name="Candle Data",
        showlegend=False,
        marker=dict(size=18, color="rgba(0,0,0,0)"),
        customdata=customdata,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Open=%{customdata[0]:,.5f}<br>"
            "High=%{customdata[1]:,.5f}<br>"
            "Low=%{customdata[2]:,.5f}<br>"
            "Close=%{customdata[3]:,.5f}<br>"
            "Ticks=%{customdata[4]}<br>"
            "Volume=%{customdata[5]:,.2f}<br>"
            "Signal=%{customdata[6]}<br>"
            "Confidence=%{customdata[7]:.3f}<br>"
            "P(LONG)=%{customdata[8]:.3f}<br>"
            "P(SHORT)=%{customdata[9]:.3f}<br>"
            "CVD Δ=%{customdata[10]:.3f}<br>"
            "OBI=%{customdata[11]:.3f}<br>"
            "Absorption=%{customdata[12]:.3f}<br>"
            "Kyle λ=%{customdata[13]:.3f}<br>"
            "Hawkes=%{customdata[14]:.3f}<br>"
            "Regime=%{customdata[15]}<extra></extra>"
        ),
    )


def _build_dashboard(bars: pd.DataFrame, stats: dict, mbo: pd.DataFrame | None = None) -> go.Figure:
    ts = bars["ts_event"]

    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.48, 0.10, 0.10, 0.10, 0.10, 0.12],
        vertical_spacing=0.02,
        subplot_titles=[
            "① 5m Candles + CatBoost Overlay",
            "② CVD Delta (5m)",
            "③ OBI Mean (5m)",
            "④ Absorption Mean (5m)",
            "⑤ Kyle's Lambda (5m)",
            "⑥ Regime + Hawkes Intensity (5m)",
        ],
    )

    fig.add_trace(
        go.Candlestick(
            x=ts,
            open=bars["open"],
            high=bars["high"],
            low=bars["low"],
            close=bars["close"],
            name="5m Candles",
            increasing_line_color="#00e896",
            decreasing_line_color="#ff4d6d",
            increasing_fillcolor="rgba(0,232,150,0.35)",
            decreasing_fillcolor="rgba(255,77,109,0.30)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(_build_price_hover_trace(bars), row=1, col=1)

    for row in bars[bars["cb_change_flag"] == 1].itertuples(index=False):
        fig.add_vline(
            x=row.signal_time,
            line_dash="dash",
            line_color="rgba(148,163,184,0.28)",
            line_width=1,
            row=1,
            col=1,
        )

    for label, idx, y_col, symbol, color, edge in (
        ("LONG", 0, "low", "diamond", "#00e896", "#008f63"),
        ("SHORT", 1, "high", "diamond-wide", "#ff4d6d", "#b91c3f"),
    ):
        mask = bars["cb_direction_idx"] == idx
        if not bool(mask.any()):
            continue
        y_values = bars.loc[mask, y_col]
        if y_col == "low":
            y_values = y_values * 0.9997
        elif y_col == "high":
            y_values = y_values * 1.0003
        fig.add_trace(
            go.Scatter(
                x=bars.loc[mask, "signal_time"],
                y=y_values,
                mode="markers",
                marker=dict(symbol=symbol, size=12, color=color, line=dict(color=edge, width=1)),
                name=f"CatBoost {label} ({int(mask.sum())})",
                customdata=np.stack(
                    [
                        bars.loc[mask, "open"].values,
                        bars.loc[mask, "high"].values,
                        bars.loc[mask, "low"].values,
                        bars.loc[mask, "close"].values,
                        bars.loc[mask, "cb_confidence"].fillna(0.0).values,
                        bars.loc[mask, "cb_prob_long"].fillna(0.0).values,
                        bars.loc[mask, "cb_prob_short"].fillna(0.0).values,
                        bars.loc[mask, "regime_label"].fillna("Ranging").astype(str).values,
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    f"CatBoost {label}<br>%{{x}}<br>"
                    "Open=%{customdata[0]:,.5f}<br>"
                    "High=%{customdata[1]:,.5f}<br>"
                    "Low=%{customdata[2]:,.5f}<br>"
                    "Close=%{customdata[3]:,.5f}<br>"
                    "Confidence=%{customdata[4]:.3f}<br>"
                    "P(LONG)=%{customdata[5]:.3f}<br>"
                    "P(SHORT)=%{customdata[6]:.3f}<br>"
                    "Regime=%{customdata[7]}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    if mbo is not None and len(mbo) > 0:
        for col in ("price", "size", "side", "action"):
            if col not in mbo.columns:
                mbo[col] = 0.0 if col in ("price", "size") else ""
        mbo["price"] = pd.to_numeric(mbo["price"], errors="coerce")
        mbo["size"] = pd.to_numeric(mbo["size"], errors="coerce").fillna(0.0)
        mbo = mbo[mbo["action"].astype(str).str.upper().isin({"T", "F", "TRADE"})].copy()
        sides = mbo["side"].astype(str).str.upper()
        buy_mbo = mbo[sides.isin({"B", "BID"})]
        sell_mbo = mbo[sides.isin({"A", "ASK", "S", "SELL"})]

        if len(buy_mbo) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buy_mbo["ts_event"],
                    y=buy_mbo["price"],
                    mode="markers",
                    marker=dict(symbol="circle", size=4, color="rgba(0,232,150,0.35)"),
                    name="Buy Aggressor",
                    customdata=buy_mbo["size"].values,
                    hovertemplate="BUY<br>%{x}<br>Price=%{y:.5f}<br>Size=%{customdata}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        if len(sell_mbo) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sell_mbo["ts_event"],
                    y=sell_mbo["price"],
                    mode="markers",
                    marker=dict(symbol="circle", size=4, color="rgba(255,77,109,0.35)"),
                    name="Sell Aggressor",
                    customdata=sell_mbo["size"].values,
                    hovertemplate="SELL<br>%{x}<br>Price=%{y:.5f}<br>Size=%{customdata}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    cvd_colors = ["#00e896" if v >= 0 else "#ff4d6d" for v in bars["cvd_delta"].fillna(0.0)]
    fig.add_trace(
        go.Bar(
            x=ts,
            y=bars["cvd_delta"],
            marker_color=cvd_colors,
            name="CVD Δ",
            hovertemplate="%{x}<br>CVD Δ=%{y:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_color="#555", line_width=1, row=2, col=1)

    obi_colors = ["#00e896" if v > 0.2 else "#ff4d6d" if v < -0.2 else "#58657a" for v in bars["obi"].fillna(0.0)]
    fig.add_trace(
        go.Bar(
            x=ts,
            y=bars["obi"],
            marker_color=obi_colors,
            name="OBI",
            hovertemplate="%{x}<br>OBI=%{y:.3f}<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=0.2, line_color="#00e896", line_dash="dash", line_width=1, row=3, col=1)
    fig.add_hline(y=-0.2, line_color="#ff4d6d", line_dash="dash", line_width=1, row=3, col=1)
    fig.add_hline(y=0.0, line_color="#555", line_width=1, row=3, col=1)

    fig.add_trace(
        go.Scatter(
            x=ts,
            y=bars["absorption_intensity"],
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(176,106,255,0.20)",
            line=dict(color="#b06aff", width=1.5),
            name="Absorption",
            hovertemplate="%{x}<br>Absorption=%{y:.3f}<extra></extra>",
        ),
        row=4,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ts,
            y=bars["kyle_lambda"],
            mode="lines",
            line=dict(color="#ffd060", width=1.4),
            name="Kyle λ",
            hovertemplate="%{x}<br>Kyle λ=%{y:.3f}<extra></extra>",
        ),
        row=5,
        col=1,
    )

    regime_numeric = {"Trending": 1.0, "Volatile": 0.75, "Ranging": 0.50, "Low_Liquidity": 0.25}
    fig.add_trace(
        go.Bar(
            x=ts,
            y=bars["regime_label"].map(regime_numeric).fillna(0.5),
            marker_color=[REGIME_COLORS.get(v, "rgba(90,122,150,0.15)") for v in bars["regime_label"]],
            name="Regime",
            customdata=bars["regime_label"],
            hovertemplate="%{x}<br>Regime=%{customdata}<extra></extra>",
        ),
        row=6,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=bars["hawkes_intensity"],
            mode="lines",
            line=dict(color="#ff8c42", width=1.5),
            name="Hawkes",
            hovertemplate="%{x}<br>Hawkes=%{y:.3f}<extra></extra>",
        ),
        row=6,
        col=1,
    )

    stats_text = (
        f"<b>5m CatBoost Prediction Statistics</b><br>"
        f"{stats.get('note', '5m CatBoost predictions')}<br>"
        f"LONG: {stats['n_long']:,} ({stats['pct_long']:.1f}%) | Fwd Hit Rate: {stats['long_hit_rate']:.0f}%<br>"
        f"SHORT: {stats['n_short']:,} ({stats['pct_short']:.1f}%) | Fwd Hit Rate: {stats['short_hit_rate']:.0f}%"
    )
    fig.add_annotation(
        text=stats_text,
        xref="paper",
        yref="paper",
        x=1.0,
        y=1.02,
        showarrow=False,
        align="right",
        bgcolor="rgba(20,30,45,0.92)",
        bordercolor="#1e3048",
        borderwidth=1,
        font=dict(size=11, color="#c8d8e8", family="IBM Plex Mono"),
    )

    if not bars.empty:
        last_close = float(bars["close"].iloc[-1])
        last_open = float(bars["open"].iloc[-1])
        price_color = "#00e896" if last_close >= last_open else "#ff4d6d"
        fig.add_hline(
            y=last_close,
            line_dash="dot",
            line_color=price_color,
            line_width=1,
            opacity=0.9,
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=bars["ts_event"].iloc[-1],
            y=last_close,
            text=f"{last_close:,.5f}",
            showarrow=False,
            xshift=44,
            bgcolor=price_color,
            bordercolor=price_color,
            font=dict(size=11, color="#ffffff", family="IBM Plex Mono"),
            align="left",
            row=1,
            col=1,
        )

    fig.update_layout(
        title=dict(
            text="QS V19 — 5m CatBoost Visualization Dashboard",
            font=dict(size=18, color="#ffffff", family="IBM Plex Mono"),
            x=0.02,
        ),
        paper_bgcolor="#060a0f",
        plot_bgcolor="#0b0f14",
        font=dict(color="#c8d8e8", family="IBM Plex Mono"),
        height=1380,
        showlegend=True,
        legend=dict(
            bgcolor="rgba(8,11,16,0.92)",
            bordercolor="#1e3048",
            borderwidth=1,
            font=dict(size=10),
            x=0.0,
            y=1.0,
            orientation="h",
        ),
        margin=dict(l=20, r=92, t=58, b=22),
        dragmode="pan",
        hovermode="x",
        hoverdistance=20,
        spikedistance=1000,
        xaxis=dict(rangeslider=dict(visible=False), type="date"),
        hoverlabel=dict(
            bgcolor="rgba(10,14,20,0.96)",
            bordercolor="#223248",
            font=dict(size=11, color="#dbe7f3", family="IBM Plex Mono"),
        ),
    )

    for i in range(1, 7):
        fig.update_xaxes(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.07)",
            gridwidth=0.5,
            zeroline=False,
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="rgba(255,255,255,0.28)",
            spikethickness=1,
            row=i,
            col=1,
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.07)",
            gridwidth=0.5,
            zeroline=False,
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikecolor="rgba(255,255,255,0.18)",
            spikethickness=1,
            side="right",
            tickformat=",.5f" if i == 1 else None,
            row=i,
            col=1,
        )

    return fig


def _build_confusion_chart(bars: pd.DataFrame, future_bars: int = 4) -> go.Figure | None:
    if "close" not in bars.columns or "cb_direction_idx" not in bars.columns:
        return None

    closes = bars["close"].values.astype(float)
    labels = bars["cb_direction_idx"].values.astype(int)
    results = []
    for i in range(len(bars) - future_bars):
        lbl = labels[i]
        if lbl == 2:
            continue
        future_ret = closes[i + future_bars] - closes[i]
        pct_ret = future_ret / (closes[i] + 1e-9) * 100
        outcome = "CORRECT" if (lbl == 0 and future_ret > 0) or (lbl == 1 and future_ret < 0) else "WRONG"
        results.append(
            {
                "ts": bars["ts_event"].iloc[i],
                "label": "LONG" if lbl == 0 else "SHORT",
                "outcome": outcome,
                "ret_pct": pct_ret,
            }
        )

    if not results:
        return None

    res_df = pd.DataFrame(results)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Return Distribution (5m CatBoost)", "Rolling Directional Hit Rate"])

    for lbl, color in (("LONG", "#00e896"), ("SHORT", "#ff4d6d")):
        subset = res_df[res_df["label"] == lbl]
        fig.add_trace(
            go.Histogram(x=subset["ret_pct"], name=lbl, marker_color=color, opacity=0.72, nbinsx=50),
            row=1,
            col=1,
        )

    res_df["correct_num"] = (res_df["outcome"] == "CORRECT").astype(int)
    res_df = res_df.sort_values("ts")
    res_df["rolling_acc"] = res_df["correct_num"].rolling(window=30, min_periods=8).mean() * 100
    fig.add_trace(
        go.Scatter(
            x=res_df["ts"],
            y=res_df["rolling_acc"],
            mode="lines",
            line=dict(color="#ffd060", width=2),
            name="Hit Rate (rolling 30)",
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=50, line_dash="dash", line_color="#555", row=1, col=2)
    fig.update_layout(
        paper_bgcolor="#060a0f",
        plot_bgcolor="#0d1520",
        font=dict(color="#c8d8e8", family="IBM Plex Mono"),
        title=dict(text="5m CatBoost Directional Quality Analysis", font=dict(color="#ffffff", size=16)),
        height=430,
        barmode="overlay",
    )
    return fig


def _build_turns_table(bars: pd.DataFrame) -> pd.DataFrame:
    direction_names = bars["cb_direction_idx"].map(BIAS_LABELS).fillna("UNKNOWN")
    change_mask = direction_names != direction_names.shift(1)
    out = bars.loc[change_mask, ["ts_event", "signal_time", "open", "high", "low", "close"]].copy()
    out["direction"] = direction_names.loc[change_mask].values
    out["prev_direction"] = direction_names.shift(1).loc[change_mask].fillna("START").values
    out["confidence"] = bars.loc[change_mask, "cb_confidence"].values
    out["cb_prob_long"] = bars.loc[change_mask, "cb_prob_long"].values
    out["cb_prob_short"] = bars.loc[change_mask, "cb_prob_short"].values
    return out.reset_index(drop=True)


def generate_catboost_5m_report(
    csv_path: str,
    models_dir: str,
    output_dir: str | None = None,
    freq: str = "5min",
    report_name: str = "catboost_5m",
    mbo_path: str = "",
    mbp_path: str = "",
    max_bars: int = 400,
    future_bars: int = 4,
) -> dict[str, Any]:
    output_dir = output_dir or models_dir
    os.makedirs(output_dir, exist_ok=True)
    freq = _normalize_freq(freq)

    pred_df = predict_catboost_frame(csv_path, models_dir)
    bars = _resample_catboost_bars(pred_df, freq=freq)
    if len(bars) > max_bars:
        bars = bars.iloc[-max_bars:].reset_index(drop=True)

    # ── تطبيق RangeStateMachine على الـ visualization ─────────────────
    # يُظهر فقط الإشارات المؤكدة (N تأكيدات من الحافة الصحيحة)
    # بدلاً من كل إشارة خام من CatBoost
    if RSM_AVAILABLE and not bars.empty:
        try:
            regime_col = 'regime_name' if 'regime_name' in bars.columns else 'regime'
            bars = apply_range_filter_to_dataframe(
                bars,
                signal_col  = 'cb_direction',
                conf_col    = 'cb_confidence',
                price_col   = 'close',
                regime_col  = regime_col,
                tick_size   = 0.0001,
                min_confirmations   = 3,
                confirmation_window = 6,
                min_candles_between = 4,
            )
            # استبدل cb_direction بالإشارة المفلترة للـ stats والرسم
            bars['cb_direction_raw'] = bars['cb_direction'].copy()
            bars['cb_direction']     = bars['rsm_direction']
            bars['cb_direction_idx'] = bars['cb_direction'].map(
                {'LONG': 0, 'SHORT': 1, 'NEUTRAL': 2}
            ).fillna(2).astype(int)
            print(f"  ✅ RSM applied: {(bars['rsm_action']=='ENTER').sum()} confirmed signals")
        except Exception as _rsm_err:
            print(f"  ⚠️ RSM skipped: {_rsm_err}")

    t_min = bars["ts_event"].min() if not bars.empty else pd.Timestamp.min
    t_max = bars["signal_time"].max() if not bars.empty and "signal_time" in bars.columns else pd.Timestamp.max
    mbo = _load_optional_market_csv(mbo_path, t_min, t_max)
    _ = _load_optional_market_csv(mbp_path, t_min, t_max)

    stats = _compute_signal_stats(bars, future_bars=future_bars)
    turns = _build_turns_table(bars)
    fig_main = _build_dashboard(bars, stats, mbo=mbo)
    fig_conf = _build_confusion_chart(bars, future_bars=future_bars)

    html_path = os.path.join(output_dir, f"{report_name}.html")
    turns_csv = os.path.join(output_dir, f"{report_name}_turns.csv")
    signals_csv = os.path.join(output_dir, f"{report_name}_signals.csv")
    summary_json = os.path.join(output_dir, f"{report_name}_summary.json")

    bars.to_csv(signals_csv, index=False)
    turns.to_csv(turns_csv, index=False)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'>")
        f.write("<title>QS V19 5m CatBoost Visualizer</title>")
        f.write(
            "<style>"
            "body{background:#060a0f;margin:0;padding:0;font-family:'IBM Plex Mono',monospace;}"
            ".shell{padding:10px 12px 18px 12px;}"
            "</style>"
        )
        f.write("</head><body>")
        f.write("<div class='shell'>")
        f.write(
            fig_main.to_html(
                full_html=False,
                include_plotlyjs="cdn",
                config={
                    "displaylogo": False,
                    "responsive": True,
                    "scrollZoom": True,
                    "doubleClick": "reset+autosize",
                },
            )
        )
        if fig_conf is not None:
            f.write("<br>")
            f.write(
                fig_conf.to_html(
                    full_html=False,
                    include_plotlyjs=False,
                    config={"displaylogo": False, "responsive": True},
                )
            )
        f.write("</div></body></html>")

    summary = {
        "rows": int(len(pred_df)),
        "bars": int(len(bars)),
        "transitions": int(len(turns)),
        "direction_counts": bars["cb_direction"].value_counts().to_dict() if not bars.empty else {},
        "files": {
            "html": html_path,
            "signals_csv": signals_csv,
            "turns_csv": turns_csv,
        },
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    summary["files"]["summary_json"] = summary_json
    return summary
