"""
label_visualizer_5m.py
======================
Interactive 5-minute dashboard for QuantSystem labels.

Example:
  python label_visualizer_5m.py \
    --features outputs_v19/training_features_ready.csv \
    --mbo rich_mbo.csv \
    --mbp rich_mbp.csv \
    --output label_viz_5m.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.catboost_5m_report import predict_catboost_frame


BIAS_LABELS = {0: "LONG", 1: "SHORT", 2: "NEUTRAL"}
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
    return value.replace("T", "min").replace("H", "h")


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, engine="python", low_memory=False)
    if "ts_event" not in df.columns:
        raise ValueError(f"'ts_event' column missing in {path}")
    df["ts_event"] = pd.to_datetime(df["ts_event"], errors="coerce")
    df = df[df["ts_event"].notna()].sort_values("ts_event").reset_index(drop=True)
    return df


def load_and_prepare(
    features_path: str,
    mbo_path: str = "",
    mbp_path: str = "",
    models_dir: str = "",
    freq: str = "5min",
    max_bars: int = 400,
    max_trade_points: int = 4000,
) -> dict:
    freq = _normalize_freq(freq)

    print(f"📥 قراءة features: {features_path}")
    if models_dir:
        print(f"🤖 تحميل CatBoost من: {models_dir}")
        df = predict_catboost_frame(features_path, models_dir)
    else:
        df = _load_csv(features_path)

    price_col = "price" if "price" in df.columns else "micro_price"
    if price_col not in df.columns:
        raise ValueError("Need either 'price' or 'micro_price' in features CSV")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df[df[price_col].notna()].copy()

    if "size" not in df.columns:
        df["size"] = 0.0
    df["size"] = pd.to_numeric(df["size"], errors="coerce").fillna(0.0)

    for col in (
        "cvd",
        "obi",
        "absorption_intensity",
        "kyle_lambda",
        "hawkes_intensity",
        "bias_label",
        "regime_label",
    ):
        if col not in df.columns:
            df[col] = 0.0 if col not in ("bias_label", "regime_label") else 2

    bars = resample_feature_bars(df, price_col=price_col, freq=freq)
    if len(bars) > max_bars:
        bars = bars.iloc[-max_bars:].reset_index(drop=True)

    t_min = bars["ts_event"].min()
    t_max = bars["ts_event"].max() + pd.tseries.frequencies.to_offset(freq)

    mbo = None
    if mbo_path and Path(mbo_path).exists():
        print(f"📥 قراءة MBO: {mbo_path}")
        mbo = _load_csv(mbo_path)
        for col in ("price", "size", "side", "action"):
            if col not in mbo.columns:
                mbo[col] = 0.0 if col in ("price", "size") else ""
        mbo["price"] = pd.to_numeric(mbo["price"], errors="coerce")
        mbo["size"] = pd.to_numeric(mbo["size"], errors="coerce").fillna(0.0)
        mbo = mbo[(mbo["ts_event"] >= t_min) & (mbo["ts_event"] <= t_max)]
        mbo = mbo[mbo["action"].astype(str).str.upper().isin({"T", "F", "TRADE"})].copy()
        if len(mbo) > max_trade_points:
            step = max(1, len(mbo) // max_trade_points)
            mbo = mbo.iloc[::step].reset_index(drop=True)

    mbp = None
    if mbp_path and Path(mbp_path).exists():
        print(f"📥 قراءة MBP: {mbp_path}")
        mbp = _load_csv(mbp_path)
        mbp = mbp[(mbp["ts_event"] >= t_min) & (mbp["ts_event"] <= t_max)].copy()

    signal_col = "cb_direction_idx" if models_dir and "cb_direction_idx" in bars.columns else "bias_label"
    signal_name = "CatBoost" if signal_col == "cb_direction_idx" else "Labels"

    print(f"  ✅ {len(bars):,} شمعة {freq} | من {bars['ts_event'].min()} إلى {bars['ts_event'].max()}")
    print(f"  {signal_name}: {bars[signal_col].value_counts().to_dict()}")
    return {
        "bars": bars,
        "mbo": mbo,
        "mbp": mbp,
        "price_col": price_col,
        "freq": freq,
        "signal_col": signal_col,
        "signal_name": signal_name,
        "models_dir": models_dir,
    }


def _mode_or_default(series: pd.Series, default_value) -> object:
    s = series.dropna()
    if s.empty:
        return default_value
    mode = s.mode(dropna=True)
    if len(mode) == 0:
        return default_value
    return mode.iloc[0]


def _dominant_bias(series: pd.Series) -> int:
    s = pd.to_numeric(series, errors="coerce").dropna().astype(int)
    if s.empty:
        return 2
    counts = s.value_counts()
    if counts.empty:
        return 2
    best = int(counts.index[0])
    best_count = int(counts.iloc[0])
    total = int(counts.sum())
    if total <= 0:
        return 2
    if best == 2 and len(counts) > 1:
        alt = int(counts.index[1])
        alt_count = int(counts.iloc[1])
        if alt_count / total >= 0.35:
            return alt
    if best_count / total < 0.35:
        return 2
    return best


def _regime_to_name(value) -> str:
    try:
        iv = int(value)
        return REGIME_LABELS.get(iv, str(value))
    except Exception:
        return str(value)


def resample_feature_bars(df: pd.DataFrame, price_col: str, freq: str) -> pd.DataFrame:
    freq = _normalize_freq(freq)
    base = df.copy().set_index("ts_event")
    offset = pd.tseries.frequencies.to_offset(freq)

    ohlc = base[price_col].resample(freq).ohlc()
    volume = base["size"].resample(freq).sum().rename("volume")
    event_count = base[price_col].resample(freq).size().rename("event_count")

    cvd_last = pd.to_numeric(base["cvd"], errors="coerce").resample(freq).last().rename("cvd")
    cvd_first = pd.to_numeric(base["cvd"], errors="coerce").resample(freq).first().rename("cvd_first")
    cvd_delta = (cvd_last - cvd_first).rename("cvd_delta")

    obi = pd.to_numeric(base["obi"], errors="coerce").resample(freq).mean().rename("obi")
    absorption = pd.to_numeric(base["absorption_intensity"], errors="coerce").resample(freq).mean().rename("absorption_intensity")
    kyle = pd.to_numeric(base["kyle_lambda"], errors="coerce").resample(freq).mean().rename("kyle_lambda")
    hawkes = pd.to_numeric(base["hawkes_intensity"], errors="coerce").resample(freq).mean().rename("hawkes_intensity")

    bias = base["bias_label"].resample(freq).apply(_dominant_bias).rename("bias_label")
    regime = base["regime_label"].resample(freq).apply(lambda s: _mode_or_default(s, 1)).rename("regime_label")

    bars = pd.concat(
        [ohlc, volume, event_count, cvd_last, cvd_delta, obi, absorption, kyle, hawkes, bias, regime],
        axis=1,
    ).dropna(subset=["open", "high", "low", "close"])

    bars["bias_label"] = pd.to_numeric(bars["bias_label"], errors="coerce").fillna(2).astype(int)
    bars["regime_label"] = bars["regime_label"].apply(_regime_to_name)
    bars["label_name"] = bars["bias_label"].map(BIAS_LABELS).fillna("NEUTRAL")
    bars["label_strength"] = (
        pd.concat(
            [
                bars["obi"].abs().fillna(0.0),
                bars["absorption_intensity"].abs().fillna(0.0),
                bars["hawkes_intensity"].abs().fillna(0.0),
            ],
            axis=1,
        ).mean(axis=1)
    ).astype(float)

    if {"cb_prob_long", "cb_prob_short", "cb_prob_neutral"}.issubset(base.columns):
        cb_probs = base[["cb_prob_long", "cb_prob_short", "cb_prob_neutral"]].resample(freq).mean()
        bars = pd.concat([bars, cb_probs], axis=1)
        prob_cols = ["cb_prob_long", "cb_prob_short", "cb_prob_neutral"]
        prob_matrix = bars[prob_cols].fillna(0.0).values
        bars["cb_direction_idx"] = np.argmax(prob_matrix, axis=1).astype(int)
        bars["cb_direction"] = bars["cb_direction_idx"].map(BIAS_LABELS).fillna("NEUTRAL")
        bars["cb_confidence"] = prob_matrix.max(axis=1).astype(float)
        bars["cb_change_flag"] = (bars["cb_direction"] != bars["cb_direction"].shift(1)).astype(int)
        bars["signal_time"] = pd.to_datetime(bars.index) + offset

    return bars.reset_index()


def compute_signal_stats(bars: pd.DataFrame, signal_col: str = "bias_label", future_bars: int = 4) -> dict:
    total = len(bars)
    bc = bars[signal_col].value_counts()

    n_long = int(bc.get(0, 0))
    n_short = int(bc.get(1, 0))
    n_neutral = int(bc.get(2, 0))

    correct_long = correct_short = total_long = total_short = 0
    closes = bars["close"].values.astype(float)
    labels = bars[signal_col].values.astype(int)
    for i in range(len(bars) - future_bars):
        lbl = labels[i]
        if lbl == 2:
            continue
        future_return = closes[i + future_bars] - closes[i]
        if lbl == 0:
            total_long += 1
            if future_return > 0:
                correct_long += 1
        elif lbl == 1:
            total_short += 1
            if future_return < 0:
                correct_short += 1

    regime_dist = bars["regime_label"].value_counts().to_dict() if "regime_label" in bars.columns else {}
    return {
        "total": total,
        "n_long": n_long,
        "n_short": n_short,
        "n_neutral": n_neutral,
        "pct_long": n_long / max(total, 1) * 100,
        "pct_short": n_short / max(total, 1) * 100,
        "pct_neutral": n_neutral / max(total, 1) * 100,
        "long_acc": correct_long / max(total_long, 1) * 100,
        "short_acc": correct_short / max(total_short, 1) * 100,
        "regime_dist": regime_dist,
    }


def build_dashboard(data: dict, stats: dict) -> go.Figure:
    bars = data["bars"]
    mbo = data["mbo"]
    signal_col = data.get("signal_col", "bias_label")
    signal_name = data.get("signal_name", "Labels")
    is_catboost = signal_col == "cb_direction_idx"
    strength_col = "cb_confidence" if is_catboost and "cb_confidence" in bars.columns else "label_strength"
    signal_time_col = "signal_time" if is_catboost and "signal_time" in bars.columns else "ts_event"

    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.36, 0.13, 0.13, 0.12, 0.12, 0.14],
        vertical_spacing=0.02,
        subplot_titles=[
            f"① 5m Candles + {signal_name} Overlay",
            "② CVD Delta (5m)",
            "③ OBI Mean (5m)",
            "④ Absorption Mean (5m)",
            "⑤ Kyle's Lambda (5m)",
            "⑥ Regime + Hawkes Intensity (5m)",
        ],
    )

    ts = bars["ts_event"]
    signal_ts = pd.to_datetime(bars[signal_time_col]) if signal_time_col in bars.columns else ts
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

    long_mask = bars[signal_col] == 0
    short_mask = bars[signal_col] == 1
    neutral_mask = bars[signal_col] == 2
    marker_symbol_long = "diamond" if is_catboost else "triangle-up"
    marker_symbol_short = "diamond-wide" if is_catboost else "triangle-down"
    neutral_symbol = "square" if is_catboost else "circle"

    if is_catboost and "cb_change_flag" in bars.columns:
        change_rows = bars[bars["cb_change_flag"] == 1]
        for row in change_rows.itertuples(index=False):
            fig.add_vline(
                x=row.signal_time,
                line_dash="dash",
                line_color="rgba(148,163,184,0.28)",
                line_width=1,
                row=1,
                col=1,
            )

    if long_mask.any():
        fig.add_trace(
            go.Scatter(
                x=signal_ts[long_mask],
                y=bars.loc[long_mask, "low"] * 0.9997,
                mode="markers",
                marker=dict(symbol=marker_symbol_long, size=12 if is_catboost else 11, color="#00e896", line=dict(color="#008f63", width=1)),
                name=f"{signal_name} LONG ({int(long_mask.sum())})",
                customdata=np.stack(
                    [
                        bars.loc[long_mask, "close"].values,
                        bars.loc[long_mask, strength_col].fillna(0.0).values,
                    ],
                    axis=1,
                ),
                hovertemplate=f"{signal_name} LONG<br>%{{x}}<br>Close=%{{customdata[0]:.5f}}<br>Score=%{{customdata[1]:.3f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if short_mask.any():
        fig.add_trace(
            go.Scatter(
                x=signal_ts[short_mask],
                y=bars.loc[short_mask, "high"] * 1.0003,
                mode="markers",
                marker=dict(symbol=marker_symbol_short, size=12 if is_catboost else 11, color="#ff4d6d", line=dict(color="#b91c3f", width=1)),
                name=f"{signal_name} SHORT ({int(short_mask.sum())})",
                customdata=np.stack(
                    [
                        bars.loc[short_mask, "close"].values,
                        bars.loc[short_mask, strength_col].fillna(0.0).values,
                    ],
                    axis=1,
                ),
                hovertemplate=f"{signal_name} SHORT<br>%{{x}}<br>Close=%{{customdata[0]:.5f}}<br>Score=%{{customdata[1]:.3f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if neutral_mask.any():
        fig.add_trace(
            go.Scatter(
                x=signal_ts[neutral_mask],
                y=bars.loc[neutral_mask, "close"],
                mode="markers",
                marker=dict(symbol=neutral_symbol, size=6 if is_catboost else 5, color="rgba(200,216,232,0.35)"),
                name=f"{signal_name} NEUTRAL ({int(neutral_mask.sum())})",
                hovertemplate=f"{signal_name} NEUTRAL<br>%{{x}}<br>Close=%{{y:.5f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if mbo is not None and len(mbo) > 0:
        sides = mbo["side"].astype(str).str.upper()
        buy_mbo = mbo[sides.isin({"A", "ASK", "BUY", "BOT"})]
        sell_mbo = mbo[sides.isin({"B", "BID", "S", "SELL"})]

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

    obi_colors = [
        "#00e896" if v > 0.2 else "#ff4d6d" if v < -0.2 else "#58657a"
        for v in bars["obi"].fillna(0.0)
    ]
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

    regime_numeric = {
        "Trending": 1.0,
        "Volatile": 0.75,
        "Ranging": 0.50,
        "Low_Liquidity": 0.25,
    }
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
        f"<b>5m {signal_name} Statistics</b><br>"
        f"LONG: {stats['n_long']:,} ({stats['pct_long']:.1f}%) | Fwd Acc: {stats['long_acc']:.0f}%<br>"
        f"SHORT: {stats['n_short']:,} ({stats['pct_short']:.1f}%) | Fwd Acc: {stats['short_acc']:.0f}%<br>"
        f"NEUTRAL: {stats['n_neutral']:,} ({stats['pct_neutral']:.1f}%)"
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

    fig.update_layout(
        title=dict(
            text=f"QS V19 — 5m {signal_name} Visualization Dashboard",
            font=dict(size=18, color="#ffffff", family="IBM Plex Mono"),
            x=0.02,
        ),
        paper_bgcolor="#060a0f",
        plot_bgcolor="#0d1520",
        font=dict(color="#c8d8e8", family="IBM Plex Mono"),
        height=1280,
        showlegend=True,
        legend=dict(
            bgcolor="rgba(13,21,32,0.90)",
            bordercolor="#1e3048",
            borderwidth=1,
            font=dict(size=10),
            x=0.0,
            y=1.0,
            orientation="h",
        ),
        hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=False), type="date"),
    )

    for i in range(1, 7):
        fig.update_xaxes(showgrid=True, gridcolor="#1e3048", gridwidth=0.5, zeroline=False, row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#1e3048", gridwidth=0.5, zeroline=False, row=i, col=1)

    return fig


def build_confusion_chart(bars: pd.DataFrame, signal_col: str = "bias_label", signal_name: str = "Labels", future_bars: int = 4) -> go.Figure | None:
    if "close" not in bars.columns or signal_col not in bars.columns:
        return None

    closes = bars["close"].values.astype(float)
    labels = bars[signal_col].values.astype(int)
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
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Return Distribution (5m labels)", "Rolling Label Accuracy"])
    fig.layout.annotations[0].update(text=f"Return Distribution (5m {signal_name})")
    fig.layout.annotations[1].update(text=f"Rolling {signal_name} Accuracy")

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
            name="Accuracy (rolling 30)",
        ),
        row=1,
        col=2,
    )
    fig.add_hline(y=50, line_dash="dash", line_color="#555", row=1, col=2)
    fig.update_layout(
        paper_bgcolor="#060a0f",
        plot_bgcolor="#0d1520",
        font=dict(color="#c8d8e8", family="IBM Plex Mono"),
        title=dict(text=f"5m {signal_name} Quality Analysis", font=dict(color="#ffffff", size=16)),
        height=430,
        barmode="overlay",
    )
    return fig


def build_turns_table(bars: pd.DataFrame, signal_col: str, signal_name: str) -> pd.DataFrame:
    direction_names = bars[signal_col].map(BIAS_LABELS).fillna("NEUTRAL")
    change_mask = direction_names != direction_names.shift(1)
    out = bars.loc[change_mask, ["ts_event", "open", "high", "low", "close"]].copy()
    out["signal_name"] = signal_name
    out["direction"] = direction_names.loc[change_mask].values
    out["prev_direction"] = direction_names.shift(1).loc[change_mask].fillna("START").values
    if signal_col == "cb_direction_idx" and "cb_confidence" in bars.columns:
        out["confidence"] = bars.loc[change_mask, "cb_confidence"].values
    else:
        out["confidence"] = bars.loc[change_mask, "label_strength"].values if "label_strength" in bars.columns else 0.0
    return out.reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description="QS V19 5m Label Visualizer")
    p.add_argument("--features", required=True, help="مسار training_features_ready.csv")
    p.add_argument("--mbo", default="", help="مسار raw MBO csv (اختياري)")
    p.add_argument("--mbp", default="", help="مسار raw MBP10 csv (اختياري)")
    p.add_argument("--models_dir", default="", help="إذا وُجد، يعرض قرارات CatBoost بدل labels")
    p.add_argument("--output", default="label_viz_5m.html", help="ملف الإخراج HTML")
    p.add_argument("--freq", default="5min", help="الفريم للتجميع، الافتراضي 5min")
    p.add_argument("--bars", type=int, default=400, help="عدد الشموع للرسم")
    p.add_argument("--future_bars", type=int, default=4, help="عدد شموع 5m للأمام لفحص جودة الليبل")
    a = p.parse_args()

    data = load_and_prepare(
        a.features,
        mbo_path=a.mbo,
        mbp_path=a.mbp,
        models_dir=a.models_dir,
        freq=a.freq,
        max_bars=a.bars,
    )
    stats = compute_signal_stats(data["bars"], signal_col=data["signal_col"], future_bars=a.future_bars)

    print(f"\n📊 إحصائيات {data['signal_name']} على 5m:")
    print(f"   LONG:    {stats['n_long']:,} ({stats['pct_long']:.1f}%) | Fwd Acc: {stats['long_acc']:.0f}%")
    print(f"   SHORT:   {stats['n_short']:,} ({stats['pct_short']:.1f}%) | Fwd Acc: {stats['short_acc']:.0f}%")
    print(f"   NEUTRAL: {stats['n_neutral']:,} ({stats['pct_neutral']:.1f}%)")

    print("\n🎨 بناء Dashboard...")
    fig_main = build_dashboard(data, stats)
    fig_conf = build_confusion_chart(
        data["bars"],
        signal_col=data["signal_col"],
        signal_name=data["signal_name"],
        future_bars=a.future_bars,
    )
    turns = build_turns_table(data["bars"], signal_col=data["signal_col"], signal_name=data["signal_name"])

    with open(a.output, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'>")
        f.write(f"<title>QS V19 5m {data['signal_name']} Visualizer</title>")
        f.write("<style>body{background:#060a0f;margin:0;padding:10px;}</style>")
        f.write("</head><body>")
        f.write(fig_main.to_html(full_html=False, include_plotlyjs="cdn"))
        if fig_conf is not None:
            f.write("<br>")
            f.write(fig_conf.to_html(full_html=False, include_plotlyjs=False))
        f.write("</body></html>")

    turns_path = str(Path(a.output).with_name(Path(a.output).stem + "_turns.csv"))
    turns.to_csv(turns_path, index=False)

    print(f"\n✅ Dashboard محفوظ: {a.output}")
    print(f"✅ Turns CSV محفوظ: {turns_path}")
    print(f"   افتح في المتصفح: {Path(a.output).resolve()}")


if __name__ == "__main__":
    main()
