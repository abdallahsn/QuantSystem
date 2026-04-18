"""
refinery_compare_5m_report.py
=============================
Daily 5m comparison dashboard for:
  1. Raw market candles (from MBO trades, with optional MBP mid fallback)
  2. Refinery labels/events/regime
  3. Refinery context metrics

The implementation is intentionally chunk-friendly so it can handle large
monthly files without loading the full raw dataset into memory.
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


TRADE_ACTIONS = {"T", "F", "TRADE", "EXECUTE", "0"}
BIAS_LABELS = {0: "LONG", 1: "SHORT", 2: "NEUTRAL"}
QUALITY_LABELS = {0: "NONE", 1: "WEAK", 2: "STRONG"}
REGIME_LABELS = {
    0: "Trending",
    1: "Ranging",
    2: "Volatile",
    3: "Low_Liquidity",
}
REGIME_COLORS = {
    "Trending": "rgba(0,232,150,0.16)",
    "Ranging": "rgba(107,114,128,0.12)",
    "Volatile": "rgba(255,77,109,0.16)",
    "Low_Liquidity": "rgba(59,130,246,0.12)",
}


def _normalize_freq(freq: str) -> str:
    value = str(freq or "5min").strip()
    if not value:
        return "5min"
    return value.replace("T", "min").replace("H", "h")


def _normalize_timestamp_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_localize(None)


def _normalize_bound(value):
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        return ts.tz_convert(None)
    return ts.tz_localize(None)


def _iter_market_chunks(path: str, columns: Iterable[str], chunksize: int) -> Iterable[pd.DataFrame]:
    ext = Path(path).suffix.lower()
    wanted = set(columns)

    if ext in {".parquet", ".pq", ".snappy"}:
        df = pd.read_parquet(path)
        keep = [c for c in df.columns if c in wanted]
        yield df[keep].copy()
        return

    reader = pd.read_csv(
        path,
        low_memory=False,
        compression="infer",
        chunksize=max(int(chunksize), 10_000),
        usecols=lambda c: c in wanted,
    )
    for chunk in reader:
        yield chunk


def _filter_timeframe(df: pd.DataFrame, start_ts=None, end_ts=None) -> pd.DataFrame:
    out = df.copy()
    if "ts_event" not in out.columns and "ts_recv" in out.columns:
        out["ts_event"] = out["ts_recv"]
    if "ts_event" not in out.columns:
        return out.iloc[0:0].copy()

    out["ts_event"] = _normalize_timestamp_series(out["ts_event"])
    out = out[out["ts_event"].notna()]

    start_bound = _normalize_bound(start_ts)
    end_bound = _normalize_bound(end_ts)
    if start_bound is not None:
        out = out[out["ts_event"] >= start_bound]
    if end_bound is not None:
        out = out[out["ts_event"] < end_bound]
    return out.sort_values("ts_event")


def _write_table_with_fallback(df: pd.DataFrame, preferred_path: str) -> str:
    try:
        df.to_parquet(preferred_path, index=False)
        return preferred_path
    except Exception:
        fallback = os.path.splitext(preferred_path)[0] + ".csv"
        df.to_csv(fallback, index=False)
        return fallback


def _resample_ohlc(df: pd.DataFrame, price_col: str, freq: str, size_col: str = "size", count_name: str = "tick_count") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", count_name])

    base = df.set_index("ts_event").sort_index()
    ohlc = base[price_col].resample(freq).ohlc()
    volume = base[size_col].resample(freq).sum().rename("volume")
    counts = base[price_col].resample(freq).size().rename(count_name)
    bars = pd.concat([ohlc, volume, counts], axis=1)
    return bars.dropna(subset=["open", "high", "low", "close"])


def _combine_ohlc_partials(partials: list[pd.DataFrame], count_name: str = "tick_count") -> pd.DataFrame:
    if not partials:
        return pd.DataFrame(columns=["ts_event", "open", "high", "low", "close", "volume", count_name])
    merged = pd.concat(partials).sort_index()
    out = (
        merged.groupby(level=0)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            **{count_name: (count_name, "sum")},
        )
        .reset_index()
        .rename(columns={"index": "ts_event"})
    )
    out["ts_event"] = pd.to_datetime(out["ts_event"], errors="coerce")
    return out.sort_values("ts_event").reset_index(drop=True)


def _aggregate_raw_bars(
    mbo_path: str,
    mbp_path: str = "",
    freq: str = "5min",
    chunksize: int = 500_000,
    start_ts=None,
    end_ts=None,
) -> pd.DataFrame:
    freq = _normalize_freq(freq)
    trade_partials: list[pd.DataFrame] = []
    trade_cols = {"ts_event", "ts_recv", "action", "price", "size"}

    for chunk in _iter_market_chunks(mbo_path, trade_cols, chunksize):
        chunk = _filter_timeframe(chunk, start_ts=start_ts, end_ts=end_ts)
        if chunk.empty:
            continue
        if "action" not in chunk.columns:
            chunk["action"] = "T"
        chunk["action"] = chunk["action"].astype(str).str.strip().str.upper()
        chunk = chunk[chunk["action"].isin(TRADE_ACTIONS)]
        if chunk.empty:
            continue
        chunk["price"] = pd.to_numeric(chunk.get("price"), errors="coerce")
        chunk["size"] = pd.to_numeric(chunk.get("size", 0.0), errors="coerce").fillna(0.0)
        chunk = chunk[chunk["price"].notna() & (chunk["price"] > 0)]
        if chunk.empty:
            continue
        trade_partials.append(_resample_ohlc(chunk[["ts_event", "price", "size"]].copy(), "price", freq=freq))

    raw_trades = _combine_ohlc_partials(trade_partials, count_name="tick_count")
    raw_trades["source"] = "MBO"

    if not mbp_path:
        return raw_trades

    mid_partials: list[pd.DataFrame] = []
    mbp_cols = {"ts_event", "ts_recv", "price", "bid_px_00", "ask_px_00"}
    for chunk in _iter_market_chunks(mbp_path, mbp_cols, chunksize):
        chunk = _filter_timeframe(chunk, start_ts=start_ts, end_ts=end_ts)
        if chunk.empty:
            continue
        bid0 = pd.to_numeric(chunk.get("bid_px_00"), errors="coerce")
        ask0 = pd.to_numeric(chunk.get("ask_px_00"), errors="coerce")
        mid = (bid0.where(bid0 > 0, np.nan) + ask0.where(ask0 > 0, np.nan)) / 2.0
        if mid.isna().all():
            mid = pd.to_numeric(chunk.get("price"), errors="coerce")
        chunk["mid_price"] = mid
        chunk = chunk[chunk["mid_price"].notna() & (chunk["mid_price"] > 0)]
        if chunk.empty:
            continue
        chunk["size"] = 0.0
        mbp_bars = _resample_ohlc(chunk[["ts_event", "mid_price", "size"]].copy(), "mid_price", freq=freq, count_name="snapshot_count")
        mid_partials.append(mbp_bars)

    raw_mid = _combine_ohlc_partials(mid_partials, count_name="snapshot_count")
    if raw_mid.empty:
        return raw_trades

    raw_mid["source"] = "MBP"
    if raw_trades.empty:
        raw_mid["tick_count"] = 0
        raw_mid["volume"] = 0.0
        return raw_mid

    merged = pd.merge(raw_mid, raw_trades, on="ts_event", how="outer", suffixes=("_mbp", "_mbo"))
    out = pd.DataFrame({"ts_event": merged["ts_event"]})
    for col in ("open", "high", "low", "close"):
        out[col] = merged[f"{col}_mbo"].combine_first(merged[f"{col}_mbp"])
    out["volume"] = merged.get("volume_mbo", pd.Series(np.zeros(len(merged)))).fillna(0.0)
    out["tick_count"] = merged.get("tick_count", pd.Series(np.zeros(len(merged)))).fillna(0).astype(int)
    out["snapshot_count"] = merged.get("snapshot_count", pd.Series(np.zeros(len(merged)))).fillna(0).astype(int)
    out["source"] = np.where(merged["open_mbo"].notna(), "MBO", "MBP")
    return out.dropna(subset=["open", "high", "low", "close"]).sort_values("ts_event").reset_index(drop=True)


def _dominant_bias_from_counts(row: pd.Series) -> int:
    counts = np.array([row.get("bias_0", 0), row.get("bias_1", 0), row.get("bias_2", 0)], dtype=float)
    total = counts.sum()
    if total <= 0:
        return 2
    best = int(counts.argmax())
    best_count = float(counts[best])
    if best == 2:
        alt = int(np.array([counts[0], counts[1]]).argmax())
        alt_count = float(counts[alt])
        if alt_count / total >= 0.35:
            return alt
    if best_count / total < 0.35:
        return 2
    return best


def _quality_from_counts(row: pd.Series) -> int:
    if row.get("qual_2", 0) > 0:
        return 2
    if row.get("qual_1", 0) > 0:
        return 1
    return 0


def _regime_from_counts(row: pd.Series) -> int:
    counts = np.array([row.get("reg_0", 0), row.get("reg_1", 0), row.get("reg_2", 0), row.get("reg_3", 0)], dtype=float)
    if counts.sum() <= 0:
        return 1
    return int(counts.argmax())


def _aggregate_refinery_bars(
    csv_path: str,
    freq: str = "5min",
    chunksize: int = 250_000,
    start_ts=None,
    end_ts=None,
) -> pd.DataFrame:
    freq = _normalize_freq(freq)
    needed = {
        "ts_event",
        "price",
        "event_flag",
        "bias_label",
        "signal_quality",
        "regime_label",
        "liq_score",
        "obi",
        "cvd",
        "forward_return",
    }

    partials: list[pd.DataFrame] = []
    for chunk in _iter_market_chunks(csv_path, needed, chunksize):
        chunk = _filter_timeframe(chunk, start_ts=start_ts, end_ts=end_ts)
        if chunk.empty:
            continue

        chunk["price"] = pd.to_numeric(chunk.get("price"), errors="coerce")
        chunk["event_flag"] = pd.to_numeric(chunk.get("event_flag", 0), errors="coerce").fillna(0).clip(lower=0)
        chunk["bias_label"] = pd.to_numeric(chunk.get("bias_label", 2), errors="coerce").fillna(2).astype(int)
        chunk["signal_quality"] = pd.to_numeric(chunk.get("signal_quality", 0), errors="coerce").fillna(0).astype(int)
        chunk["regime_label"] = pd.to_numeric(chunk.get("regime_label", 1), errors="coerce").fillna(1).astype(int)
        chunk["liq_score"] = pd.to_numeric(chunk.get("liq_score", 0.0), errors="coerce").fillna(0.0)
        chunk["obi"] = pd.to_numeric(chunk.get("obi", 0.0), errors="coerce").fillna(0.0)
        chunk["cvd"] = pd.to_numeric(chunk.get("cvd", 0.0), errors="coerce").fillna(0.0)
        chunk["forward_return"] = pd.to_numeric(chunk.get("forward_return", 0.0), errors="coerce").fillna(0.0)
        chunk = chunk[chunk["ts_event"].notna()]
        if chunk.empty:
            continue

        base = chunk.set_index("ts_event").sort_index()
        frame = pd.DataFrame(index=base.resample(freq).size().index)
        frame["event_count"] = base["event_flag"].resample(freq).sum()
        frame["bar_count"] = base["price"].resample(freq).size()
        frame["liq_sum"] = base["liq_score"].resample(freq).sum()
        frame["liq_count"] = base["liq_score"].resample(freq).count()
        frame["obi_sum"] = base["obi"].resample(freq).sum()
        frame["obi_count"] = base["obi"].resample(freq).count()
        frame["cvd_first"] = base["cvd"].resample(freq).first()
        frame["cvd_last"] = base["cvd"].resample(freq).last()
        frame["fwd_sum"] = base["forward_return"].resample(freq).sum()
        frame["fwd_count"] = base["forward_return"].resample(freq).count()

        for label in (0, 1, 2):
            frame[f"bias_{label}"] = (base["bias_label"] == label).astype(int).resample(freq).sum()
            frame[f"qual_{label}"] = (base["signal_quality"] == label).astype(int).resample(freq).sum()
        for label in (0, 1, 2, 3):
            frame[f"reg_{label}"] = (base["regime_label"] == label).astype(int).resample(freq).sum()

        frame = frame.dropna(how="all")
        partials.append(frame)

    if not partials:
        return pd.DataFrame(
            columns=[
                "ts_event",
                "event_count",
                "event_flag",
                "bias_label",
                "bias_name",
                "signal_quality",
                "quality_name",
                "regime_label",
                "regime_name",
                "liq_score",
                "obi",
                "cvd_delta",
                "forward_return_mean",
            ]
        )

    merged = pd.concat(partials).sort_index()
    agg_map: dict[str, tuple[str, str]] = {
        "event_count": ("event_count", "sum"),
        "bar_count": ("bar_count", "sum"),
        "liq_sum": ("liq_sum", "sum"),
        "liq_count": ("liq_count", "sum"),
        "obi_sum": ("obi_sum", "sum"),
        "obi_count": ("obi_count", "sum"),
        "cvd_first": ("cvd_first", "first"),
        "cvd_last": ("cvd_last", "last"),
        "fwd_sum": ("fwd_sum", "sum"),
        "fwd_count": ("fwd_count", "sum"),
    }
    for label in (0, 1, 2):
        agg_map[f"bias_{label}"] = (f"bias_{label}", "sum")
        agg_map[f"qual_{label}"] = (f"qual_{label}", "sum")
    for label in (0, 1, 2, 3):
        agg_map[f"reg_{label}"] = (f"reg_{label}", "sum")

    bars = merged.groupby(level=0).agg(**agg_map).reset_index().rename(columns={"index": "ts_event"})
    bars["ts_event"] = pd.to_datetime(bars["ts_event"], errors="coerce")
    bars["event_flag"] = (bars["event_count"] > 0).astype(int)
    bars["liq_score"] = bars["liq_sum"] / np.maximum(bars["liq_count"], 1)
    bars["obi"] = bars["obi_sum"] / np.maximum(bars["obi_count"], 1)
    bars["cvd_delta"] = bars["cvd_last"] - bars["cvd_first"]
    bars["forward_return_mean"] = bars["fwd_sum"] / np.maximum(bars["fwd_count"], 1)
    bars["bias_label"] = bars.apply(_dominant_bias_from_counts, axis=1).astype(int)
    bars["signal_quality"] = bars.apply(_quality_from_counts, axis=1).astype(int)
    bars["regime_label"] = bars.apply(_regime_from_counts, axis=1).astype(int)
    bars["bias_name"] = bars["bias_label"].map(BIAS_LABELS).fillna("NEUTRAL")
    bars["quality_name"] = bars["signal_quality"].map(QUALITY_LABELS).fillna("NONE")
    bars["regime_name"] = bars["regime_label"].map(REGIME_LABELS).fillna("Ranging")
    return bars.sort_values("ts_event").reset_index(drop=True)


def _build_candle_hover_trace(page_df: pd.DataFrame) -> go.Scatter:
    y = ((page_df["high"] + page_df["low"]) / 2.0).fillna(page_df["close"]).astype(float)
    customdata = np.column_stack(
        [
            page_df["open"].astype(float).values,
            page_df["high"].astype(float).values,
            page_df["low"].astype(float).values,
            page_df["close"].astype(float).values,
            page_df["volume"].fillna(0.0).astype(float).values,
            page_df["tick_count"].fillna(0).astype(int).values,
            page_df.get("snapshot_count", pd.Series(np.zeros(len(page_df)))).fillna(0).astype(int).values,
            page_df["source"].fillna("MBO").astype(str).values,
            page_df["event_flag"].fillna(0).astype(int).values,
            page_df["bias_name"].fillna("NEUTRAL").astype(str).values,
            page_df["quality_name"].fillna("NONE").astype(str).values,
            page_df["regime_name"].fillna("Ranging").astype(str).values,
            page_df["liq_score"].fillna(0.0).astype(float).values,
            page_df["obi"].fillna(0.0).astype(float).values,
            page_df["cvd_delta"].fillna(0.0).astype(float).values,
        ]
    )
    return go.Scatter(
        x=page_df["ts_event"],
        y=y,
        mode="markers",
        marker=dict(size=18, color="rgba(0,0,0,0)"),
        name="Raw Candle Data",
        showlegend=False,
        customdata=customdata,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Open=%{customdata[0]:,.5f}<br>"
            "High=%{customdata[1]:,.5f}<br>"
            "Low=%{customdata[2]:,.5f}<br>"
            "Close=%{customdata[3]:,.5f}<br>"
            "Volume=%{customdata[4]:,.2f}<br>"
            "Trade Ticks=%{customdata[5]}<br>"
            "MBP Snapshots=%{customdata[6]}<br>"
            "Price Source=%{customdata[7]}<br>"
            "Event Flag=%{customdata[8]}<br>"
            "Bias=%{customdata[9]}<br>"
            "Quality=%{customdata[10]}<br>"
            "Regime=%{customdata[11]}<br>"
            "Liq Score=%{customdata[12]:.2f}<br>"
            "OBI=%{customdata[13]:.3f}<br>"
            "CVD Δ=%{customdata[14]:.3f}<extra></extra>"
        ),
    )


def _quality_marker_size(quality: pd.Series) -> np.ndarray:
    return quality.map({0: 6, 1: 11, 2: 16}).fillna(6).astype(float).values


def _build_daily_compare_figure(raw_day: pd.DataFrame, ref_day: pd.DataFrame, freq: str, day_label: str) -> go.Figure:
    bars = pd.merge(raw_day, ref_day, on="ts_event", how="outer").sort_values("ts_event").reset_index(drop=True)
    offset = pd.tseries.frequencies.to_offset(freq)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.56, 0.24, 0.20],
        vertical_spacing=0.03,
        subplot_titles=[
            f"① Raw Market Candles (5m) — {day_label}",
            "② Refinery Labels / Events / Regime",
            "③ Refinery Context (CVD Δ + Liquidity)",
        ],
        specs=[[{}], [{}], [{"secondary_y": True}]],
    )

    candle_mask = bars["open"].notna()
    if candle_mask.any():
        candle_df = bars.loc[candle_mask].copy()
        fig.add_trace(
            go.Candlestick(
                x=candle_df["ts_event"],
                open=candle_df["open"],
                high=candle_df["high"],
                low=candle_df["low"],
                close=candle_df["close"],
                name="Raw 5m",
                increasing_line_color="#00d3b6",
                decreasing_line_color="#ff4d6d",
                increasing_fillcolor="rgba(0,211,182,0.34)",
                decreasing_fillcolor="rgba(255,77,109,0.30)",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(_build_candle_hover_trace(candle_df), row=1, col=1)

        last_close = float(candle_df["close"].iloc[-1])
        last_open = float(candle_df["open"].iloc[-1])
        last_color = "#00d3b6" if last_close >= last_open else "#ff4d6d"
        fig.add_hline(y=last_close, line_dash="dot", line_color=last_color, line_width=1, row=1, col=1)
        fig.add_annotation(
            x=candle_df["ts_event"].iloc[-1],
            y=last_close,
            text=f"{last_close:,.5f}",
            showarrow=False,
            xshift=42,
            bgcolor=last_color,
            bordercolor=last_color,
            font=dict(size=11, color="#ffffff", family="IBM Plex Mono"),
            row=1,
            col=1,
        )

    if not ref_day.empty:
        for row in ref_day.itertuples(index=False):
            fig.add_vrect(
                x0=row.ts_event,
                x1=row.ts_event + offset,
                fillcolor=REGIME_COLORS.get(row.regime_name, "rgba(107,114,128,0.10)"),
                opacity=1.0,
                line_width=0,
                row=2,
                col=1,
            )

        fig.add_trace(
            go.Bar(
                x=ref_day["ts_event"],
                y=ref_day["event_flag"].fillna(0).astype(float) * 0.45,
                name="Event Flag",
                marker_color="rgba(59,130,246,0.45)",
                hovertemplate="%{x}<br>Event=%{y:.0f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=ref_day["ts_event"],
                y=ref_day["obi"].clip(-1.0, 1.0),
                mode="lines",
                line=dict(color="#8b5cf6", width=1.6),
                name="OBI",
                hovertemplate="%{x}<br>OBI=%{y:.3f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        for label, idx, y_val, symbol, color in (
            ("LONG", 0, 1.55, "triangle-up", "#00e896"),
            ("SHORT", 1, -1.55, "triangle-down", "#ff4d6d"),
            ("NEUTRAL", 2, 0.0, "circle", "rgba(203,213,225,0.60)"),
        ):
            mask = ref_day["bias_label"] == idx
            if not bool(mask.any()):
                continue
            y = np.full(mask.sum(), y_val, dtype=float)
            fig.add_trace(
                go.Scatter(
                    x=ref_day.loc[mask, "ts_event"],
                    y=y,
                    mode="markers",
                    name=f"Bias {label}",
                    marker=dict(
                        symbol=symbol,
                        size=_quality_marker_size(ref_day.loc[mask, "signal_quality"]),
                        color=color,
                        line=dict(color="#0b0f14", width=1),
                    ),
                    customdata=np.column_stack(
                        [
                            ref_day.loc[mask, "quality_name"].fillna("NONE").astype(str).values,
                            ref_day.loc[mask, "regime_name"].fillna("Ranging").astype(str).values,
                            ref_day.loc[mask, "liq_score"].fillna(0.0).astype(float).values,
                            ref_day.loc[mask, "obi"].fillna(0.0).astype(float).values,
                            ref_day.loc[mask, "cvd_delta"].fillna(0.0).astype(float).values,
                            ref_day.loc[mask, "forward_return_mean"].fillna(0.0).astype(float).values,
                        ]
                    ),
                    hovertemplate=(
                        f"Refinery {label}<br>%{{x}}<br>"
                        "Quality=%{customdata[0]}<br>"
                        "Regime=%{customdata[1]}<br>"
                        "Liq Score=%{customdata[2]:.2f}<br>"
                        "OBI=%{customdata[3]:.3f}<br>"
                        "CVD Δ=%{customdata[4]:.3f}<br>"
                        "Forward Return=%{customdata[5]:.5f}<extra></extra>"
                    ),
                ),
                row=2,
                col=1,
            )

        cvd_colors = ["#00e896" if v >= 0 else "#ff4d6d" for v in ref_day["cvd_delta"].fillna(0.0)]
        fig.add_trace(
            go.Bar(
                x=ref_day["ts_event"],
                y=ref_day["cvd_delta"].fillna(0.0),
                marker_color=cvd_colors,
                opacity=0.70,
                name="CVD Δ",
                hovertemplate="%{x}<br>CVD Δ=%{y:.3f}<extra></extra>",
            ),
            row=3,
            col=1,
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=ref_day["ts_event"],
                y=ref_day["liq_score"].fillna(0.0),
                mode="lines",
                line=dict(color="#f59e0b", width=1.8),
                name="Liq Score",
                hovertemplate="%{x}<br>Liq Score=%{y:.2f}<extra></extra>",
            ),
            row=3,
            col=1,
            secondary_y=True,
        )
        fig.add_hline(y=0.0, line_color="rgba(255,255,255,0.22)", line_width=1, row=3, col=1)

    summary_text = (
        f"<b>Refinery Summary</b><br>"
        f"Raw Bars: {len(raw_day):,}<br>"
        f"Event Bars: {int(ref_day.get('event_flag', pd.Series(dtype=int)).sum()) if not ref_day.empty else 0:,}<br>"
        f"LONG: {int((ref_day.get('bias_label', pd.Series(dtype=int)) == 0).sum()) if not ref_day.empty else 0:,} | "
        f"SHORT: {int((ref_day.get('bias_label', pd.Series(dtype=int)) == 1).sum()) if not ref_day.empty else 0:,} | "
        f"NEUTRAL: {int((ref_day.get('bias_label', pd.Series(dtype=int)) == 2).sum()) if not ref_day.empty else 0:,}"
    )
    fig.add_annotation(
        text=summary_text,
        xref="paper",
        yref="paper",
        x=1.0,
        y=1.03,
        showarrow=False,
        align="right",
        bgcolor="rgba(10,14,20,0.94)",
        bordercolor="#223248",
        borderwidth=1,
        font=dict(size=11, color="#dbe7f3", family="IBM Plex Mono"),
    )

    fig.update_layout(
        title=dict(text=f"QS V19 — Raw vs Refinery (5m) — {day_label}", x=0.02, font=dict(size=18, color="#ffffff", family="IBM Plex Mono")),
        paper_bgcolor="#060a0f",
        plot_bgcolor="#0b0f14",
        font=dict(color="#dbe7f3", family="IBM Plex Mono"),
        height=1180,
        margin=dict(l=20, r=92, t=58, b=22),
        hovermode="x",
        hoverdistance=20,
        spikedistance=1000,
        dragmode="pan",
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0.0,
            y=1.0,
            bgcolor="rgba(8,11,16,0.92)",
            bordercolor="#1e3048",
            borderwidth=1,
            font=dict(size=10),
        ),
        hoverlabel=dict(
            bgcolor="rgba(10,14,20,0.96)",
            bordercolor="#223248",
            font=dict(size=11, color="#dbe7f3", family="IBM Plex Mono"),
        ),
        xaxis=dict(rangeslider=dict(visible=False), type="date"),
    )

    for i in range(1, 4):
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
            row=i,
            col=1,
        )

    fig.update_yaxes(title_text="Price", tickformat=",.5f", row=1, col=1)
    fig.update_yaxes(
        title_text="Labels / OBI",
        range=[-2.1, 2.1],
        tickvals=[-1.55, 0.0, 1.55],
        ticktext=["SHORT", "NEUTRAL", "LONG"],
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="CVD Δ", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Liq Score", row=3, col=1, secondary_y=True)
    return fig


def _page_shell(title: str, nav_html: str, body_html: str) -> str:
    return (
        "<html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        "<style>"
        "body{background:#060a0f;margin:0;color:#dbe7f3;font-family:'IBM Plex Mono',monospace;}"
        ".shell{padding:10px 12px 18px 12px;}"
        ".nav{display:flex;gap:10px;align-items:center;overflow-x:auto;padding:14px 16px;"
        "background:#0b0f14;border-bottom:1px solid #1e3048;position:sticky;top:0;z-index:20;}"
        ".nav a,.nav span{color:#dbe7f3;text-decoration:none;padding:8px 12px;border:1px solid #223248;"
        "border-radius:999px;background:#0f1720;white-space:nowrap;font-size:12px;}"
        ".nav .active{background:#12263a;border-color:#3b82f6;}"
        ".muted{opacity:.78;}"
        ".summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;padding:14px 16px 0 16px;}"
        ".card{background:#0b0f14;border:1px solid #1e3048;border-radius:14px;padding:12px 14px;}"
        ".card b{display:block;margin-bottom:6px;font-size:13px;color:#ffffff;}"
        "</style></head><body>"
        f"{nav_html}<div class='shell'>{body_html}</div></body></html>"
    )


def _daily_nav(day: str, days: list[str], current_idx: int) -> str:
    prev_link = f"<a href='{days[current_idx - 1]}.html'>&larr; {days[current_idx - 1]}</a>" if current_idx > 0 else "<span class='muted'>&larr; First Day</span>"
    next_link = f"<a href='{days[current_idx + 1]}.html'>{days[current_idx + 1]} &rarr;</a>" if current_idx + 1 < len(days) else "<span class='muted'>Last Day &rarr;</span>"
    return (
        "<div class='nav'>"
        f"{prev_link}"
        "<a href='../compare_refinery_5m_index.html'>Index</a>"
        f"<span class='active'>{day}</span>"
        f"{next_link}"
        "</div>"
    )


def _write_index_page(output_dir: str, days: list[str], summaries: list[dict[str, Any]]) -> str:
    day_links = "".join(
        f"<a href='pages/{html.escape(item['day'])}.html'>{html.escape(item['day'])}</a>"
        for item in summaries
    )
    nav_html = f"<div class='nav'><span class='active'>Available Days</span>{day_links}</div>"

    cards = []
    for item in summaries:
        cards.append(
            "<div class='card'>"
            f"<b>{html.escape(item['day'])}</b>"
            f"Raw Bars: {int(item['raw_bars']):,}<br>"
            f"Event Bars: {int(item['event_bars']):,}<br>"
            f"LONG={int(item['long_bars']):,} | SHORT={int(item['short_bars']):,} | NEUTRAL={int(item['neutral_bars']):,}<br>"
            f"<a href='pages/{html.escape(item['day'])}.html'>Open Day</a>"
            "</div>"
        )
    body = "<div class='summary'>" + "".join(cards) + "</div>"
    html_out = _page_shell("QS V19 — Raw vs Refinery Index", nav_html, body)
    path = os.path.join(output_dir, "compare_refinery_5m_index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_out)
    return path


def generate_refinery_compare_5m_report(
    mbo_path: str,
    refinery_csv: str,
    output_dir: str,
    mbp_path: str = "",
    freq: str = "5min",
    raw_chunksize: int = 500_000,
    refinery_chunksize: int = 250_000,
    start_ts=None,
    end_ts=None,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    pages_dir = os.path.join(output_dir, "pages")
    os.makedirs(pages_dir, exist_ok=True)
    freq = _normalize_freq(freq)

    raw_bars = _aggregate_raw_bars(
        mbo_path=mbo_path,
        mbp_path=mbp_path,
        freq=freq,
        chunksize=raw_chunksize,
        start_ts=start_ts,
        end_ts=end_ts,
    )
    refinery_bars = _aggregate_refinery_bars(
        csv_path=refinery_csv,
        freq=freq,
        chunksize=refinery_chunksize,
        start_ts=start_ts,
        end_ts=end_ts,
    )

    raw_parquet = _write_table_with_fallback(raw_bars, os.path.join(output_dir, "compare_refinery_raw_5m.parquet"))
    refinery_parquet = _write_table_with_fallback(refinery_bars, os.path.join(output_dir, "compare_refinery_refinery_5m.parquet"))

    all_days = sorted(
        set(pd.to_datetime(raw_bars.get("ts_event", pd.Series(dtype="datetime64[ns]"))).dt.strftime("%Y-%m-%d").dropna().tolist())
        | set(pd.to_datetime(refinery_bars.get("ts_event", pd.Series(dtype="datetime64[ns]"))).dt.strftime("%Y-%m-%d").dropna().tolist())
    )

    summaries: list[dict[str, Any]] = []
    for idx, day in enumerate(all_days):
        raw_day = raw_bars[raw_bars["ts_event"].dt.strftime("%Y-%m-%d") == day].copy()
        ref_day = refinery_bars[refinery_bars["ts_event"].dt.strftime("%Y-%m-%d") == day].copy()
        fig = _build_daily_compare_figure(raw_day, ref_day, freq=freq, day_label=day)

        html_body = pio.to_html(
            fig,
            full_html=False,
            include_plotlyjs="cdn",
            config={
                "displaylogo": False,
                "responsive": True,
                "scrollZoom": True,
                "doubleClick": "reset+autosize",
            },
        )
        nav_html = _daily_nav(day, all_days, idx)
        html_out = _page_shell(f"QS V19 — Raw vs Refinery — {day}", nav_html, html_body)
        page_path = os.path.join(pages_dir, f"{day}.html")
        with open(page_path, "w", encoding="utf-8") as f:
            f.write(html_out)

        summaries.append(
            {
                "day": day,
                "page": page_path,
                "raw_bars": int(len(raw_day)),
                "event_bars": int(ref_day.get("event_flag", pd.Series(dtype=int)).sum()) if not ref_day.empty else 0,
                "long_bars": int((ref_day.get("bias_label", pd.Series(dtype=int)) == 0).sum()) if not ref_day.empty else 0,
                "short_bars": int((ref_day.get("bias_label", pd.Series(dtype=int)) == 1).sum()) if not ref_day.empty else 0,
                "neutral_bars": int((ref_day.get("bias_label", pd.Series(dtype=int)) == 2).sum()) if not ref_day.empty else 0,
            }
        )

    index_path = _write_index_page(output_dir, all_days, summaries)
    summary_path = os.path.join(output_dir, "compare_refinery_5m_summary.json")
    summary = {
        "days": all_days,
        "n_days": len(all_days),
        "raw_rows": int(len(raw_bars)),
        "refinery_rows": int(len(refinery_bars)),
        "files": {
            "index_html": index_path,
            "raw_parquet": raw_parquet,
            "refinery_parquet": refinery_parquet,
            "summary_json": summary_path,
            "pages_dir": pages_dir,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary
