"""
hourly_catboost_report.py - Hourly candles + CatBoost direction overlay
"""

from __future__ import annotations

import json
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from modules.oof_stacking import align_probability_columns
from prepare_training_data import CATBOOST_ADVISOR_FEATURES
from train_v19 import _apply_scaler_to_stat_frame, _raw_stat_frame, load_training_csv

try:
    from catboost import CatBoostClassifier

    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False


BIAS_LABELS = {0: "LONG", 1: "SHORT", 2: "NEUTRAL"}
BIAS_COLORS = {
    "LONG": "#1f9d55",
    "SHORT": "#c0392b",
    "NEUTRAL": "#6b7280",
}
BIAS_MARKERS = {
    "LONG": "^",
    "SHORT": "v",
    "NEUTRAL": "o",
}


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


def _predict_catboost_frame(csv_path: str, models_dir: str) -> pd.DataFrame:
    df = load_training_csv(csv_path).copy()
    if "price" not in df.columns:
        raise ValueError("Column 'price' is required to draw hourly candles")

    for col in ("ts_event", "price"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

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
        3,
        classes=classes or getattr(model, "classes_", None),
    )

    out = df.copy()
    out["cb_prob_long"] = probs[:, 0]
    out["cb_prob_short"] = probs[:, 1]
    out["cb_prob_neutral"] = probs[:, 2]
    pred_idx = np.argmax(probs, axis=1).astype(np.int8)
    out["cb_direction_idx"] = pred_idx
    out["cb_direction"] = pd.Series(pred_idx).map(BIAS_LABELS).values
    out["cb_confidence"] = probs.max(axis=1).astype(np.float32)
    return out


def _build_hourly_summary(pred_df: pd.DataFrame, freq: str = "1H") -> pd.DataFrame:
    frame = pred_df.copy()
    frame["ts_event"] = pd.to_datetime(frame["ts_event"], utc=True, errors="coerce").dt.tz_localize(None)
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame = frame.dropna(subset=["ts_event", "price"]).sort_values("ts_event")
    if frame.empty:
        return pd.DataFrame()

    frame = frame.set_index("ts_event")
    ohlc = frame["price"].resample(freq).ohlc()
    probs = frame[["cb_prob_long", "cb_prob_short", "cb_prob_neutral"]].resample(freq).mean()
    ticks = frame["price"].resample(freq).size().rename("tick_count")
    last_ts = frame.index.to_series().resample(freq).last().rename("last_tick_ts")
    last_price = frame["price"].resample(freq).last().rename("last_price")

    hourly = pd.concat([ohlc, probs, ticks, last_ts, last_price], axis=1)
    hourly = hourly.dropna(subset=["open", "high", "low", "close"]).reset_index()
    if hourly.empty:
        return hourly

    prob_cols = ["cb_prob_long", "cb_prob_short", "cb_prob_neutral"]
    prob_matrix = hourly[prob_cols].fillna(0.0).values
    direction_idx = np.argmax(prob_matrix, axis=1).astype(np.int8)
    confidence = prob_matrix.max(axis=1).astype(np.float32)
    strength = (hourly["cb_prob_long"] - hourly["cb_prob_short"]).astype(np.float32)

    hourly["direction_idx"] = direction_idx
    hourly["direction"] = pd.Series(direction_idx).map(BIAS_LABELS).values
    hourly["confidence"] = confidence
    hourly["direction_strength"] = strength
    hourly["change_flag"] = (hourly["direction"] != hourly["direction"].shift(1)).astype(np.int8)
    hourly["signal_time"] = pd.to_datetime(hourly["ts_event"]) + pd.Timedelta(hours=1)
    return hourly


def _extract_transitions(hourly: pd.DataFrame) -> pd.DataFrame:
    if hourly.empty:
        return pd.DataFrame()

    transitions = hourly.loc[hourly["change_flag"] == 1, [
        "ts_event",
        "signal_time",
        "direction",
        "confidence",
        "close",
        "cb_prob_long",
        "cb_prob_short",
        "cb_prob_neutral",
    ]].copy()
    transitions = transitions.rename(columns={
        "ts_event": "hour_start",
        "close": "hour_close",
    })
    transitions["prev_direction"] = hourly["direction"].shift(1).loc[transitions.index].fillna("START").values
    return transitions.reset_index(drop=True)


def _plot_hourly_candles(hourly: pd.DataFrame, transitions: pd.DataFrame, output_png: str, title: str) -> None:
    if hourly.empty:
        raise ValueError("Hourly frame is empty")

    fig = plt.figure(figsize=(18, 11), dpi=140)
    gs = fig.add_gridspec(3, 1, height_ratios=[3.8, 1.5, 0.9], hspace=0.08)
    ax_price = fig.add_subplot(gs[0, 0])
    ax_probs = fig.add_subplot(gs[1, 0], sharex=ax_price)
    ax_bias = fig.add_subplot(gs[2, 0], sharex=ax_price)

    x = mdates.date2num(hourly["ts_event"].dt.to_pydatetime())
    width = 0.028

    for xi, row in zip(x, hourly.itertuples(index=False)):
        candle_color = BIAS_COLORS["LONG"] if row.close >= row.open else BIAS_COLORS["SHORT"]
        ax_price.vlines(xi, row.low, row.high, color=candle_color, linewidth=1.2, alpha=0.95)
        body_low = min(row.open, row.close)
        body_height = max(abs(row.close - row.open), 1e-8)
        ax_price.add_patch(
            Rectangle(
                (xi - width / 2.0, body_low),
                width,
                body_height,
                facecolor=candle_color,
                edgecolor=candle_color,
                linewidth=1.0,
                alpha=0.68,
            )
        )

    for label in ("LONG", "SHORT", "NEUTRAL"):
        mask = hourly["direction"] == label
        if not bool(mask.any()):
            continue
        ax_price.scatter(
            hourly.loc[mask, "ts_event"],
            hourly.loc[mask, "close"],
            s=40 + hourly.loc[mask, "confidence"].fillna(0.0).values * 110.0,
            marker=BIAS_MARKERS[label],
            color=BIAS_COLORS[label],
            alpha=0.95,
            label=f"CatBoost {label}",
            zorder=5,
        )

    if not transitions.empty:
        for row in transitions.itertuples(index=False):
            ax_price.axvline(row.signal_time, color="#94a3b8", linestyle="--", linewidth=0.8, alpha=0.35)

    ax_price.set_title(title, fontsize=15, weight="bold")
    ax_price.set_ylabel("Price")
    ax_price.grid(True, alpha=0.18, linestyle=":")
    ax_price.legend(loc="upper left", ncol=3, frameon=False)

    ax_probs.plot(hourly["ts_event"], hourly["cb_prob_long"], color=BIAS_COLORS["LONG"], linewidth=1.8, label="P(LONG)")
    ax_probs.plot(hourly["ts_event"], hourly["cb_prob_short"], color=BIAS_COLORS["SHORT"], linewidth=1.8, label="P(SHORT)")
    ax_probs.plot(hourly["ts_event"], hourly["cb_prob_neutral"], color=BIAS_COLORS["NEUTRAL"], linewidth=1.6, label="P(NEUTRAL)")
    ax_probs.set_ylim(-0.02, 1.02)
    ax_probs.set_ylabel("Prob")
    ax_probs.grid(True, alpha=0.18, linestyle=":")
    ax_probs.legend(loc="upper left", ncol=3, frameon=False)

    bias_value = hourly["direction"].map({"LONG": 1.0, "NEUTRAL": 0.0, "SHORT": -1.0}).astype(np.float32)
    ax_bias.axhline(0.0, color="#cbd5e1", linewidth=1.0)
    ax_bias.step(hourly["ts_event"], bias_value, where="mid", color="#0f172a", linewidth=1.4)
    for label, value in (("LONG", 1.0), ("NEUTRAL", 0.0), ("SHORT", -1.0)):
        mask = hourly["direction"] == label
        ax_bias.scatter(
            hourly.loc[mask, "ts_event"],
            bias_value.loc[mask],
            s=28 + hourly.loc[mask, "confidence"].fillna(0.0).values * 80.0,
            marker="s",
            color=BIAS_COLORS[label],
            alpha=0.92,
        )
    ax_bias.set_yticks([-1, 0, 1], labels=["SHORT", "NEUTRAL", "LONG"])
    ax_bias.set_ylabel("Bias")
    ax_bias.grid(True, axis="y", alpha=0.18, linestyle=":")

    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    ax_bias.xaxis.set_major_locator(locator)
    ax_bias.xaxis.set_major_formatter(formatter)
    plt.setp(ax_price.get_xticklabels(), visible=False)
    plt.setp(ax_probs.get_xticklabels(), visible=False)

    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def generate_hourly_catboost_report(
    csv_path: str,
    models_dir: str,
    output_dir: str | None = None,
    freq: str = "1H",
    report_name: str = "hourly_catboost",
) -> dict[str, Any]:
    output_dir = output_dir or models_dir
    os.makedirs(output_dir, exist_ok=True)

    pred_df = _predict_catboost_frame(csv_path, models_dir)
    hourly = _build_hourly_summary(pred_df, freq=freq)
    transitions = _extract_transitions(hourly)

    hourly_csv = os.path.join(output_dir, f"{report_name}_signals.csv")
    transitions_csv = os.path.join(output_dir, f"{report_name}_turns.csv")
    chart_png = os.path.join(output_dir, f"{report_name}_chart.png")
    summary_json = os.path.join(output_dir, f"{report_name}_summary.json")

    hourly.to_csv(hourly_csv, index=False)
    transitions.to_csv(transitions_csv, index=False)

    span = ""
    if not hourly.empty:
        start_ts = pd.to_datetime(hourly["ts_event"].iloc[0]).strftime("%Y-%m-%d %H:%M")
        end_ts = pd.to_datetime(hourly["ts_event"].iloc[-1]).strftime("%Y-%m-%d %H:%M")
        span = f"{start_ts} -> {end_ts}"
    title = f"QuantSystem V19 | Hourly Candles + CatBoost Direction | {span}".strip(" |")
    _plot_hourly_candles(hourly, transitions, chart_png, title=title)

    summary = {
        "rows": int(len(pred_df)),
        "hours": int(len(hourly)),
        "transitions": int(len(transitions)),
        "direction_counts": hourly["direction"].value_counts().to_dict() if not hourly.empty else {},
        "files": {
            "chart_png": chart_png,
            "hourly_signals_csv": hourly_csv,
            "turns_csv": transitions_csv,
        },
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    summary["files"]["summary_json"] = summary_json
    return summary
