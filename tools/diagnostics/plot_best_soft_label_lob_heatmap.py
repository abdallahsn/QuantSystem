#!/usr/bin/env python3
"""
لوحة حرارية لصفّ آخر وقتي من lob_tensors مع لحظة مواءمة أسعارية لصفّ فيه أعلى soft_label.

يشترط نفس تشغيل refinery (lob_tensors.npy + lob_tensor_timestamps.npy + final/features_*.parquet).

التشغيل (من جذر المشروع) — مع --out يُنشأ أيضاً PNG لمسار السعر باسم *_price_context.png
(التحكم: --price-window-minutes، --price-out، --no-price-path).
  py -3.13 plot_best_soft_label_lob_heatmap.py --pipeline pipeline_mbo7_refinery_latest --out lob.png
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _load_lob(root: str) -> tuple[np.ndarray, pd.Series]:
    lob_path = os.path.join(root, "lob_tensors.npy")
    ts_path = os.path.join(root, "lob_tensor_timestamps.npy")
    if not os.path.isfile(lob_path):
        raise FileNotFoundError(f"missing {lob_path}")
    if not os.path.isfile(ts_path):
        raise FileNotFoundError(f"missing {ts_path}")
    lob = np.load(lob_path, mmap_mode="r")
    ts_raw = np.load(ts_path)
    idx = pd.to_datetime(ts_raw.astype(np.int64), unit="ns", utc=True, errors="coerce")
    if isinstance(idx, pd.DatetimeIndex):
        naive = idx.tz_convert(None) if idx.tz is not None else idx
        ts = pd.Series(naive.to_numpy(dtype="datetime64[ns]"), copy=False)
    else:
        s = idx if isinstance(idx, pd.Series) else pd.Series(idx)
        ts = s.dt.tz_convert(None) if s.dt.tz is not None else s
        ts = ts.reset_index(drop=True)
    if len(ts) != len(lob):
        n = min(len(ts), len(lob))
        print(f"⚠ trimming LOB tensors/timestamps mismatch to {n:,}", flush=True)
        lob = lob[:n]
        ts = ts.iloc[:n]
    return lob, ts


def _parquet_feature_columns(sample_parquet_path: str) -> set[str]:
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415

        return set(pq.read_schema(sample_parquet_path).names)
    except Exception:
        return set(pd.read_parquet(sample_parquet_path).columns.astype(str))


def _load_price_window(final_dir: str, t_center: pd.Timestamp, window_minutes: float) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(final_dir, "features_*.parquet")))
    if not paths:
        return pd.DataFrame()

    hw = pd.Timedelta(minutes=float(window_minutes))
    t0 = t_center - hw
    t1 = t_center + hw
    chunks: list[pd.DataFrame] = []
    for shard in paths:
        shard_cols = _parquet_feature_columns(shard)
        want = ["ts_event"]
        for c in ("price", "bid_px_00", "ask_px_00"):
            if c in shard_cols:
                want.append(c)
        df = pd.read_parquet(shard, columns=[c for c in want])
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce").dt.tz_localize(None)
        df = df.dropna(subset=["ts_event"])
        chunks.append(df.loc[(df["ts_event"] >= t0) & (df["ts_event"] <= t1)])
    if not chunks:
        return pd.DataFrame()
    price_df = pd.concat(chunks, ignore_index=True)
    price_df = price_df.sort_values("ts_event").drop_duplicates(subset=["ts_event"], keep="last")
    if "price" in price_df.columns:
        price_df["chart_price"] = pd.to_numeric(price_df["price"], errors="coerce")
    elif "bid_px_00" in price_df.columns and "ask_px_00" in price_df.columns:
        b = pd.to_numeric(price_df["bid_px_00"], errors="coerce")
        a = pd.to_numeric(price_df["ask_px_00"], errors="coerce")
        price_df["chart_price"] = (b + a) / 2.0
    else:
        price_df["chart_price"] = np.nan
    return price_df


def _plot_price_context(
    price_df: pd.DataFrame,
    *,
    best_ts: pd.Timestamp,
    direction: str,
    soft: float,
    window_minutes: float,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 4.5))
    vv = price_df.dropna(subset=["chart_price"])
    if vv.empty:
        ax.text(0.5, 0.5, "no price / bid_px_ask in window columns", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.plot(vv["ts_event"], vv["chart_price"], color="steelblue", linewidth=0.9)
        ax.scatter(vv["ts_event"], vv["chart_price"], s=10, alpha=0.45, color="steelblue")
    ax.axvline(best_ts, color="cyan", linestyle="--", linewidth=2.0, label="best soft_label ts")
    ax.set_xlabel("time")
    ax.set_ylabel("price (feature row; tick-by-tick from final parquet)")
    ax.set_title(
        f"context ±{window_minutes:g} min | {direction} | soft_label={soft:.4f} | ts_event={best_ts}\n"
        "(not OHLC candles — scatter/line over event rows)"
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.28)
    fig.autofmt_xdate()
    plt.tight_layout()
    outp = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    fig.savefig(outp, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _find_best_soft_label_shard(final_dir: str) -> tuple[pd.Timestamp, float, pd.Series]:
    paths = sorted(glob.glob(os.path.join(final_dir, "features_*.parquet")))
    if not paths:
        raise FileNotFoundError(f"no features_*.parquet under {final_dir}")
    best_shard: str | None = None
    best_soft = float("-inf")
    best_loc = None  # positional index للصف الفائز في شارد واحد بعد فلتر NaN إن احتجنا

    for shard in paths:
        df = pd.read_parquet(shard, columns=["ts_event", "soft_label"]).dropna(subset=["soft_label"])
        if df.empty:
            continue
        j = df["soft_label"].idxmax()
        s = float(df.loc[j, "soft_label"])
        if s > best_soft:
            best_soft = s
            best_shard = shard
            best_loc = j
    if not best_shard or best_loc is None:
        raise RuntimeError("could not find a row with valid soft_label in final shards")

    row = pd.read_parquet(best_shard).loc[[best_loc]].iloc[0]
    best_ts = pd.to_datetime(row["ts_event"], utc=True, errors="coerce").tz_localize(None)
    if pd.isna(best_ts):
        raise RuntimeError("best row has invalid ts_event")
    return best_ts, best_soft, row


def _tensor_index_merge_asof(best_ts: pd.Timestamp, lob_ts: pd.Series) -> int:
    lob_df = (
        pd.DataFrame({"tensor_idx": np.arange(len(lob_ts), dtype=np.int32), "ts_event": lob_ts})
        .dropna(subset=["ts_event"])
        .sort_values("ts_event")
        .reset_index(drop=True)
    )
    row = pd.DataFrame({"ts_event": [best_ts]})
    merged = pd.merge_asof(row.sort_values("ts_event"), lob_df, on="ts_event", direction="backward")
    idx = merged["tensor_idx"].iloc[0]
    if pd.isna(idx):
        # أقرب زمني
        tns = lob_df["ts_event"].astype("int64").to_numpy(dtype=np.int64)
        bn = pd.Timestamp(best_ts).value if hasattr(best_ts, "value") else int(pd.Timestamp(best_ts).asm8)
        j = np.searchsorted(tns, bn, side="left")
        j = np.clip(j, 0, len(lob_df) - 1)
        idx = int(lob_df["tensor_idx"].iloc[j])
    return int(idx)


def main() -> None:
    p = argparse.ArgumentParser(description="Heatmap لواجهة LOB عند أعلى soft_label")
    p.add_argument(
        "--pipeline",
        default="pipeline_mbo7_refinery_latest",
        help="جذر refinery الذي فيه lob_tensors نهائي ومجلّد final",
    )
    p.add_argument("--out", default="", help="حفظ PNG (يفتح نافذة إذا فارغ)")
    p.add_argument(
        "--price-window-minutes",
        type=float,
        default=120.0,
        help="عرض مسار السعر: ± دقائق حول ts_event لأفضل soft_label",
    )
    p.add_argument(
        "--price-out",
        default="",
        help="PNG لمسار السعر؛ فارغ + --out ⇒ يُحفظ ملف _price_context بجانب الهيتماب",
    )
    p.add_argument("--no-price-path", action="store_true", help="عدم رسم/حفظ مسار السعر")
    args = p.parse_args()

    root = os.path.abspath(args.pipeline.rstrip("\\/ "))
    final_dir = os.path.join(root, "final")
    lob, lob_ts = _load_lob(root)
    best_ts, best_soft, best_row = _find_best_soft_label_shard(final_dir)
    lob_idx = _tensor_index_merge_asof(best_ts, lob_ts)

    direction = "LONG" if best_soft > 0.5 else "SHORT"
    price_cols = ["price", "mid_price"]
    ref_price = next((best_row[c] for c in price_cols if c in best_row.index and pd.notna(best_row[c])), np.nan)

    print(
        f"أفضل soft_label للمراقبة glob: {direction} | soft_label={best_soft:.6f} | ts_event={best_ts} | lob_idx={lob_idx:,}",
        flush=True,
    )
    if pd.notna(ref_price):
        print(f"reference price (صفّ الميزات): {ref_price}", flush=True)

    # شكل V19 DeepLOB: (N, time_steps=50, price_levels=20, channels=3)
    snap = np.asarray(lob[lob_idx], dtype=np.float32)
    if snap.ndim != 3 or snap.shape[-1] < 3:
        print(f"unexpected LOB slice shape={snap.shape}, expected (T,P,≥3)", file=sys.stderr)
        sys.exit(1)

    t_steps, levels, _ = snap.shape
    channel_names = ["Depth (MBP map)", "Buy footprint", "Sell footprint"]

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    for c, ax in enumerate(axes):
        mat = snap[:, :, c].T
        sns.heatmap(
            mat,
            cmap="YlOrRd",
            xticklabels=False,
            yticklabels=False,
            cbar_kws={"label": channel_names[c]},
            ax=ax,
        )
        ax.axvline(x=t_steps - 0.5, color="cyan", linestyle="--", linewidth=1.5, label="window end (decision tick)")
        ax.set_ylabel("price level (0..19)")
        ax.set_title(channel_names[c])
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("time step (oldest → newest within 50-tick window)")
    fig.suptitle(
        f"LOB evolution @ merge_asof ts | {direction} | soft_label={best_soft:.4f} | ts={best_ts}",
        fontsize=12,
    )
    plt.tight_layout()

    if args.out:
        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"saved: {out_path}", flush=True)
        plt.close(fig)
    else:
        plt.show()

    price_out_final = ""
    if not args.no_price_path:
        if args.price_out.strip():
            price_out_final = os.path.abspath(args.price_out.strip())
        elif args.out.strip():
            b, ext = os.path.splitext(os.path.abspath(args.out))
            price_out_final = f"{b}_price_context{ext or '.png'}"

    if price_out_final:
        price_df = _load_price_window(final_dir, best_ts, args.price_window_minutes)
        print(f"price context rows in ±{args.price_window_minutes:g} min: {len(price_df):,}", flush=True)
        _plot_price_context(
            price_df,
            best_ts=best_ts,
            direction=direction,
            soft=float(best_soft),
            window_minutes=float(args.price_window_minutes),
            out_path=price_out_final,
        )
        print(f"saved: {price_out_final}", flush=True)


if __name__ == "__main__":
    main()
