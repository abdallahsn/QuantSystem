"""
pipeline_feature_diagnostic.py
==============================
QuantSystem-aware diagnostics for stage1 refinery shards.

Key differences vs generic checks:
- CVD is validated on first differences (delta flow), not absolute level.
- kyle_lambda is treated as a normalized signal (can be negative).
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class CheckResult:
    passed: bool
    message: str


@dataclass
class Rule:
    name: str
    description: str
    severity: str  # CRITICAL | WARNING
    checks: list[Callable[[pd.DataFrame], CheckResult]]


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col, pd.Series(dtype=np.float64)), errors="coerce")


def _pct(mask: pd.Series | np.ndarray) -> float:
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0:
        return 0.0
    return float(arr.mean())


def _load_data(path: str, max_files: int) -> pd.DataFrame:
    if os.path.isfile(path):
        return pd.read_parquet(path)

    files = sorted(glob.glob(os.path.join(path, "**/*.parquet"), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(path, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {path}")

    use_files = files[: max(int(max_files), 1)]
    print(f"  ← total files found: {len(files)} | loading first {len(use_files)}")
    frames = [pd.read_parquet(f) for f in use_files]
    df = pd.concat(frames, ignore_index=True)
    print(f"  ← loaded rows={len(df):,} | cols={len(df.columns)}")
    return df


def _build_rules() -> list[Rule]:
    return [
        Rule(
            name="cvd",
            description="CVD flow balance on diff(cvd), not absolute level.",
            severity="CRITICAL",
            checks=[
                lambda df: CheckResult(
                    passed=("cvd" in df.columns),
                    message="column present" if "cvd" in df.columns else "column missing",
                ),
                lambda df: (
                    CheckResult(True, "skipped (missing cvd)")
                    if "cvd" not in df.columns
                    else _check_cvd_diff(df)
                ),
            ],
        ),
        Rule(
            name="kyle_lambda",
            description="Kyle signal must be finite and non-collapsed (negative values allowed).",
            severity="CRITICAL",
            checks=[
                lambda df: CheckResult(
                    passed=("kyle_lambda" in df.columns),
                    message="column present" if "kyle_lambda" in df.columns else "column missing",
                ),
                lambda df: (
                    CheckResult(True, "skipped (missing kyle_lambda)")
                    if "kyle_lambda" not in df.columns
                    else _check_kyle_signal(df)
                ),
            ],
        ),
        Rule(
            name="hawkes_intensity",
            description="Hawkes should be non-negative with useful dispersion.",
            severity="CRITICAL",
            checks=[
                lambda df: CheckResult(
                    passed=("hawkes_intensity" in df.columns),
                    message="column present" if "hawkes_intensity" in df.columns else "column missing",
                ),
                lambda df: (
                    CheckResult(True, "skipped (missing hawkes_intensity)")
                    if "hawkes_intensity" not in df.columns
                    else _check_hawkes(df)
                ),
            ],
        ),
        Rule(
            name="absorption_intensity",
            description="Absorption spikes should survive aggregation.",
            severity="WARNING",
            checks=[
                lambda df: CheckResult(
                    passed=("absorption_intensity" in df.columns),
                    message="column present" if "absorption_intensity" in df.columns else "column missing -> skipped",
                ),
                lambda df: (
                    CheckResult(True, "skipped (missing absorption_intensity)")
                    if "absorption_intensity" not in df.columns
                    else _check_absorption(df)
                ),
            ],
        ),
        Rule(
            name="cancel_ratio",
            description="Cancel ratio should stay in [0,1] and not collapse.",
            severity="WARNING",
            checks=[
                lambda df: (
                    CheckResult(True, "skipped (missing cancel_ratio)")
                    if "cancel_ratio" not in df.columns
                    else _check_cancel_ratio(df)
                ),
            ],
        ),
        Rule(
            name="micro_atr",
            description="Micro ATR should be non-negative with non-trivial variance.",
            severity="WARNING",
            checks=[
                lambda df: (
                    CheckResult(True, "skipped (missing micro_atr)")
                    if "micro_atr" not in df.columns
                    else _check_micro_atr(df)
                ),
            ],
        ),
        Rule(
            name="size",
            description="Trade size should be positive.",
            severity="CRITICAL",
            checks=[_check_size],
        ),
        Rule(
            name="price",
            description="Price should be positive with low extreme gaps.",
            severity="CRITICAL",
            checks=[_check_price],
        ),
        Rule(
            name="ts_event",
            description="Timestamps should parse and be monotonic after sort.",
            severity="CRITICAL",
            checks=[_check_timestamps],
        ),
        Rule(
            name="trade_side_mix",
            description="Trade side mix sanity for action in trade set.",
            severity="WARNING",
            checks=[_check_trade_side_mix],
        ),
    ]


def _check_cvd_diff(df: pd.DataFrame) -> CheckResult:
    s = _num(df, "cvd").dropna()
    if s.empty:
        return CheckResult(False, "cvd empty after numeric coercion")
    d = s.diff().fillna(0.0)
    pos = _pct(d > 0)
    neg = _pct(d < 0)
    zer = _pct(d == 0)
    passed = pos > 0.30 and neg > 0.30
    return CheckResult(
        passed,
        f"diff(cvd): pos={pos:.1%} | neg={neg:.1%} | zero={zer:.1%} (target pos/neg >30%)",
    )


def _check_kyle_signal(df: pd.DataFrame) -> CheckResult:
    s = _num(df, "kyle_lambda")
    finite = np.isfinite(s).mean() if len(s) else 0.0
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return CheckResult(False, "all values are NaN/inf")
    spread = float(s.quantile(0.95) - s.quantile(0.05))
    std = float(s.std(ddof=0))
    passed = finite > 0.995 and std > 1e-4 and spread > 0.10
    return CheckResult(
        passed,
        f"finite={finite:.2%} | std={std:.4f} | p95-p05={spread:.4f} (negative values allowed)",
    )


def _check_hawkes(df: pd.DataFrame) -> CheckResult:
    s = _num(df, "hawkes_intensity").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return CheckResult(False, "empty after coercion")
    neg = _pct(s < 0)
    z = _pct(s == 0)
    cv = float(s.std(ddof=0) / max(s.mean(), 1e-9))
    p95_ratio = float(s.quantile(0.95) / max(s.mean(), 1e-9))
    passed = neg < 0.01 and z < 0.30 and cv > 0.20 and p95_ratio > 1.5
    return CheckResult(
        passed,
        f"neg={neg:.1%} | zero={z:.1%} | CV={cv:.2f} | p95/mean={p95_ratio:.2f}x",
    )


def _check_absorption(df: pd.DataFrame) -> CheckResult:
    s = _num(df, "absorption_intensity").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return CheckResult(False, "empty after coercion")
    neg = _pct(s < 0)
    top = float(s.max())
    top_share = float(np.isclose(s.to_numpy(dtype=np.float64), top, rtol=0.0, atol=1e-9).mean())
    ratio = float(s.quantile(0.95) / max(s.mean(), 1e-9))
    spread = float(s.quantile(0.95) - s.quantile(0.05))
    cv = float(s.std(ddof=0) / max(s.mean(), 1e-9))
    saturated_legacy_cap = top >= 9.999 and top_share >= 0.10
    passed = neg < 0.01 and not saturated_legacy_cap and (ratio > 3.0 or spread > 1.0 or cv > 0.35)
    return CheckResult(
        passed,
        f"neg={neg:.1%} | p95/mean={ratio:.2f}x | p95-p05={spread:.3f} | "
        f"CV={cv:.2f} | top_share={top_share:.1%}",
    )


def _check_cancel_ratio(df: pd.DataFrame) -> CheckResult:
    s = _num(df, "cancel_ratio").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return CheckResult(False, "empty after coercion")
    in_range = float(s.between(0, 1.001).mean())
    z = _pct(s == 0)
    passed = in_range > 0.99 and z < 0.95
    return CheckResult(passed, f"in_range={in_range:.1%} | zero={z:.1%}")


def _check_micro_atr(df: pd.DataFrame) -> CheckResult:
    s = _num(df, "micro_atr").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return CheckResult(False, "empty after coercion")
    neg = _pct(s < 0)
    cv = float(s.std(ddof=0) / max(s.mean(), 1e-9))
    passed = neg < 0.01 and cv > 0.02
    return CheckResult(passed, f"neg={neg:.1%} | CV={cv:.3f}")


def _check_size(df: pd.DataFrame) -> CheckResult:
    col = (
        "size"
        if "size" in df.columns
        else ("trade_size" if "trade_size" in df.columns else ("volume" if "volume" in df.columns else None))
    )
    if col is None:
        return CheckResult(False, "size/trade_size/volume missing")
    s = _num(df, col).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return CheckResult(False, "empty after coercion")
    neg = _pct(s < 0)
    zer = _pct(s == 0)
    passed = neg < 0.001 and zer < 0.001
    return CheckResult(passed, f"neg={neg:.2%} | zero={zer:.2%}")


def _check_price(df: pd.DataFrame) -> CheckResult:
    if "price" not in df.columns:
        return CheckResult(False, "price missing")
    s = _num(df, "price").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return CheckResult(False, "empty after coercion")
    neg = _pct(s < 0)
    gaps = float((s.pct_change().abs().dropna() > 0.05).mean())
    passed = neg < 0.001 and gaps < 0.001
    return CheckResult(passed, f"neg={neg:.2%} | gaps>5%={gaps:.3%}")


def _check_timestamps(df: pd.DataFrame) -> CheckResult:
    if "ts_event" not in df.columns:
        return CheckResult(False, "ts_event missing")
    ts = pd.to_datetime(df["ts_event"], errors="coerce")
    bad = int(ts.isna().sum())
    s = ts.dropna().sort_values()
    back = int((s.diff().dropna() < pd.Timedelta(0)).sum())
    passed = bad == 0 and back == 0
    return CheckResult(passed, f"NaT={bad:,} | backwards={back:,} | range={s.min()} -> {s.max()}")


def _check_trade_side_mix(df: pd.DataFrame) -> CheckResult:
    if "side" not in df.columns or "action" not in df.columns:
        return CheckResult(True, "skipped (side/action missing)")
    action = df["action"].astype(str).str.upper()
    side = df["side"].astype(str).str.upper()
    trade_mask = action.isin({"T", "F", "TRADE", "EXECUTE", "E", "0"})
    if not trade_mask.any():
        return CheckResult(True, "skipped (no trade rows)")
    s = side[trade_mask]
    pa = _pct(s == "A")
    pb = _pct(s == "B")
    passed = pa > 0.20 and pb > 0.20
    return CheckResult(passed, f"trade side mix: A={pa:.1%} | B={pb:.1%}")


def _run(df: pd.DataFrame) -> tuple[dict[str, list[tuple[Rule, list[CheckResult]]]], str]:
    buckets: dict[str, list[tuple[Rule, list[CheckResult]]]] = {
        "CRITICAL_FAIL": [],
        "WARNING_FAIL": [],
        "PASS": [],
        "SKIP": [],
    }
    rules = _build_rules()

    for rule in rules:
        results = [check(df) for check in rule.checks]
        failed = any(not r.passed for r in results if "skipped" not in r.message.lower())
        skipped_only = all("skipped" in r.message.lower() for r in results)
        if skipped_only:
            buckets["SKIP"].append((rule, results))
        elif failed:
            buckets[f"{rule.severity}_FAIL"].append((rule, results))
        else:
            buckets["PASS"].append((rule, results))

    lines: list[str] = []
    w = 98
    lines.append("\n" + "═" * w)
    lines.append("  PIPELINE FEATURE DIAGNOSTIC REPORT (QuantSystem-aware)")
    lines.append(f"  rows={len(df):,} | cols={len(df.columns)}")
    lines.append("═" * w)

    for bucket, icon, title in (
        ("CRITICAL_FAIL", "🔴", "CRITICAL"),
        ("WARNING_FAIL", "🟠", "WARNING"),
        ("PASS", "✅", "PASS"),
        ("SKIP", "⏭️", "SKIP"),
    ):
        items = buckets[bucket]
        lines.append(f"\n{icon} {title} ({len(items)}):")
        if not items:
            lines.append("   none")
            continue
        for rule, checks in items:
            lines.append(f"  - {rule.name}: {rule.description}")
            for r in checks:
                lines.append(f"      {'✓' if r.passed else '✗'} {r.message}")

    lines.append("\n" + "─" * w)
    lines.append("📊 quick stats")
    for c in (
        "cvd",
        "kyle_lambda",
        "hawkes_intensity",
        "absorption_intensity",
        "cancel_ratio",
        "micro_atr",
        "size",
        "price",
    ):
        if c not in df.columns:
            continue
        s = _num(df, c).replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue
        lines.append(
            f"  {c:20s} mean={s.mean():+10.4f} std={s.std(ddof=0):9.4f} "
            f"p25={s.quantile(0.25):+9.4f} p75={s.quantile(0.75):+9.4f} "
            f"neg={_pct(s < 0):5.1%} zer={_pct(s == 0):5.1%}"
        )

    lines.append("\n" + "═" * w)
    lines.append(
        f"SUMMARY: CRITICAL={len(buckets['CRITICAL_FAIL'])} | "
        f"WARNING={len(buckets['WARNING_FAIL'])} | "
        f"PASS={len(buckets['PASS'])} | SKIP={len(buckets['SKIP'])}"
    )
    lines.append("═" * w + "\n")
    report = "\n".join(lines)
    return buckets, report


def main() -> None:
    parser = argparse.ArgumentParser(description="QuantSystem-aware refinery feature diagnostics")
    parser.add_argument("--path", required=True, help="Path to parquet file or mbo_final directory")
    parser.add_argument("--save", default=None, help="Optional path to save report")
    parser.add_argument("--max_files", type=int, default=10, help="Max parquet shards to load from directory")
    args = parser.parse_args()

    print(f"\n📥 Loading parquet from: {args.path}")
    df = _load_data(args.path, args.max_files)
    print("\n🔍 Running diagnostics...")
    _, report = _run(df)
    print(report)
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"💾 Saved report: {args.save}")


if __name__ == "__main__":
    main()
