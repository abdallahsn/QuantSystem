#!/usr/bin/env python3
"""Leakage-aware feature intelligence report for QuantSystem V19 artifacts."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.feature_selection import mutual_info_classif
except Exception:  # pragma: no cover - optional runtime dependency
    mutual_info_classif = None


LABEL_ARTIFACT_COLS = {
    "bias_label",
    "conf_label",
    "signal_quality",
    "forward_return",
    "label_horizon_steps",
    "effective_horizon",
    "label_dynamic_threshold",
    "path_outcome",
    "adverse_path_flag",
    "bias_label_detail",
    "neutral_reason",
    "timeout_move_exceeded_band",
    "soft_label",
    "label_confidence",
    "soft_label_long",
    "soft_label_short",
    "soft_sample_weight",
    "soft_label_confidence",
    "soft_label_entropy",
    "soft_label_scenarios",
    "mc_sample_weight",
    "label_stability",
    "label_end_ts",
    "ts_event",
}


def _is_label_artifact(col: str) -> bool:
    raw = col[5:] if col.startswith("raw__") else col
    return raw in LABEL_ARTIFACT_COLS or raw.startswith("soft_label")


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _mi_score(x: pd.Series, y: np.ndarray) -> float:
    if mutual_info_classif is None:
        return float("nan")


def _spearman_no_scipy(x: pd.Series, y: np.ndarray) -> float:
    xr = pd.to_numeric(x, errors="coerce").fillna(0.0).rank(method="average")
    yr = pd.Series(y, index=x.index).rank(method="average")
    corr = xr.corr(yr, method="pearson")
    return float(corr) if pd.notna(corr) else 0.0
    arr = x.to_numpy(dtype=np.float64).reshape(-1, 1)
    if len(np.unique(arr[np.isfinite(arr)])) <= 1:
        return 0.0
    try:
        return float(mutual_info_classif(arr, y, discrete_features=False, random_state=42)[0])
    except Exception:
        return float("nan")


def build_report(df: pd.DataFrame, label_col: str) -> dict[str, Any]:
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")
    y = pd.to_numeric(df[label_col], errors="coerce").fillna(2).astype(np.int32).to_numpy()
    rows: list[dict[str, Any]] = []
    artifacts: list[str] = []

    for col in df.columns:
        if col == label_col:
            artifacts.append(col)
            continue
        if _is_label_artifact(col):
            artifacts.append(col)
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        x = _numeric_series(df, col)
        std = float(x.std())
        zero_pct = float(np.mean(np.abs(x.to_numpy(dtype=np.float64)) < 1e-12)) if len(x) else 1.0
        if std <= 1e-12:
            verdict = "DEAD"
            mi = 0.0
            rho = 0.0
        else:
            mi = _mi_score(x, y)
            rho = _spearman_no_scipy(x, y)
            verdict = "STRONG" if np.isfinite(mi) and mi >= 0.10 else "WEAK" if (np.isfinite(mi) and mi >= 0.01) or abs(rho) >= 0.06 else "NOISE"
        rows.append(
            {
                "feature": col,
                "verdict": verdict,
                "mi": None if not np.isfinite(mi) else float(mi),
                "rho": float(rho),
                "std": std,
                "zero_pct": zero_pct,
            }
        )

    ranked = sorted(
        rows,
        key=lambda r: (-1.0 if r["mi"] is None else -float(r["mi"]), -abs(float(r["rho"]))),
    )
    return {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "label_col": label_col,
        "label_distribution": {str(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().to_dict().items()},
        "label_artifacts_excluded": sorted(artifacts),
        "features": rows,
        "top_features": ranked[:25],
    }


def write_markdown(report: dict[str, Any], path: str) -> None:
    lines = [
        "# Feature Intelligence Report",
        "",
        f"Rows: {report['rows']:,} | Columns: {report['cols']:,}",
        f"Label: `{report['label_col']}` | Distribution: `{report['label_distribution']}`",
        "",
        "## Label Artifacts Excluded",
        "",
    ]
    for col in report["label_artifacts_excluded"]:
        lines.append(f"- `{col}`: LABEL_ARTIFACT")
    lines.extend(["", "## Top Features", ""])
    for row in report["top_features"]:
        mi = "n/a" if row["mi"] is None else f"{float(row['mi']):.4f}"
        lines.append(f"- `{row['feature']}` verdict={row['verdict']} MI={mi} rho={float(row['rho']):+.4f}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description="Leakage-aware feature intelligence report.")
    p.add_argument("--data", required=True, help="Parquet/CSV artifact")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--label-col", default="bias_label")
    args = p.parse_args()

    data = os.path.abspath(args.data)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_parquet(data) if data.lower().endswith((".parquet", ".pq")) else pd.read_csv(data, low_memory=False)
    report = build_report(df, args.label_col)
    json_path = os.path.join(out_dir, "feature_intelligence_report.json")
    md_path = os.path.join(out_dir, "feature_intelligence_report.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    write_markdown(report, md_path)
    print(f"✅ Report saved:\n   JSON: {json_path}\n   MD:   {md_path}")
    if report["label_artifacts_excluded"]:
        print(f"✅ LABEL_ARTIFACT excluded from ranking: {len(report['label_artifacts_excluded'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
