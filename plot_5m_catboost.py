"""
plot_5m_catboost.py - Standalone 5m CatBoost dashboard
"""

from __future__ import annotations

import argparse

from modules.catboost_5m_report import generate_catboost_5m_report


def main():
    p = argparse.ArgumentParser(description="Draw a 5m CatBoost dashboard")
    p.add_argument("--csv", required=True, help="training_features_ready.csv from stage 1")
    p.add_argument("--models_dir", default="outputs_v19", help="directory containing CatBoost/scaler artifacts")
    p.add_argument("--output", default=None, help="directory to write HTML and CSV reports")
    p.add_argument("--freq", default="5min", help="resample frequency, e.g. 5min")
    p.add_argument("--name", default="catboost_5m", help="output file prefix")
    p.add_argument("--raw_only", action="store_true", help="skip decision policy and RSM; show raw CatBoost directions only")
    args = p.parse_args()

    summary = generate_catboost_5m_report(
        csv_path=args.csv,
        models_dir=args.models_dir,
        output_dir=args.output,
        freq=args.freq,
        report_name=args.name,
        apply_decision_policy=not args.raw_only,
        apply_rsm=not args.raw_only,
    )

    print("\n📊 5m CatBoost dashboard جاهز")
    print(f"  report_mode: {summary.get('report_mode')}")
    for key, path in summary.get("files", {}).items():
        print(f"  {key}: {path}")
    print(f"  direction_counts: {summary.get('direction_counts', {})}")
    print(f"  transitions: {summary.get('transitions', 0)}")


if __name__ == "__main__":
    main()
