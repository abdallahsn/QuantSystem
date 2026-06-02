"""
plot_refinery_compare_5m.py
---------------------------
Build daily 5m comparison pages for raw market data vs refinery labels.
"""

from __future__ import annotations

import argparse

from modules.refinery_compare_5m_report import generate_refinery_compare_5m_report


def main():
    p = argparse.ArgumentParser(description="Draw daily 5m Raw vs Refinery comparison pages")
    p.add_argument("--mbo", required=True, help="raw MBO file")
    p.add_argument("--refinery_csv", required=True, help="training_features_ready.csv from stage 1")
    p.add_argument("--output", default="outputs_v19/refinery_compare_5m", help="directory to write comparison pages")
    p.add_argument("--mbp", default="", help="optional MBP10 file for candle fallback when MBO has gaps")
    p.add_argument("--freq", default="5min", help="resample frequency")
    p.add_argument("--raw_chunksize", type=int, default=500_000, help="chunk size for raw MBO/MBP reading")
    p.add_argument("--refinery_chunksize", type=int, default=250_000, help="chunk size for refinery CSV reading")
    p.add_argument("--start", default=None, help="optional start timestamp")
    p.add_argument("--end", default=None, help="optional end timestamp")
    args = p.parse_args()

    summary = generate_refinery_compare_5m_report(
        mbo_path=args.mbo,
        refinery_csv=args.refinery_csv,
        output_dir=args.output,
        mbp_path=args.mbp,
        freq=args.freq,
        raw_chunksize=args.raw_chunksize,
        refinery_chunksize=args.refinery_chunksize,
        start_ts=args.start,
        end_ts=args.end,
    )

    print("\n📊 Raw vs Refinery 5m comparison جاهز")
    for key, path in summary.get("files", {}).items():
        print(f"  {key}: {path}")
    print(f"  days: {summary.get('n_days', 0)}")


if __name__ == "__main__":
    main()
