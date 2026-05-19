import pandas as pd

from clean_mbo_mbp_data import CleanConfig, clean_mbo_frame, clean_mbp_frame


def test_clean_mbo_removes_invalid_rows_duplicates_and_outliers():
    cfg = CleanConfig(tick_size=0.0001, expected_root="6B")
    df = pd.DataFrame(
        [
            {
                "ts_recv": "2025-01-01T00:00:01Z",
                "ts_event": "2025-01-01T00:00:01Z",
                "action": "A",
                "side": "B",
                "price": 1.2500,
                "size": 1,
                "order_id": 10,
                "sequence": 1,
                "symbol": "6BM5",
            },
            {
                "ts_recv": "2025-01-01T00:00:02Z",
                "ts_event": "2025-01-01T00:00:01Z",
                "action": "A",
                "side": "B",
                "price": 1.2500,
                "size": 1,
                "order_id": 10,
                "sequence": 2,
                "symbol": "6BM5",
            },
            {
                "ts_recv": "2025-01-01T00:00:03Z",
                "ts_event": "2025-01-01T00:00:03Z",
                "action": "R",
                "side": "N",
                "price": None,
                "size": 0,
                "order_id": 0,
                "sequence": 3,
                "symbol": "6BM5",
            },
            {
                "ts_recv": "2025-01-01T00:00:04Z",
                "ts_event": "2025-01-01T00:00:04Z",
                "action": "A",
                "side": "A",
                "price": 99.0,
                "size": 1,
                "order_id": 11,
                "sequence": 4,
                "symbol": "6BM5",
            },
        ]
    )

    cleaned, stats = clean_mbo_frame(df, cfg=cfg)

    assert len(cleaned) == 1
    assert cleaned["price"].tolist() == [1.25]
    assert cleaned["sequence"].tolist() == [2]
    assert stats["total_dropped_rows"] == 3
    assert cleaned["ts_event"].is_monotonic_increasing


def test_clean_mbp_drops_invalid_bbo_and_recomputes_mid_price():
    cfg = CleanConfig(tick_size=0.0001, expected_root="6B")
    base = {
        "ts_recv": "2025-01-01T00:00:00Z",
        "rtype": 10,
        "publisher_id": 1,
        "instrument_id": 1,
        "action": "A",
        "side": "N",
        "depth": 0,
        "price": 99.0,
        "size": 1,
        "flags": 0,
        "ts_in_delta": 0,
        "sequence": 1,
        "symbol": "6BM5",
        "bid_sz_00": 10,
        "ask_sz_00": 10,
    }
    rows = []
    rows.append({**base, "ts_event": "2025-01-01T00:00:03Z", "bid_px_00": 1.2500, "ask_px_00": 1.2502})
    rows.append({**base, "ts_event": "2025-01-01T00:00:01Z", "bid_px_00": 1.2503, "ask_px_00": 1.2502})
    rows.append({**base, "ts_event": "2025-01-01T00:00:02Z", "bid_px_00": 1.2500, "ask_px_00": 1.2500})
    rows.append({**base, "ts_event": "2025-01-01T00:00:04Z", "bid_px_00": None, "ask_px_00": 1.2502})
    df = pd.DataFrame(rows)

    cleaned, stats = clean_mbp_frame(df, cfg=cfg)

    assert len(cleaned) == 1
    assert cleaned["ts_event"].is_monotonic_increasing
    assert cleaned["price"].iloc[0] == 1.2501
    assert cleaned["price_raw"].iloc[0] == 99.0
    assert stats["total_dropped_rows"] == 3
