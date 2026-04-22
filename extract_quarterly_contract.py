"""
extract_quarterly_contract.py
=============================
يقرأ ملف CSV من Databento يحتوي على أكثر من عقد لنفس المنتج،
ويستخرج العقد الفصلي المناسب بناءً على شهر البيانات.

Quarter Mapping:
  Q1 (January-March)   -> H  (March contract)
  Q2 (April-June)      -> M  (June contract)
  Q3 (July-September)  -> U  (September contract)
  Q4 (October-December)-> Z  (December contract)

أمثلة:
  python extract_quarterly_contract.py databento.csv
  python extract_quarterly_contract.py databento.csv --month 4
  python extract_quarterly_contract.py databento.csv --month 4 --root ES
  python extract_quarterly_contract.py databento.csv --output filtered.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


MONTH_TO_QUARTER_CODE = {
    1: "H",
    2: "H",
    3: "H",
    4: "M",
    5: "M",
    6: "M",
    7: "U",
    8: "U",
    9: "U",
    10: "Z",
    11: "Z",
    12: "Z",
}

QUARTER_MONTH_NAME = {
    "H": "March (Q1)",
    "M": "June (Q2)",
    "U": "September (Q3)",
    "Z": "December (Q4)",
}

MONTH_NAMES = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

TIMESTAMP_COLUMNS = ("ts_event", "ts_recv", "timestamp", "date")
SYMBOL_COLUMNS = ("raw_symbol", "symbol")
COMPRESSION_SUFFIXES = (".csv.gz", ".csv.zst", ".csv.bz2", ".csv.xz", ".csv")
CONTRACT_RE = re.compile(r"^(?P<root>.+?)(?P<month>[FGHJKMNQUVXZ])(?P<year>\d{1,4})$", re.IGNORECASE)


def detect_month_from_filename(filename: str) -> int | None:
    """
    يحاول استخراج الشهر من اسم الملف.
    يدعم صيغ مثل:
      data_2025_04.csv
      ES_2025-04.csv
      april_2025.csv
      2025_04_ES.csv
    """
    name = Path(filename).stem.lower()

    match = re.search(r"(?<!\d)20\d{2}[-_](\d{1,2})(?!\d)", name)
    if match:
        month = int(match.group(1))
        return month if 1 <= month <= 12 else None

    match = re.search(r"(?<!\d)(\d{1,2})[-_]20\d{2}(?!\d)", name)
    if match:
        month = int(match.group(1))
        return month if 1 <= month <= 12 else None

    for month_name, month_number in MONTH_NAMES.items():
        if re.search(rf"(?<![a-z]){re.escape(month_name)}(?![a-z])", name):
            return month_number

    return None


def detect_year_from_filename(filename: str) -> int | None:
    name = Path(filename).stem.lower()
    match = re.search(r"(?<!\d)(20\d{2})(?!\d)", name)
    if match:
        return int(match.group(1))
    return None


def choose_existing_column(columns: list[str], preferred: tuple[str, ...]) -> str | None:
    for col in preferred:
        if col in columns:
            return col
    return None


def iter_csv_chunks(path: str, *, chunksize: int, usecols=None):
    yield from pd.read_csv(
        path,
        chunksize=chunksize,
        usecols=usecols,
        low_memory=False,
        compression="infer",
    )


def read_header(path: str) -> list[str]:
    return pd.read_csv(path, nrows=0, compression="infer").columns.tolist()


def detect_month_from_data(path: str, ts_col: str, *, chunksize: int) -> tuple[int, str]:
    month_counts: Counter[int] = Counter()

    for chunk in iter_csv_chunks(path, chunksize=chunksize, usecols=[ts_col]):
        timestamps = pd.to_datetime(chunk[ts_col], utc=True, errors="coerce").dropna()
        if timestamps.empty:
            continue
        month_counts.update(timestamps.dt.month.value_counts().to_dict())

    if not month_counts:
        raise ValueError(f"مش قادر أحدد الشهر من عمود الوقت: {ts_col}")

    month = month_counts.most_common(1)[0][0]
    return int(month), ts_col


def detect_year_from_data(path: str, ts_col: str, *, chunksize: int) -> int | None:
    year_counts: Counter[int] = Counter()

    for chunk in iter_csv_chunks(path, chunksize=chunksize, usecols=[ts_col]):
        timestamps = pd.to_datetime(chunk[ts_col], utc=True, errors="coerce").dropna()
        if timestamps.empty:
            continue
        year_counts.update(timestamps.dt.year.value_counts().to_dict())

    if not year_counts:
        return None

    return int(year_counts.most_common(1)[0][0])


def parse_contract_symbol(symbol: str) -> dict | None:
    token = str(symbol).strip().upper()
    if not token:
        return None

    match = CONTRACT_RE.match(token)
    if not match:
        return None

    return {
        "symbol": token,
        "root": match.group("root"),
        "month_code": match.group("month").upper(),
        "year_suffix": match.group("year"),
    }


def infer_full_year(year_suffix: str, reference_year: int | None) -> int | None:
    if not year_suffix:
        return None

    if len(year_suffix) == 4:
        return int(year_suffix)

    value = int(year_suffix)
    modulus = 10 ** len(year_suffix)

    if reference_year is None:
        if len(year_suffix) <= 2:
            return 2000 + value
        return value

    base = reference_year - (reference_year % modulus)
    candidates = [base + value + (offset * modulus) for offset in range(-2, 3)]
    return min(candidates, key=lambda year: abs(year - reference_year))


def scan_contract_symbols(path: str, symbol_col: str, *, chunksize: int) -> tuple[Counter, dict, Counter]:
    symbol_counts: Counter[str] = Counter()
    symbol_info: dict[str, dict] = {}
    root_counts: Counter[str] = Counter()

    for chunk in iter_csv_chunks(path, chunksize=chunksize, usecols=[symbol_col]):
        if symbol_col not in chunk.columns:
            continue

        counts = (
            chunk[symbol_col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
            .value_counts()
        )

        for symbol, count in counts.items():
            parsed = parse_contract_symbol(symbol)
            if parsed is None:
                continue

            symbol_counts[symbol] += int(count)
            symbol_info[symbol] = parsed
            root_counts[parsed["root"]] += int(count)

    return symbol_counts, symbol_info, root_counts


def choose_target_symbol(
    *,
    symbol_counts: Counter,
    symbol_info: dict[str, dict],
    root_counts: Counter,
    target_month_code: str,
    dataset_year: int | None,
    prefer_year_match: bool,
    forced_root: str | None,
) -> tuple[str, str]:
    if not symbol_counts:
        raise ValueError("لم أجد أي رموز عقود قابلة للتحليل داخل الملف (مثل ESH5 أو 6BM5)")

    root = forced_root.strip().upper() if forced_root else None
    if root is None:
        root = root_counts.most_common(1)[0][0]

    candidates = [
        symbol
        for symbol, info in symbol_info.items()
        if info["root"] == root and info["month_code"] == target_month_code
    ]

    if not candidates:
        available_roots = ", ".join(root for root, _ in root_counts.most_common(10))
        raise ValueError(
            f"لم أجد عقداً فصلياً بالكود {target_month_code} للـ root={root}. "
            f"Available roots: {available_roots or 'N/A'}"
        )

    def sort_key(symbol: str):
        info = symbol_info[symbol]
        full_year = infer_full_year(info["year_suffix"], dataset_year)
        year_distance = abs(full_year - dataset_year) if dataset_year is not None and full_year is not None else 10**9
        if prefer_year_match:
            return (year_distance, -symbol_counts[symbol], symbol)
        return (-symbol_counts[symbol], year_distance, symbol)

    target_symbol = sorted(candidates, key=sort_key)[0]
    return target_symbol, root


def build_default_output_path(input_file: str, target_symbol: str) -> str:
    input_path = Path(input_file)
    filename = input_path.name

    base_name = None
    for suffix in COMPRESSION_SUFFIXES:
        if filename.lower().endswith(suffix):
            base_name = filename[: -len(suffix)]
            break

    if base_name is None:
        base_name = input_path.stem

    return str(input_path.with_name(f"{base_name}_{target_symbol}.csv"))


def filter_to_symbol(
    input_file: str,
    output_file: str,
    symbol_col: str,
    target_symbol: str,
    *,
    chunksize: int,
) -> int:
    temp_output = f"{output_file}.tmp"
    rows_written = 0
    wrote_header = False

    if os.path.exists(temp_output):
        os.remove(temp_output)

    for chunk in iter_csv_chunks(input_file, chunksize=chunksize):
        if symbol_col not in chunk.columns:
            raise ValueError(f"الملف لا يحتوي على عمود الرموز المطلوب: {symbol_col}")

        mask = chunk[symbol_col].astype(str).str.strip().str.upper() == target_symbol
        filtered = chunk.loc[mask]

        if filtered.empty:
            continue

        filtered.to_csv(
            temp_output,
            mode="w" if not wrote_header else "a",
            header=not wrote_header,
            index=False,
        )
        rows_written += len(filtered)
        wrote_header = True

    if rows_written == 0:
        if os.path.exists(temp_output):
            os.remove(temp_output)
        raise ValueError(f"لم أجد أي صفوف مطابقة للعقد {target_symbol}")

    os.replace(temp_output, output_file)
    return rows_written


def extract_quarterly_contract(
    input_file: str,
    *,
    month: int | None = None,
    year: int | None = None,
    output_file: str | None = None,
    root: str | None = None,
    chunksize: int = 250_000,
) -> str:
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"الملف غير موجود: {input_file}")

    columns = read_header(input_file)
    symbol_col = choose_existing_column(columns, SYMBOL_COLUMNS)
    ts_col = choose_existing_column(columns, TIMESTAMP_COLUMNS)

    if symbol_col is None:
        raise ValueError("الملف لا يحتوي على عمود symbol أو raw_symbol")

    month_source = None
    if month is not None:
        if not 1 <= month <= 12:
            raise ValueError("--month يجب أن يكون بين 1 و 12")
        month_source = "manual"
    else:
        month = detect_month_from_filename(input_file)
        if month is not None:
            month_source = "filename"
        elif ts_col is not None:
            month, ts_used = detect_month_from_data(input_file, ts_col, chunksize=chunksize)
            month_source = f"data:{ts_used}"
        else:
            raise ValueError("لم أستطع تحديد الشهر من اسم الملف أو من أعمدة الوقت. استخدم --month")

    year_source = None
    if year is not None:
        year_source = "manual"
    else:
        year = detect_year_from_filename(input_file)
        if year is not None:
            year_source = "filename"
        elif ts_col is not None:
            year = detect_year_from_data(input_file, ts_col, chunksize=chunksize)
            if year is not None:
                year_source = f"data:{ts_col}"

    target_month_code = MONTH_TO_QUARTER_CODE[month]
    symbol_counts, symbol_info, root_counts = scan_contract_symbols(input_file, symbol_col, chunksize=chunksize)
    target_symbol, detected_root = choose_target_symbol(
        symbol_counts=symbol_counts,
        symbol_info=symbol_info,
        root_counts=root_counts,
        target_month_code=target_month_code,
        dataset_year=year,
        prefer_year_match=year_source in {"manual", "filename"},
        forced_root=root,
    )

    if output_file is None:
        output_file = build_default_output_path(input_file, target_symbol)

    rows_written = filter_to_symbol(
        input_file,
        output_file,
        symbol_col=symbol_col,
        target_symbol=target_symbol,
        chunksize=chunksize,
    )

    print(f"\n{'=' * 64}")
    print("Databento Quarterly Contract Extractor")
    print(f"{'=' * 64}")
    print(f"Input file        : {input_file}")
    print(f"Output file       : {output_file}")
    print(f"Symbol column     : {symbol_col}")
    print(f"Timestamp column  : {ts_col or 'N/A'}")
    print(f"Detected month    : {month} ({month_source})")
    print(f"Detected year     : {year if year is not None else 'N/A'} ({year_source or 'unavailable'})")
    print(f"Quarter code      : {target_month_code} -> {QUARTER_MONTH_NAME[target_month_code]}")
    print(f"Selected root     : {detected_root}")
    print(f"Target contract   : {target_symbol}")
    print(f"Rows extracted    : {rows_written:,}")
    print(f"{'=' * 64}\n")

    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract the correct quarterly contract from a Databento CSV file."
    )
    parser.add_argument("input_file", help="Databento CSV file path")
    parser.add_argument("--month", type=int, help="Override month manually (1-12)")
    parser.add_argument("--year", type=int, help="Override year manually")
    parser.add_argument("--root", help="Force the contract root, e.g. ES, MES, 6B")
    parser.add_argument("--output", help="Output CSV path")
    parser.add_argument("--chunksize", type=int, default=250_000, help="CSV chunksize for streaming")

    args = parser.parse_args()

    try:
        extract_quarterly_contract(
            args.input_file,
            month=args.month,
            year=args.year,
            output_file=args.output,
            root=args.root,
            chunksize=max(1, int(args.chunksize)),
        )
    except Exception as exc:  # pragma: no cover - CLI-friendly exit path
        print(f"\n[ERROR] {exc}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
