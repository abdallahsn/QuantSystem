#!/usr/bin/env python3
"""
يطبّق على backtest_v19.py:
  - باراميتر long_only في run_causal_backtest + تخطي فتح صفقات SHORT
  - --long_only في argparse + تمرير long_only=args.long_only
  - طباعة تشخيصية عند long_only
  - حقول long_only و max_horizon_steps في ملخص JSON

يدعم التشغيل المتكرر (idempotent): إن وُجد التعديل يُتخطّى.

الاستخدام:
  python3 patch_backtest_v19_daytrade.py
  python3 patch_backtest_v19_daytrade.py --target /path/to/backtest_v19.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _repl(text: str, old: str, new: str, label: str, changes: list[str]) -> str:
    if old not in text:
        if new.strip() in text or all(part in text for part in new.split() if len(part) > 20):
            return text
        raise RuntimeError(f"[patch] لم أجد النص المتوقع لـ {label!r} — راجع نسخة الملف.")
    changes.append(label)
    return text.replace(old, new, 1)


def patch_file(path: Path) -> None:
    raw = path.read_text(encoding="utf-8")
    original = raw
    changes: list[str] = []

    # 1) توقيع run_causal_backtest
    sig_old = "    skip_event_gate: bool = False,\n) -> tuple[pd.DataFrame, pd.DataFrame, dict]:"
    sig_new = "    skip_event_gate: bool = False,\n    long_only: bool = False,\n) -> tuple[pd.DataFrame, pd.DataFrame, dict]:"
    if sig_old in raw:
        raw = _repl(raw, sig_old, sig_new, "signature: long_only parameter", changes)
    elif "long_only: bool = False" not in raw:
        raise RuntimeError("[patch] توقيع run_causal_backtest غير متوقع (لا long_only ولا النمط القديم).")

    # 2) طباعة Causal replay
    old_print = (
        "    print(\n"
        "        f\"  [backtest] Causal replay: {n_replay:,} rows | \"\n"
        "        f\"scoring_mask={int(scoring_mask.sum()):,}\",\n"
        "        flush=True,\n"
        "    )\n"
    )
    new_print = (
        "    print(\n"
        "        f\"  [backtest] Causal replay: {n_replay:,} rows | \"\n"
        "        f\"scoring_mask={int(scoring_mask.sum()):,}\"\n"
        "        f\"{' | long_only=True (SHORT signals not executed)' if long_only else ''}\",\n"
        "        flush=True,\n"
        "    )\n"
    )
    if old_print in raw:
        raw = _repl(raw, old_print, new_print, "print: long_only hint", changes)
    elif "long_only=True (SHORT signals not executed)" in raw:
        pass
    else:
        raise RuntimeError(
            "[patch] كتلة الطباعة لـ Causal replay لا تطابق النسخة المتوقعة ولا تحتوي long_only."
        )

    # 3) تخطي SHORT
    old_trade = (
        "        if tradeable and direction in ('LONG', 'SHORT'):\n"
        "            size = int(position_size_from_prediction(\n"
    )
    new_trade = (
        "        if tradeable and direction in ('LONG', 'SHORT'):\n"
        "            if long_only and direction == 'SHORT':\n"
        "                pred['trade_skip_reason'] = 'long_only_backtest'\n"
        "                results.append(pred)\n"
        "                continue\n"
        "            size = int(position_size_from_prediction(\n"
    )
    if old_trade in raw:
        raw = _repl(raw, old_trade, new_trade, "loop: skip SHORT when long_only", changes)
    elif "long_only_backtest" in raw:
        pass
    else:
        raise RuntimeError("[patch] لم أجد كتلة tradeable/LONG/SHORT المتوقعة وليس فيها long_only_backtest.")

    # 4) ملخص JSON
    old_sum = (
        "        'visual_diagnostics': visual_diag,\n"
        "    }\n"
        "    if 'true_bias' in results_df.columns and 'direction_probs' in results_df.columns and len(results_df):\n"
    )
    new_sum = (
        "        'visual_diagnostics': visual_diag,\n"
        "        'long_only': bool(long_only),\n"
        "        'max_horizon_steps': None if max_horizon_steps is None else int(max_horizon_steps),\n"
        "    }\n"
        "    if 'true_bias' in results_df.columns and 'direction_probs' in results_df.columns and len(results_df):\n"
    )
    if "'long_only': bool(long_only)," in raw and "'max_horizon_steps': None if max_horizon_steps is None else int(max_horizon_steps)," in raw:
        pass
    elif "'long_only': bool(long_only)," not in raw:
        raw = _repl(raw, old_sum, new_sum, "summary: long_only + max_horizon_steps", changes)
    else:
        raise RuntimeError("[patch] long_only في الملخص موجود لكن max_horizon_steps ناقص — راجع الملف يدوياً.")

    # 5) argparse
    old_arg = (
        "    p.add_argument('--skip_event_gate', action='store_true',\n"
        "                   help='disable EventGate for this run (diagnostic only)')\n"
        "    args = p.parse_args()\n"
    )
    new_arg = (
        "    p.add_argument('--skip_event_gate', action='store_true',\n"
        "                   help='disable EventGate for this run (diagnostic only)')\n"
        "    p.add_argument(\n"
        "        '--long_only',\n"
        "        action='store_true',\n"
        "        help='open only LONG positions; SHORT signals stay in logs (trade_skip_reason=long_only_backtest)',\n"
        "    )\n"
        "    args = p.parse_args()\n"
    )
    if "'--long_only'" in raw or '"--long_only"' in raw:
        pass
    elif old_arg in raw:
        raw = _repl(raw, old_arg, new_arg, "argparse: --long_only", changes)
    else:
        raise RuntimeError(
            "[patch] لم أجد كتلة argparse (--skip_event_gate ثم parse_args) ولا وجدت --long_only."
        )

    # 6) استدعاء run_causal_backtest
    old_call = (
        "        skip_event_gate=args.skip_event_gate,\n"
        "    )\n"
        "    summary['oos_guard'] = oos_guard\n"
    )
    new_call = (
        "        skip_event_gate=args.skip_event_gate,\n"
        "        long_only=args.long_only,\n"
        "    )\n"
        "    summary['oos_guard'] = oos_guard\n"
    )
    if "long_only=args.long_only" in raw:
        pass
    elif old_call in raw:
        raw = _repl(raw, old_call, new_call, "main: pass long_only", changes)
    else:
        raise RuntimeError("[patch] لم أجد استدعاء run_causal_backtest النهائي المتوقع.")

    if raw == original:
        print(f"[OK] لا تغييرات مطلوبة (الملف محدّث مسبقاً): {path}")
        return

    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    path.write_text(raw, encoding="utf-8")
    print(f"[OK] نُسخ احتياطي: {bak}")
    print(f"[OK] تُطبَّق التعديلات على: {path}")
    for c in changes:
        print(f"     - {c}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Patch backtest_v19.py for day-trade LONG-only + horizon via CLI.")
    ap.add_argument(
        "--target",
        default=None,
        help="مسار backtest_v19.py (الافتراضي: بجانب هذا السكربت)",
    )
    args = ap.parse_args()
    base = Path(__file__).resolve().parent
    target = Path(args.target).resolve() if args.target else base / "backtest_v19.py"
    if not target.is_file():
        print(f"[ERR] الملف غير موجود: {target}", file=sys.stderr)
        return 1
    try:
        patch_file(target)
    except RuntimeError as e:
        print(f"[ERR] {e}", file=sys.stderr)
        return 2
    print()
    print("تذكير: أفق 2–3 ساعات على شموع 5 دقائق يمرّر من سطر الأوامر:")
    print("  --max_horizon_steps 24   # ~2 ساعة")
    print("  --max_horizon_steps 36   # ~3 ساعات")
    print("  --long_only              # تنفيذ LONG فقط")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
