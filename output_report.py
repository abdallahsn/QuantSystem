"""
output_report.py — HTML signal report compatible with test_system.py
"""
from __future__ import annotations

import html
import os


BIAS_LABELS = {
    0: "LONG",
    1: "SHORT",
    2: "NEUTRAL",
}

SETUP_LABELS = {
    0: "Absorption",
    1: "Spoofing",
    2: "OBI",
    3: "Mixed",
}


def _fmt_num(value) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))

    if abs(num) >= 1000:
        return f"{num:,.2f}"
    if abs(num) >= 1:
        return f"{num:.4f}"
    return f"{num:.6f}".rstrip("0").rstrip(".")


def _reason_cards(reasons: list[tuple]) -> str:
    cards = []
    for reason in reasons or []:
        if len(reason) >= 3:
            icon, text, color = reason[:3]
        elif len(reason) == 2:
            icon, text = reason
            color = "#4caf50"
        else:
            icon, text, color = "•", str(reason[0]), "#607d8b"

        cards.append(
            f"""
            <div class="reason-card" style="border-color:{html.escape(str(color))}">
              <div class="reason-icon">{html.escape(str(icon))}</div>
              <div class="reason-text">{html.escape(str(text))}</div>
            </div>
            """
        )
    return "".join(cards) or '<div class="empty">No reasons available.</div>'


def _kv_rows(items: dict) -> str:
    rows = []
    for key, value in (items or {}).items():
        rows.append(
            f"""
            <tr>
              <td>{html.escape(str(key))}</td>
              <td>{_fmt_num(value)}</td>
            </tr>
            """
        )
    return "".join(rows) or '<tr><td colspan="2">No data</td></tr>'


def generate_report(signal_data: dict, output_path: str) -> str:
    signal_data = signal_data or {}
    bias = BIAS_LABELS.get(signal_data.get("bias"), str(signal_data.get("bias", "N/A")))
    setup = SETUP_LABELS.get(signal_data.get("setup"), str(signal_data.get("setup", "N/A")))
    confidence = float(signal_data.get("confidence", 0.0) or 0.0)
    session = html.escape(str(signal_data.get("session", "Unknown")))
    timestamp = html.escape(str(signal_data.get("timestamp", "Unknown")))
    reasons_html = _reason_cards(signal_data.get("reasons", []))
    features_html = _kv_rows(signal_data.get("features", {}))
    levels_html = _kv_rows(signal_data.get("levels", {}))

    confidence_pct = max(0.0, min(confidence, 1.0)) * 100.0
    accent = "#1f9d55" if confidence_pct >= 65 else "#e67e22" if confidence_pct >= 45 else "#c0392b"

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>QuantSystem Signal Report</title>
  <style>
    :root {{
      --bg: #f4f0e8;
      --panel: rgba(255, 252, 247, 0.94);
      --ink: #1f2430;
      --muted: #6b7280;
      --line: rgba(31, 36, 48, 0.10);
      --accent: {accent};
      --accent-soft: rgba(31, 157, 85, 0.12);
      --shadow: 0 18px 50px rgba(55, 41, 25, 0.14);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(184, 115, 51, 0.14), transparent 30%),
        radial-gradient(circle at top right, rgba(32, 114, 164, 0.12), transparent 26%),
        linear-gradient(180deg, #f8f4eb 0%, #efe7d7 100%);
      min-height: 100vh;
    }}
    .page {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 40px 24px 56px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 20px;
      align-items: stretch;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }}
    .hero-main {{
      padding: 28px;
      position: relative;
      overflow: hidden;
    }}
    .hero-main::after {{
      content: "";
      position: absolute;
      inset: auto -120px -120px auto;
      width: 280px;
      height: 280px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(31, 157, 85, 0.18), transparent 68%);
    }}
    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.18em;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 12px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(34px, 5vw, 58px);
      line-height: 0.95;
    }}
    .sub {{
      margin: 0;
      font-size: 18px;
      color: var(--muted);
      max-width: 42rem;
    }}
    .hero-stats {{
      display: grid;
      gap: 14px;
      padding: 20px;
    }}
    .stat {{
      padding: 18px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(245,239,228,0.95));
    }}
    .stat .label {{
      display: block;
      font-size: 12px;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .stat .value {{
      font-size: 28px;
      font-weight: 700;
    }}
    .accent {{
      color: var(--accent);
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 20px;
      margin-top: 20px;
    }}
    .section {{
      padding: 24px;
    }}
    h2 {{
      margin: 0 0 16px;
      font-size: 20px;
      letter-spacing: 0.02em;
    }}
    .reason-grid {{
      display: grid;
      gap: 12px;
    }}
    .reason-card {{
      display: grid;
      grid-template-columns: 42px 1fr;
      gap: 12px;
      align-items: center;
      padding: 14px 16px;
      border: 1px solid var(--line);
      border-left-width: 6px;
      border-radius: 16px;
      background: rgba(255,255,255,0.70);
    }}
    .reason-icon {{
      font-size: 24px;
      text-align: center;
    }}
    .reason-text {{
      font-size: 15px;
      line-height: 1.5;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border-radius: 16px;
    }}
    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      font-size: 14px;
    }}
    th {{
      font-size: 12px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--muted);
      background: rgba(31, 36, 48, 0.03);
    }}
    tr:last-child td {{
      border-bottom: none;
    }}
    .footer-note {{
      margin-top: 20px;
      padding: 18px 20px;
      border-radius: 18px;
      background: linear-gradient(135deg, rgba(31, 157, 85, 0.10), rgba(32, 114, 164, 0.08));
      border: 1px solid rgba(31, 36, 48, 0.08);
      color: #334155;
      line-height: 1.7;
    }}
    .empty {{
      color: var(--muted);
      padding: 16px;
      border: 1px dashed var(--line);
      border-radius: 16px;
    }}
    @media (max-width: 900px) {{
      .hero, .grid {{
        grid-template-columns: 1fr;
      }}
      .page {{
        padding: 24px 14px 36px;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="panel hero-main">
        <div class="eyebrow">QuantSystem Signal Report</div>
        <h1>{html.escape(str(bias))}</h1>
        <p class="sub">
          Setup: <strong>{html.escape(str(setup))}</strong><br>
          Session: <strong>{session}</strong><br>
          Timestamp: <strong>{timestamp}</strong>
        </p>
      </div>
      <div class="hero-stats">
        <div class="panel stat">
          <span class="label">Confidence</span>
          <div class="value accent">{confidence_pct:.1f}%</div>
        </div>
        <div class="panel stat">
          <span class="label">Bias</span>
          <div class="value">{html.escape(str(bias))}</div>
        </div>
        <div class="panel stat">
          <span class="label">Setup</span>
          <div class="value">{html.escape(str(setup))}</div>
        </div>
      </div>
    </section>

    <section class="grid">
      <div class="panel section">
        <h2>Decision Reasons</h2>
        <div class="reason-grid">{reasons_html}</div>
        <div class="footer-note">
          This report is designed to be read quickly during live review. It surfaces the directional bias,
          the setup family, the feature state, and the market reference levels in one place so the operator
          can validate whether the machine view matches the tape and the broader session context.
        </div>
      </div>

      <div class="panel section">
        <h2>Feature Snapshot</h2>
        <table>
          <thead><tr><th>Feature</th><th>Value</th></tr></thead>
          <tbody>{features_html}</tbody>
        </table>
      </div>
    </section>

    <section class="grid">
      <div class="panel section">
        <h2>Reference Levels</h2>
        <table>
          <thead><tr><th>Level</th><th>Value</th></tr></thead>
          <tbody>{levels_html}</tbody>
        </table>
      </div>

      <div class="panel section">
        <h2>Interpretation Guide</h2>
        <div class="footer-note">
          High confidence does not replace execution discipline. Use this page to confirm that price,
          imbalance, context, and levels are aligned. If the snapshot conflicts with the live order flow,
          treat the signal as informational rather than automatic. The goal is confluence, not blind trust.
        </div>
        <div class="footer-note">
          A healthy report should contain meaningful reasons, non-empty features, and reference levels that
          contextualize where price sits relative to the prior day and week structure. When those pieces line
          up, the report becomes far more useful as an execution filter.
        </div>
      </div>
    </section>
  </div>
</body>
</html>
"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    return output_path
