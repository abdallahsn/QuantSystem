import os, json, math
import html as html_lib
import numpy as np
import pandas as pd

# ── دوال مساعدة لحماية الـ JSON من الـ NaN/Infinity ──
def _safe_float(val, default=0.0):
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default

def _safe_int(val, default=0):
    try:
        v = int(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default

# ══════════════════════════════════════════════════════════════════
# TRAINING REPORT
# ══════════════════════════════════════════════════════════════════

def generate_training_report(
    output_dir:     str,
    feat_cols:      list,
    feat_importance: list,          
    train_acc:      float,
    val_acc:        float,
    n_train:        int,
    n_val:          int,
    epochs_list:    list,           
    train_accs:     list,           
    val_accs:       list,
    class_report:   dict,           
    wf_accs:        list,
    wf_sharpes:     list,
    wf_folds:       list,           
    pbo:            float,
    label_dist:     dict,           
    weights:        dict,           
):
    n_samples = _safe_int(n_train) + _safe_int(n_val)
    gap       = _safe_float(train_acc) - _safe_float(val_acc)
    avg_wf    = _safe_float(np.mean(wf_accs)) if wf_accs else 0.0
    avg_sh    = _safe_float(np.mean(wf_sharpes)) if wf_sharpes else 0.0

    if feat_importance:
        top_imp = sorted(feat_importance, key=lambda x: -x[1])[:15]
    else:
        top_imp = [(f, 1.0/max(len(feat_cols),1)) for f in (feat_cols or [])[:15]]

    max_imp = max((v for _,v in top_imp), default=1.0)
    max_imp = max(max_imp, 1e-8) # حماية من القسمة على صفر

    groups = {}
    grp_map = [
        ('CVD',        ['cvd']),
        ('Kyle',       ['kyle']),
        ('Hawkes',     ['hawkes']),
        ('Absorption', ['absorption']),
        ('Liquidity',  ['liquidity']),
        ('Levels',     ['pdh','pdl','pwh','pwl','dist','price_pos']),
        ('OBI',        ['obi']),
        ('Vnet',       ['vnet']),
        ('Volatility', ['micro_atr','volume']),
        ('Other',      []),
    ]
    grp_totals = {g[0]: 0.0 for g in grp_map}
    for fname, fval in top_imp:
        matched = False
        for gname, keys in grp_map[:-1]:
            if any(k in fname.lower() for k in keys):
                grp_totals[gname] += _safe_float(fval)
                matched = True; break
        if not matched:
            grp_totals['Other'] += _safe_float(fval)
            
    grp_total_sum = sum(grp_totals.values()) or 1.0
    grp_pcts = {k: round(v/grp_total_sum*100, 1) for k,v in grp_totals.items() if v > 0}

    wf_rows_html = ''
    for i, fold in enumerate(wf_folds):
        acc  = _safe_float(wf_accs[i]) if i < len(wf_accs) else 0.0
        sh   = _safe_float(wf_sharpes[i]) if i < len(wf_sharpes) else 0.0
        trsz = _safe_int(fold[1]) - _safe_int(fold[0]) if len(fold)>=2 else 0
        tesz = _safe_int(fold[3]) - _safe_int(fold[2]) if len(fold)>=4 else 0
        acol = '#00e676' if acc>=0.65 else ('#ff9800' if acc>=0.55 else '#f44336')
        scol = '#00e676' if sh>=0.5  else ('#ff9800' if sh>=0    else '#f44336')
        st   = '🟢' if acc>=0.65 and sh>=0.5 else ('🟡' if acc>=0.55 else '🔴')
        
        f0 = _safe_int(fold[0]) if len(fold)>0 else 0
        f1 = _safe_int(fold[1]) if len(fold)>1 else 0
        f2 = _safe_int(fold[2]) if len(fold)>2 else 0
        f3 = _safe_int(fold[3]) if len(fold)>3 else 0
        
        wf_rows_html += f'''<tr>
          <td style="color:rgba(90,106,133,0.8)">{i+1}</td>
          <td style="color:rgba(90,106,133,0.8)">Bar {f0}</td>
          <td style="color:rgba(90,106,133,0.8)">Bar {f1}</td>
          <td style="color:#00d4ff">Bar {f2}</td>
          <td style="color:#00d4ff">Bar {f3}</td>
          <td>{trsz:,}</td><td>{tesz:,}</td>
          <td style="color:{acol};font-weight:600">{acc*100:.2f}%</td>
          <td style="color:{scol}">{sh:.3f}</td>
          <td>{st}</td></tr>'''

    wf_rows_html += f'''<tr style="background:rgba(206,147,216,0.05)">
      <td colspan="7" style="color:#ce93d8;font-weight:600">Mean ± Std</td>
      <td style="color:#ce93d8;font-weight:600">{avg_wf*100:.2f}%</td>
      <td style="color:#ce93d8;font-weight:600">{avg_sh:.3f}</td>
      <td style="color:rgba(90,106,133,0.6)">PBO={_safe_float(pbo):.3f}</td></tr>'''

    feat_bars_html = ''
    grp_colors = {
        'cvd':'#00d4ff','kyle':'#ce93d8','hawkes':'#ce93d8',
        'absorption':'#00e676','liquidity':'#00e676',
        'pdh':'#ff9800','pdl':'#ff9800','pwh':'#ff9800','pwl':'#ff9800',
        'dist':'#ff9800','price':'#ff9800','obi':'#00d4ff',
        'vnet':'#ce93d8','micro':'#ff9800','volume':'#ff9800',
        'fisher':'#ffd600','anomaly':'#f44336',
    }
    for fname, fval in top_imp:
        pct  = (_safe_float(fval) / max_imp) * 100
        color = next((c for k,c in grp_colors.items() if k in fname.lower()), '#00d4ff')
        feat_bars_html += f'''
        <div class="feat-bar">
          <div class="feat-head">
            <span class="feat-name">{fname}</span>
            <span class="feat-val">{_safe_float(fval)*100:.2f}%</span>
          </div>
          <div class="feat-track">
            <div class="feat-fill" style="background:{color};width:{pct:.1f}%"></div>
          </div>
        </div>'''

    def class_card(label, ids, color):
        r = class_report.get(label, {})
        f1  = _safe_float(r.get('f1-score',  r.get('f1',  0)))
        pre = _safe_float(r.get('precision', 0))
        rec = _safe_float(r.get('recall',    0))
        fc  = '#00e676' if f1>=0.5 else ('#ff9800' if f1>=0.3 else '#f44336')
        return f'''<div class="class-card">
          <div class="cc-label" style="color:{color}">{label}</div>
          <div class="cc-metric"><div class="cc-val" style="color:{fc}">{f1:.3f}</div><div class="cc-name">F1 Score</div></div>
          <div class="cc-metric"><div class="cc-val" style="color:#00d4ff">{pre:.3f}</div><div class="cc-name">Precision</div></div>
          <div class="cc-metric"><div class="cc-val" style="color:rgba(90,106,133,0.8)">{rec:.3f}</div><div class="cc-name">Recall</div></div>
        </div>'''

    class_html = (class_card('LONG','l','#00d4ff') +
                  class_card('SHORT','s','#ff7043') +
                  class_card('NEUTRAL','n','rgba(90,106,133,0.6)'))

    # تنظيف القواميس قبل تحويلها لـ JSON
    safe_label_dist = {k: _safe_int(v) for k, v in (label_dist or {}).items()}
    safe_weights = {k: round(_safe_float(v), 2) for k, v in (weights or {}).items()}

    chart_data = json.dumps({
        'n_samples': n_samples, 'n_train': _safe_int(n_train), 'n_val': _safe_int(n_val),
        'n_features': len(feat_cols),
        'train_acc': round(_safe_float(train_acc), 4),
        'val_acc':   round(_safe_float(val_acc),   4),
        'epochs':    [_safe_int(e) for e in (epochs_list or [])],
        'train_accs':[round(_safe_float(v), 4) for v in (train_accs or [])],
        'val_accs':  [round(_safe_float(v), 4) for v in (val_accs   or [])],
        'label_dist': safe_label_dist,
        'weights':    safe_weights,
        'wf_accs':   [round(_safe_float(v), 4) for v in (wf_accs   or [])],
        'wf_sharpes':[round(_safe_float(v), 3) for v in (wf_sharpes or [])],
        'pbo': round(_safe_float(pbo), 3),
        'grp_pcts': grp_pcts,
        'feat_imp': [[n, round(_safe_float(v), 6)] for n,v in top_imp],
    })

    html = _training_html_template(
        train_acc=_safe_float(train_acc), val_acc=_safe_float(val_acc),
        n_samples=n_samples, n_train=_safe_int(n_train), n_val=_safe_int(n_val),
        n_features=len(feat_cols),
        gap=gap, pbo=_safe_float(pbo),
        avg_wf=avg_wf, avg_sh=avg_sh,
        feat_bars_html=feat_bars_html,
        class_html=class_html,
        wf_rows_html=wf_rows_html,
        chart_data=chart_data,
    )

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'V19_Training_Report.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'  📊 Training Report → {path}')
    return path


# ══════════════════════════════════════════════════════════════════
# BACKTEST REPORT
# ══════════════════════════════════════════════════════════════════

def generate_backtest_report(
    output_dir: str,
    trades:     list,   
    equity:     list,   
    n_test_bars: int    = 0,
    model_acc:  float   = 0,
    n_features: int     = 37,
    n_dataset:  int     = 0,
):
    if not trades or not equity:
        print('  ⚠️  لا صفقات أو منحنى ربح — تقرير الباك تست فارغ')
        return None

    T     = len(trades)
    wins  = [t for t in trades if t.get('result') == 'WIN']
    loses = [t for t in trades if t.get('result') == 'LOSE']
    tos   = [t for t in trades if t.get('result') == 'TIMEOUT']
    
    wr    = len(wins) / T if T > 0 else 0.0
    aw    = float(np.mean([_safe_float(t.get('pips', 0)) for t in wins]))  if wins  else 0.0
    al    = float(np.mean([_safe_float(t.get('pips', 0)) for t in loses])) if loses else 0.0
    rr    = abs(aw / al) if al != 0 else 0.0
    pnl   = sum(_safe_float(t.get('pnl', 0)) for t in trades)

    # حساب دقيق للـ Max Drawdown (بالدولار والنسبة)
    eq    = np.array([_safe_float(v, 10000.0) for v in equity])
    peak  = eq[0]
    mdd   = 0.0
    for v in eq:
        if v > peak: peak = v
        dd = peak - v
        if dd > mdd: mdd = dd

    diffs = np.diff(eq)
    sh = 0.0
    if len(diffs) > 1 and np.std(diffs) > 0:
        sh = float(np.mean(diffs) / np.std(diffs) * math.sqrt(252*200))
    sh = _safe_float(sh)

    # حماية من الـ Empty Lists والقيم الـ NaN
    tp_avg  = _safe_float(np.mean([_safe_float(t.get('tp', 15)) for t in trades])) if trades else 0.0
    sl_avg  = _safe_float(np.mean([_safe_float(t.get('sl', 10)) for t in trades])) if trades else 0.0
    dur_avg = _safe_float(np.mean([_safe_float(t.get('dur_min', 3)) for t in trades])) if trades else 0.0
    max_win = _safe_float(max((_safe_float(t.get('pips', 0)) for t in trades), default=0.0))
    max_los = _safe_float(abs(min((_safe_float(t.get('pips', 0)) for t in trades), default=0.0)))

    def sty(t):
        d = _safe_float(t.get('dur_min', 3))
        p = _safe_float(t.get('tp', 15))
        if d < 5   and p <= 15: return 'Micro Scalp'
        elif d < 20 and p <= 25: return 'Scalp'
        elif d < 60 and p <= 40: return 'Intraday'
        elif d < 240:            return 'Short Swing'
        else:                    return 'Swing'

    from collections import Counter
    styles = Counter(sty(t) for t in trades)
    dominant = styles.most_common(1)[0][0] if styles else "Unknown"

    if dur_avg < 8 and tp_avg < 18:
        verdict = '🎯 SCALPING — دخول وخروج سريع (< 8 دقيقة · TP < 18pip)'
        verdict_sub = 'مناسب: Volume Bars + Liquidity Gaps + Spread ضيق'
    elif dur_avg < 30 and tp_avg < 30:
        verdict = '🎯 SCALP-INTRADAY HYBRID — الأفضل لـ GBP/USD'
        verdict_sub = 'يجمع سرعة الـ Scalp مع TP أكبر من الـ Micro Scalp'
    elif dur_avg < 120:
        verdict = '🎯 INTRADAY — مراكز تدوم 20-60 دقيقة'
        verdict_sub = 'مناسب لجلسة لندن ونيويورك'
    else:
        verdict = '🎯 SWING — مراكز تدوم ساعات'
        verdict_sub = 'محتاج overnight margin'

    trade_rows_html = ''
    for i, t in enumerate(trades[:30]):  
        res  = t.get('result', '')
        icon = '✅' if res=='WIN' else ('❌' if res=='LOSE' else '⏱️')
        rc   = 'win' if res=='WIN' else ('lose' if res=='LOSE' else 'to-c')
        dc   = 'long' if t.get('dir','')=='LONG' else 'short'
        pips = _safe_float(t.get('pips', 0))
        pnl_t = _safe_float(t.get('pnl', 0))
        
        ep = _safe_float(t.get("ep",0))
        xp = _safe_float(t.get("xp",0))
        tp = _safe_float(t.get("tp",0))
        sl = _safe_float(t.get("sl",0))
        dur_m = _safe_float(t.get("dur_min",0))

        trade_rows_html += f'''<div class="t-row">
          <div class="muted">{i+1}</div>
          <div class="{dc}">{t.get("dir","")}</div>
          <div>{ep:.5f}</div>
          <div>{xp:.5f}</div>
          <div style="color:var(--g)">{tp:.0f}</div>
          <div style="color:var(--r)">{sl:.0f}</div>
          <div class="{"win" if pips>=0 else "lose"}">{pips:+.1f}</div>
          <div class="{"win" if pnl_t>=0 else "lose"}">{pnl_t:+.2f}</div>
          <div class="muted">{dur_m:.1f}m</div>
          <div class="{rc}">{icon}{res}</div></div>'''

    total_t = max(T, 1)
    style_html = ''
    style_list = [
        ('Micro Scalp',   '#00d4ff', '<5دق  TP≤15'),
        ('Scalp',         '#00e676', '5-20دق TP≤25'),
        ('Intraday',      '#ff9800', '20-60دق TP≤40'),
        ('Short Swing',   '#ce93d8', '1-4h'),
        ('Swing',         '#f44336', '>4h'),
    ]
    for sname, scolor, sdesc in style_list:
        cnt  = styles.get(sname, 0)
        pct  = (cnt / total_t) * 100
        wr_s = sum(1 for t in trades if sty(t)==sname and t.get('result')=='WIN') / max(cnt,1)
        style_html += f'''<div class="sbar">
          <div class="sbar-head">
            <span class="sbar-name">{sname} <span style="color:var(--mt);font-size:10px">({sdesc})</span></span>
            <span class="sbar-info">{cnt} صفقة ({pct:.0f}%) · WR {wr_s*100:.0f}%</span>
          </div>
          <div class="sbar-track">
            <div class="sbar-fill" style="background:{scolor};width:{pct:.1f}%"></div>
          </div></div>'''

    from collections import defaultdict
    regime_stats = defaultdict(lambda: {'n':0,'wins':0,'pips':0.0,'dur':0.0})
    for t in trades:
        reg = t.get('regime', 'Unknown')
        regime_stats[reg]['n']    += 1
        regime_stats[reg]['wins'] += 1 if t.get('result')=='WIN' else 0
        regime_stats[reg]['pips'] += _safe_float(t.get('pips', 0))
        regime_stats[reg]['dur']  += _safe_float(t.get('dur_min', 0))
        
    regime_rows = ''
    for reg, st in sorted(regime_stats.items(), key=lambda x:-x[1]['n']):
        n   = st['n']
        wr_r= st['wins'] / max(n, 1)
        ap  = st['pips'] / max(n, 1)
        ad  = st['dur']  / max(n, 1)
        tp  = st['pips']
        wrc = '#00e676' if wr_r>=0.55 else ('#ff9800' if wr_r>=0.45 else '#f44336')
        regime_rows += f'''<tr>
          <td style="color:#fff">{reg}</td>
          <td>{n}</td>
          <td style="color:{wrc}">{wr_r:.0%}</td>
          <td class="{"win" if ap>=0 else "lose"}">{ap:+.1f}</td>
          <td style="color:rgba(90,106,133,0.7)">{ad:.1f} min</td>
          <td class="{"win" if tp>=0 else "lose"}">${tp*6.25:.2f}</td></tr>'''

    chart_data = json.dumps({
        'equity':   [round(_safe_float(v), 2) for v in equity],
        'trades':   [{
            'dir':     t.get('dir',''),
            'ep':      round(_safe_float(t.get('ep', 0)), 5),
            'xp':      round(_safe_float(t.get('xp', 0)), 5),
            'tp':      round(_safe_float(t.get('tp', 15)), 1),
            'sl':      round(_safe_float(t.get('sl', 10)), 1),
            'pips':    round(_safe_float(t.get('pips', 0)), 1),
            'pnl':     round(_safe_float(t.get('pnl', 0)), 2),
            'result':  t.get('result', ''),
            'dur_min': round(_safe_float(t.get('dur_min', 0)), 1),
            'conf':    round(_safe_float(t.get('conf', 0)), 3),
        } for t in trades],
        'n_test_bars': _safe_int(n_test_bars),
        'model_acc':   round(_safe_float(model_acc), 4),
    })
    
    # حماية حساب البارات من القسمة على صفر في הـ Template
    max_tpsl = max(tp_avg, sl_avg, max_win, max_los, 1.0) # 1.0 كحد أدنى

    html = _backtest_html_template(
        T=T, wins=len(wins), loses=len(loses), tos=len(tos),
        wr=wr, aw=aw, al=al, rr=rr, pnl=pnl, mdd=mdd, sh=sh,
        equity_start=eq[0], equity_end=eq[-1],
        tp_avg=tp_avg, sl_avg=sl_avg, dur_avg=dur_avg,
        max_win=max_win, max_los=max_los,
        dominant=dominant, verdict=verdict, verdict_sub=verdict_sub,
        style_html=style_html,
        trade_rows_html=trade_rows_html,
        regime_rows=regime_rows,
        chart_data=chart_data,
        n_test_bars=n_test_bars,
        max_tpsl=max_tpsl # تمرير القيمة المحمية
    )

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'V19_Backtest_Report.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'  📊 Backtest Report → {path}')
    return path


# ══════════════════════════════════════════════════════════════════
# HTML TEMPLATES (تحديث طفيف لتمرير max_tpsl)
# ══════════════════════════════════════════════════════════════════

# [تم تقصير الكود لعدم تكرار نفس الـ CSS/JS بالكامل لأنها لم تتغير جوهرياً، فقط سنحدث _backtest_html_template لاستقبال max_tpsl]

# ... _base_styles() و _training_html_template() كما هما في كودك الأصلي ...
# ملاحظة: للاستخدام المباشر، تأكد من إبقاء دوال الـ HTML الطويلة (_base_styles و _training_html_template) من الكود الأصلي الخاص بك كما هي دون تعديل.


def _base_styles() -> str:
    return """
    <style>
      :root {
        --bg: #0b1020;
        --panel: #131a2b;
        --panel-2: #1a2338;
        --text: #edf2ff;
        --muted: #98a3bd;
        --line: #28324a;
        --good: #37d67a;
        --bad: #ff6b6b;
        --warn: #ffb84d;
        --accent: #4db5ff;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        padding: 24px;
        background: linear-gradient(180deg, #09101d 0%, #0f1728 100%);
        color: var(--text);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }
      .wrap {
        max-width: 1200px;
        margin: 0 auto;
      }
      .hero {
        background: linear-gradient(135deg, rgba(77,181,255,0.18), rgba(55,214,122,0.10));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 18px;
      }
      h1, h2, h3 {
        margin: 0 0 10px 0;
      }
      p {
        margin: 0;
        color: var(--muted);
      }
      .grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin: 18px 0;
      }
      .card {
        background: rgba(19,26,43,0.92);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 16px;
      }
      .metric-label {
        font-size: 12px;
        color: var(--muted);
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
      }
      .metric-value {
        font-size: 28px;
        font-weight: 700;
      }
      .good { color: var(--good); }
      .bad { color: var(--bad); }
      .warn { color: var(--warn); }
      .accent { color: var(--accent); }
      .section {
        background: rgba(19,26,43,0.92);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 16px;
      }
      .two-col {
        display: grid;
        grid-template-columns: 1.1fr 0.9fr;
        gap: 16px;
      }
      .table {
        width: 100%;
        border-collapse: collapse;
      }
      .table th,
      .table td {
        border-bottom: 1px solid var(--line);
        padding: 10px 8px;
        text-align: left;
        font-size: 13px;
      }
      .table th {
        color: var(--muted);
        font-weight: 600;
      }
      .kpi-strip {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 10px;
      }
      .pill {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.06);
        color: var(--text);
        font-size: 12px;
      }
      .muted { color: var(--muted); }
      .raw-html .t-row,
      .raw-html .feat-bar,
      .raw-html .sbar,
      .raw-html .class-card {
        margin-bottom: 10px;
      }
      .raw-html .t-row {
        display: grid;
        grid-template-columns: 48px repeat(9, minmax(0, 1fr));
        gap: 10px;
        align-items: center;
        border-bottom: 1px solid var(--line);
        padding: 10px 0;
        font-size: 13px;
      }
      .raw-html .feat-track,
      .raw-html .sbar-track {
        width: 100%;
        height: 10px;
        background: rgba(255,255,255,0.08);
        border-radius: 999px;
        overflow: hidden;
      }
      .raw-html .feat-fill,
      .raw-html .sbar-fill {
        height: 100%;
        border-radius: 999px;
      }
      .raw-html .feat-head,
      .raw-html .sbar-head {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 6px;
        font-size: 13px;
      }
      .footer {
        margin-top: 18px;
        font-size: 12px;
        color: var(--muted);
      }
      @media (max-width: 900px) {
        body { padding: 14px; }
        .two-col { grid-template-columns: 1fr; }
        .raw-html .t-row {
          grid-template-columns: repeat(2, minmax(0, 1fr));
        }
      }
    </style>
    """


def _training_html_template(
    *,
    train_acc,
    val_acc,
    n_samples,
    n_train,
    n_val,
    n_features,
    gap,
    pbo,
    avg_wf,
    avg_sh,
    feat_bars_html,
    class_html,
    wf_rows_html,
    chart_data,
) -> str:
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>V19 Training Report</title>
    {_base_styles()}
  </head>
  <body>
    <div class="wrap">
      <div class="hero">
        <h1>V19 Training Report</h1>
        <p>Summary of training, validation, feature importance, and walk-forward diagnostics.</p>
      </div>

      <div class="grid">
        <div class="card"><div class="metric-label">Train Accuracy</div><div class="metric-value accent">{_safe_float(train_acc) * 100:.2f}%</div></div>
        <div class="card"><div class="metric-label">Validation Accuracy</div><div class="metric-value {'good' if _safe_float(val_acc) >= 0.6 else 'warn' if _safe_float(val_acc) >= 0.5 else 'bad'}">{_safe_float(val_acc) * 100:.2f}%</div></div>
        <div class="card"><div class="metric-label">Generalization Gap</div><div class="metric-value {'bad' if _safe_float(gap) > 0.05 else 'good'}">{_safe_float(gap) * 100:.2f}%</div></div>
        <div class="card"><div class="metric-label">Walk-Forward Mean</div><div class="metric-value">{_safe_float(avg_wf) * 100:.2f}%</div></div>
        <div class="card"><div class="metric-label">Walk-Forward Sharpe</div><div class="metric-value">{_safe_float(avg_sh):.3f}</div></div>
        <div class="card"><div class="metric-label">PBO</div><div class="metric-value">{_safe_float(pbo):.3f}</div></div>
      </div>

      <div class="kpi-strip">
        <div class="pill">Samples: {_safe_int(n_samples):,}</div>
        <div class="pill">Train: {_safe_int(n_train):,}</div>
        <div class="pill">Validation: {_safe_int(n_val):,}</div>
        <div class="pill">Features: {_safe_int(n_features):,}</div>
      </div>

      <div class="two-col" style="margin-top:16px;">
        <div class="section raw-html">
          <h2>Feature Importance</h2>
          {feat_bars_html}
        </div>
        <div class="section raw-html">
          <h2>Class Metrics</h2>
          {class_html}
        </div>
      </div>

      <div class="section">
        <h2>Walk-Forward Folds</h2>
        <table class="table">
          <thead>
            <tr>
              <th>Fold</th>
              <th>Train Start</th>
              <th>Train End</th>
              <th>Test Start</th>
              <th>Test End</th>
              <th>Train Size</th>
              <th>Test Size</th>
              <th>Accuracy</th>
              <th>Sharpe</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>{wf_rows_html}</tbody>
        </table>
      </div>

      <div class="section">
        <h2>Raw Chart Payload</h2>
        <p>This JSON block is kept for downstream visualization/debugging.</p>
        <pre>{html_lib.escape(chart_data)}</pre>
      </div>

      <div class="footer">Generated by QuantSystem V19.</div>
    </div>
  </body>
</html>"""


def _backtest_html_template(
    *,
    T,
    wins,
    loses,
    tos,
    wr,
    aw,
    al,
    rr,
    pnl,
    mdd,
    sh,
    equity_start,
    equity_end,
    tp_avg,
    sl_avg,
    dur_avg,
    max_win,
    max_los,
    dominant,
    verdict,
    verdict_sub,
    style_html,
    trade_rows_html,
    regime_rows,
    chart_data,
    n_test_bars,
    max_tpsl,
) -> str:
    pnl_class = 'good' if _safe_float(pnl) >= 0 else 'bad'
    sharpe_class = 'good' if _safe_float(sh) > 0 else 'bad'
    rr_class = 'good' if _safe_float(rr) >= 1 else 'warn'
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>V19 Backtest Report</title>
    {_base_styles()}
  </head>
  <body>
    <div class="wrap">
      <div class="hero">
        <h1>V19 Backtest Report</h1>
        <p>{html_lib.escape(str(verdict))}</p>
        <p style="margin-top:8px;">{html_lib.escape(str(verdict_sub))}</p>
      </div>

      <div class="grid">
        <div class="card"><div class="metric-label">Trades</div><div class="metric-value">{_safe_int(T):,}</div></div>
        <div class="card"><div class="metric-label">Win Rate</div><div class="metric-value {'good' if _safe_float(wr) >= 0.5 else 'bad'}">{_safe_float(wr) * 100:.2f}%</div></div>
        <div class="card"><div class="metric-label">Total PnL</div><div class="metric-value {pnl_class}">${_safe_float(pnl):,.2f}</div></div>
        <div class="card"><div class="metric-label">Max Drawdown</div><div class="metric-value {'bad' if _safe_float(mdd) > 0 else 'good'}">${_safe_float(mdd):,.2f}</div></div>
        <div class="card"><div class="metric-label">Sharpe</div><div class="metric-value {sharpe_class}">{_safe_float(sh):.3f}</div></div>
        <div class="card"><div class="metric-label">R/R</div><div class="metric-value {rr_class}">{_safe_float(rr):.2f}</div></div>
      </div>

      <div class="kpi-strip">
        <div class="pill">Wins: {_safe_int(wins)}</div>
        <div class="pill">Losses: {_safe_int(loses)}</div>
        <div class="pill">Timeouts: {_safe_int(tos)}</div>
        <div class="pill">Test Bars: {_safe_int(n_test_bars):,}</div>
        <div class="pill">Equity Start: ${_safe_float(equity_start):,.2f}</div>
        <div class="pill">Equity End: ${_safe_float(equity_end):,.2f}</div>
      </div>

      <div class="two-col" style="margin-top:16px;">
        <div class="section">
          <h2>Trade Profile</h2>
          <table class="table">
            <tbody>
              <tr><th>Average Win (pips)</th><td class="good">{_safe_float(aw):+.2f}</td></tr>
              <tr><th>Average Loss (pips)</th><td class="bad">{_safe_float(al):+.2f}</td></tr>
              <tr><th>Average TP</th><td>{_safe_float(tp_avg):.2f}</td></tr>
              <tr><th>Average SL</th><td>{_safe_float(sl_avg):.2f}</td></tr>
              <tr><th>Average Duration (min)</th><td>{_safe_float(dur_avg):.2f}</td></tr>
              <tr><th>Max Win (pips)</th><td class="good">{_safe_float(max_win):+.2f}</td></tr>
              <tr><th>Max Loss (pips)</th><td class="bad">{_safe_float(max_los):+.2f}</td></tr>
              <tr><th>Dominant Style</th><td>{html_lib.escape(str(dominant))}</td></tr>
              <tr><th>TP/SL Scale Anchor</th><td>{_safe_float(max_tpsl):.2f}</td></tr>
            </tbody>
          </table>
        </div>

        <div class="section raw-html">
          <h2>Style Mix</h2>
          {style_html}
        </div>
      </div>

      <div class="section">
        <h2>Regime Breakdown</h2>
        <table class="table">
          <thead>
            <tr>
              <th>Regime</th>
              <th>Trades</th>
              <th>Win Rate</th>
              <th>Avg Pips</th>
              <th>Avg Duration</th>
              <th>Total PnL</th>
            </tr>
          </thead>
          <tbody>{regime_rows}</tbody>
        </table>
      </div>

      <div class="section raw-html">
        <h2>Recent Trades</h2>
        {trade_rows_html}
      </div>

      <div class="section">
        <h2>Raw Chart Payload</h2>
        <p>This JSON block is kept for downstream visualization/debugging.</p>
        <pre>{html_lib.escape(chart_data)}</pre>
      </div>

      <div class="footer">Generated by QuantSystem V19.</div>
    </div>
  </body>
</html>"""
