import os, json, math
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
    backtest_summary: dict | None = None,
    visual_diagnostics: dict | None = None,
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

    summary_cards_html = _render_backtest_summary_cards(
        backtest_summary or {},
        n_dataset=n_dataset,
        n_test_bars=n_test_bars,
        n_features=n_features,
    )
    visual_diag_html = _render_visual_diagnostics_html(visual_diagnostics or {})
    
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
        max_tpsl=max_tpsl, # تمرير القيمة المحمية
        summary_cards_html=summary_cards_html,
        visual_diag_html=visual_diag_html,
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

def _metric_card(label: str, value: str, tone: str = '') -> str:
    tone_cls = f' {tone}' if tone else ''
    return (
        f'<div class="card{tone_cls}">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        '</div>'
    )


def _render_backtest_summary_cards(summary: dict, *, n_dataset: int, n_test_bars: int, n_features: int) -> str:
    cards = [
        _metric_card('Rows', f'{_safe_int(n_dataset):,}'),
        _metric_card('Predictions', f'{_safe_int(summary.get("predictions", n_test_bars)):,}'),
        _metric_card('Trades', f'{_safe_int(summary.get("trades", 0)):,}'),
        _metric_card('Win Rate', f'{_safe_float(summary.get("win_rate", 0.0)):.1%}',
                     'good' if _safe_float(summary.get('win_rate', 0.0)) >= 0.5 else 'bad'),
        _metric_card('Profit Factor', f'{_safe_float(summary.get("profit_factor", 0.0)):.2f}',
                     'good' if _safe_float(summary.get('profit_factor', 0.0)) >= 1.0 else 'bad'),
        _metric_card('Sharpe', f'{_safe_float(summary.get("trade_sharpe", 0.0)):.2f}',
                     'good' if _safe_float(summary.get('trade_sharpe', 0.0)) >= 0 else 'bad'),
        _metric_card('Macro F1', f'{_safe_float(summary.get("directional_f1_macro", 0.0)):.4f}',
                     'good' if _safe_float(summary.get('directional_f1_macro', 0.0)) >= 0.38 else 'bad'),
        _metric_card('Visual Coverage', f'{_safe_float(summary.get("visual_coverage", 0.0)):.1%}',
                     'good' if _safe_float(summary.get('visual_coverage', 0.0)) >= 0.5 else 'bad'),
        _metric_card('Features', f'{_safe_int(n_features):,}'),
    ]
    return ''.join(cards)


def _render_visual_diagnostics_html(diag: dict) -> str:
    if not diag:
        return '<div class="empty-note">لا توجد diagnostics بصرية مرفقة لهذا التشغيل.</div>'

    training_ref = diag.get('training_reference') or {}
    rows_with_tensor = _safe_int(diag.get('rows_with_tensor', 0))
    used_tensors = _safe_int(diag.get('used_tensor_count', 0))
    notes = diag.get('diagnosis_notes') or []
    notes_html = ''.join(f'<li>{note}</li>' for note in notes)

    rows_html = ''.join([
        f'<tr><td>Full Rows</td><td>{_safe_int(diag.get("rows_total", 0)):,}</td><td>{_safe_int(diag.get("rows_with_visual", 0)):,}</td><td>{_safe_float(diag.get("coverage_ratio", 0.0)):.1%}</td></tr>',
        f'<tr><td>Event Rows</td><td>{_safe_int(diag.get("event_rows", 0)):,}</td><td>{_safe_int(diag.get("event_rows_with_visual", 0)):,}</td><td>{_safe_float(diag.get("event_coverage_ratio", 0.0)):.1%}</td></tr>',
        f'<tr><td>Train-Event Rows</td><td>{_safe_int(diag.get("train_event_rows", 0)):,}</td><td>{_safe_int(diag.get("train_event_rows_with_visual", 0)):,}</td><td>{_safe_float(diag.get("train_event_coverage_ratio", 0.0)):.1%}</td></tr>',
        f'<tr><td>Directional Train-Event Rows</td><td>{_safe_int(diag.get("directional_train_event_rows", 0)):,}</td><td>{_safe_int(diag.get("directional_train_event_rows_with_visual", 0)):,}</td><td>{_safe_float(diag.get("directional_train_event_coverage_ratio", 0.0)):.1%}</td></tr>',
        f'<tr><td>Tradeable Rows</td><td>{_safe_int(diag.get("tradeable_rows", 0)):,}</td><td>{_safe_int(diag.get("tradeable_rows_with_visual", 0)):,}</td><td>{_safe_float(diag.get("tradeable_coverage_ratio", 0.0)):.1%}</td></tr>',
        f'<tr><td>Executed Rows</td><td>{_safe_int(diag.get("executed_rows", 0)):,}</td><td>{_safe_int(diag.get("executed_rows_with_visual", 0)):,}</td><td>{_safe_float(diag.get("executed_coverage_ratio", 0.0)):.1%}</td></tr>',
    ])

    training_html = ''
    if training_ref:
        train_ref_rows = f'{_safe_int(training_ref.get("rows_total", 0)):,}'
        train_ref_cov = f'{_safe_float(training_ref.get("coverage_ratio", 0.0)):.1%}'
        train_ref_tensors = f'{_safe_int(training_ref.get("n_tensors", 0)):,}'
        coverage_gap = _safe_float(diag.get('coverage_gap_vs_train', 0.0))
        training_html = (
            '<div class="subgrid">'
            f'{_metric_card("Train Ref Rows", train_ref_rows)}'
            f'{_metric_card("Train Ref Coverage", train_ref_cov)}'
            f'{_metric_card("Train Ref Tensors", train_ref_tensors)}'
            f'{_metric_card("Coverage Gap", f"{coverage_gap:+.1%}", "bad" if coverage_gap < 0 else "good")}'
            '</div>'
        )

    return (
        '<div class="diag-wrap">'
        '<div class="subgrid">'
        f'{_metric_card("Visual Source", str(diag.get("source", "n/a")))}'
        f'{_metric_card("Reason", str(diag.get("reason", "n/a")))}'
        f'{_metric_card("Rows With Tensor", f"{rows_with_tensor:,}")}'
        f'{_metric_card("Used Tensors", f"{used_tensors:,}")}'
        '</div>'
        f'{training_html}'
        '<table class="diag-table">'
        '<thead><tr><th>Slice</th><th>Rows</th><th>Covered</th><th>Coverage</th></tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        '</table>'
        '<div class="notes-box"><div class="notes-title">Diagnosis</div><ul>'
        f'{notes_html}'
        '</ul></div>'
        '</div>'
    )


def _base_styles() -> str:
    return """
<style>
  :root {
    --bg: #0b1320;
    --panel: #121c2d;
    --panel-2: #0f1726;
    --fg: #f4f7fb;
    --mt: #8ea1b8;
    --line: rgba(255,255,255,0.08);
    --g: #2bd67b;
    --r: #ff6b6b;
    --a: #52c7ff;
    --w: #ffbf4d;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: "Segoe UI", Arial, sans-serif;
    background: linear-gradient(180deg, #09111d 0%, #0b1320 100%);
    color: var(--fg);
  }
  .page {
    width: min(1180px, calc(100vw - 32px));
    margin: 24px auto 40px;
  }
  .hero, .panel {
    background: rgba(18, 28, 45, 0.96);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 20px 22px;
    box-shadow: 0 14px 32px rgba(0,0,0,0.24);
    margin-bottom: 18px;
  }
  .hero h1, .panel h2 {
    margin: 0 0 10px;
    font-size: 22px;
  }
  .hero p, .muted, .label {
    color: var(--mt);
  }
  .grid, .subgrid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
  }
  .card {
    background: var(--panel-2);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 14px;
  }
  .card.good .value { color: var(--g); }
  .card.bad .value { color: var(--r); }
  .value {
    font-size: 22px;
    font-weight: 700;
    margin-top: 6px;
  }
  .kpis {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 12px;
  }
  .section-title {
    margin: 0 0 12px;
    font-size: 16px;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    color: var(--mt);
  }
  table {
    width: 100%;
    border-collapse: collapse;
  }
  th, td {
    padding: 10px 12px;
    border-bottom: 1px solid var(--line);
    text-align: left;
    font-size: 14px;
  }
  th { color: var(--mt); }
  .diag-table td:last-child, .diag-table th:last-child {
    text-align: right;
  }
  .notes-box {
    margin-top: 14px;
    background: rgba(82, 199, 255, 0.06);
    border: 1px solid rgba(82, 199, 255, 0.16);
    border-radius: 14px;
    padding: 14px 16px;
  }
  .notes-title {
    font-weight: 700;
    margin-bottom: 8px;
  }
  .notes-box ul {
    margin: 0;
    padding-left: 18px;
  }
  .sbar, .feat-bar {
    margin-bottom: 10px;
  }
  .sbar-head, .feat-head {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 6px;
    font-size: 13px;
  }
  .sbar-track, .feat-track {
    width: 100%;
    background: rgba(255,255,255,0.05);
    border-radius: 999px;
    overflow: hidden;
    height: 10px;
  }
  .sbar-fill, .feat-fill {
    height: 100%;
    border-radius: 999px;
  }
  .trade-grid {
    display: grid;
    gap: 8px;
  }
  .t-row {
    display: grid;
    grid-template-columns: 44px repeat(9, minmax(0, 1fr));
    gap: 10px;
    align-items: center;
    padding: 10px 12px;
    background: var(--panel-2);
    border: 1px solid var(--line);
    border-radius: 12px;
    font-size: 13px;
  }
  .win { color: var(--g); }
  .lose { color: var(--r); }
  .long { color: var(--a); font-weight: 700; }
  .short { color: #ff9b66; font-weight: 700; }
  .to-c { color: var(--w); }
  .empty-note {
    color: var(--mt);
    padding: 14px;
    background: var(--panel-2);
    border: 1px dashed var(--line);
    border-radius: 12px;
  }
  details {
    margin-top: 14px;
  }
  pre {
    white-space: pre-wrap;
    word-break: break-word;
    background: #08101b;
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 12px;
    color: #d9e3ee;
    font-size: 12px;
  }
</style>
"""


def _training_html_template(
    *,
    train_acc: float,
    val_acc: float,
    n_samples: int,
    n_train: int,
    n_val: int,
    n_features: int,
    gap: float,
    pbo: float,
    avg_wf: float,
    avg_sh: float,
    feat_bars_html: str,
    class_html: str,
    wf_rows_html: str,
    chart_data: str,
) -> str:
    cards = ''.join([
        _metric_card('Train Acc', f'{_safe_float(train_acc):.1%}', 'good'),
        _metric_card('Val Acc', f'{_safe_float(val_acc):.1%}',
                     'good' if _safe_float(val_acc) >= 0.5 else 'bad'),
        _metric_card('Generalization Gap', f'{_safe_float(gap):+.1%}',
                     'bad' if _safe_float(gap) > 0.05 else 'good'),
        _metric_card('Samples', f'{_safe_int(n_samples):,}'),
        _metric_card('Features', f'{_safe_int(n_features):,}'),
        _metric_card('Avg WF Acc', f'{_safe_float(avg_wf):.1%}',
                     'good' if _safe_float(avg_wf) >= 0.55 else 'bad'),
        _metric_card('Avg WF Sharpe', f'{_safe_float(avg_sh):.2f}',
                     'good' if _safe_float(avg_sh) >= 0 else 'bad'),
        _metric_card('PBO', f'{_safe_float(pbo):.3f}',
                     'bad' if _safe_float(pbo) > 0.2 else 'good'),
    ])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>V19 Training Report</title>
  {_base_styles()}
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>V19 Training Report</h1>
      <p>عينات التدريب: {n_train:,} | عينات التحقق: {n_val:,} | إجمالي العينات: {n_samples:,}</p>
    </section>
    <section class="panel">
      <div class="section-title">Overview</div>
      <div class="kpis">{cards}</div>
    </section>
    <section class="panel">
      <div class="section-title">Feature Importance</div>
      {feat_bars_html or '<div class="empty-note">لا توجد feature importances متاحة.</div>'}
    </section>
    <section class="panel">
      <div class="section-title">Class Metrics</div>
      <div class="grid">{class_html}</div>
    </section>
    <section class="panel">
      <div class="section-title">Walk-Forward Folds</div>
      <table><tbody>{wf_rows_html}</tbody></table>
      <details>
        <summary>Raw Chart Data</summary>
        <pre>{chart_data}</pre>
      </details>
    </section>
  </div>
</body>
</html>"""


def _backtest_html_template(
    *,
    T: int,
    wins: int,
    loses: int,
    tos: int,
    wr: float,
    aw: float,
    al: float,
    rr: float,
    pnl: float,
    mdd: float,
    sh: float,
    equity_start: float,
    equity_end: float,
    tp_avg: float,
    sl_avg: float,
    dur_avg: float,
    max_win: float,
    max_los: float,
    dominant: str,
    verdict: str,
    verdict_sub: str,
    style_html: str,
    trade_rows_html: str,
    regime_rows: str,
    chart_data: str,
    n_test_bars: int,
    max_tpsl: float,
    summary_cards_html: str = '',
    visual_diag_html: str = '',
) -> str:
    top_cards = ''.join([
        _metric_card('Trades', f'{_safe_int(T):,}'),
        _metric_card('Wins', f'{_safe_int(wins):,}', 'good'),
        _metric_card('Losses', f'{_safe_int(loses):,}', 'bad'),
        _metric_card('Timeouts', f'{_safe_int(tos):,}'),
        _metric_card('Win Rate', f'{_safe_float(wr):.1%}', 'good' if _safe_float(wr) >= 0.5 else 'bad'),
        _metric_card('Net PnL', f'${_safe_float(pnl):,.2f}', 'good' if _safe_float(pnl) >= 0 else 'bad'),
        _metric_card('Max Drawdown', f'${_safe_float(mdd):,.2f}', 'bad'),
        _metric_card('Trade Sharpe', f'{_safe_float(sh):.2f}', 'good' if _safe_float(sh) >= 0 else 'bad'),
    ])
    micro_cards = ''.join([
        _metric_card('Avg Win (pips)', f'{_safe_float(aw):+.2f}', 'good'),
        _metric_card('Avg Loss (pips)', f'{_safe_float(al):+.2f}', 'bad'),
        _metric_card('R:R', f'{_safe_float(rr):.2f}', 'good' if _safe_float(rr) >= 1 else 'bad'),
        _metric_card('Avg TP', f'{_safe_float(tp_avg):.1f}'),
        _metric_card('Avg SL', f'{_safe_float(sl_avg):.1f}'),
        _metric_card('Avg Duration', f'{_safe_float(dur_avg):.1f}m'),
        _metric_card('Best Trade', f'{_safe_float(max_win):+.1f} pips', 'good'),
        _metric_card('Worst Trade', f'-{_safe_float(max_los):.1f} pips', 'bad'),
        _metric_card('Equity', f'${_safe_float(equity_start):,.0f} -> ${_safe_float(equity_end):,.0f}'),
        _metric_card('Dominant Style', dominant),
        _metric_card('Test Bars', f'{_safe_int(n_test_bars):,}'),
        _metric_card('TP/SL Scale', f'{_safe_float(max_tpsl):.1f}'),
    ])
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>V19 Backtest Report</title>
  {_base_styles()}
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>V19 Backtest Report</h1>
      <p>{verdict}</p>
      <p class="muted">{verdict_sub}</p>
    </section>

    <section class="panel">
      <div class="section-title">Core Metrics</div>
      <div class="kpis">{top_cards}</div>
    </section>

    <section class="panel">
      <div class="section-title">Run Summary</div>
      <div class="kpis">{summary_cards_html}</div>
    </section>

    <section class="panel">
      <div class="section-title">Visual Coverage Diagnostics</div>
      {visual_diag_html}
    </section>

    <section class="panel">
      <div class="section-title">Trade Shape</div>
      <div class="kpis">{micro_cards}</div>
      <div style="margin-top:14px">{style_html or '<div class="empty-note">لا توجد أنماط صفقات كافية.</div>'}</div>
    </section>

    <section class="panel">
      <div class="section-title">Regime Breakdown</div>
      <table>
        <thead>
          <tr><th>Regime</th><th>Trades</th><th>Win Rate</th><th>Avg Pips</th><th>Avg Duration</th><th>Total PnL</th></tr>
        </thead>
        <tbody>{regime_rows or ''}</tbody>
      </table>
    </section>

    <section class="panel">
      <div class="section-title">Recent Trades</div>
      <div class="trade-grid">{trade_rows_html or '<div class="empty-note">لا توجد صفقات لعرضها.</div>'}</div>
      <details>
        <summary>Raw Chart Data</summary>
        <pre>{chart_data}</pre>
      </details>
    </section>
  </div>
</body>
</html>"""
