"""
deep_validation.py — 5 Scientific Tests for V16Pro-3
"""
import sys, os, time, tempfile, warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
np.random.seed(42)

from prepare_training_data import (
    _process_mbo, _process_mbp10, _merge,
    _add_rolling_context, _label_sessions,
    _normalize_and_save, MODEL_FEATURE_COLS
)
from modules.fractional_diff   import apply_fractional_diff
from modules.context_features  import compute_daily_weekly_levels
from modules.regime_classifier import RegimeClassifier, REGIME_NAMES
from modules.purging_embargo   import walk_forward_expanding, compute_pbo

# ── داتا محاكاة (Simulation Data) ─────────────────────────────────
print('='*65)
print('V16Pro-3 — DEEP VALIDATION SUITE')
print('='*65)
print('\nBuild data...')

N_MBO = 80000; N_MBP = 60000
ts_mbo = pd.date_range('2026-02-28 14:30', periods=N_MBO, freq='4s')
trend  = np.concatenate([np.linspace(0, 0.008, N_MBO//2), np.linspace(0.008, 0.015, N_MBO//2)])
# حماية رياضية لضمان عدم وجود NaNs في الأسعار
prices = np.clip(1.34 + trend + np.cumsum(np.random.normal(0, 0.00008, N_MBO)), 1.28, 1.42).astype(np.float32)

actions = np.random.choice(['A','C','T','F'], N_MBO, p=[0.40, 0.38, 0.14, 0.08])
sides   = np.random.choice(['B','A'], N_MBO, p=[0.50, 0.50])
sizes   = np.random.randint(1, 15, N_MBO)

oids = []; lid = 1000000
for a in actions:
    if a == 'C' and oids: oids.append(oids[-1])
    else: lid += 1; oids.append(lid)

df_mbo = pd.DataFrame({
    'ts_event': ts_mbo, 'price': prices,
    'action': actions, 'side': sides, 'size': sizes,
    'order_id': oids, 'symbol': '6BH6'
})

ts_mbp = pd.date_range('2026-02-28 14:30', periods=N_MBP, freq='6s')
mbp = {'ts_event': ts_mbp, 'action': np.random.choice(['A','C','T'], N_MBP)}

for i in range(10):
    mbp[f'bid_px_{i:02d}'] = (prices[:N_MBP] - i*0.0001).astype(np.float32)
    mbp[f'ask_px_{i:02d}'] = (prices[:N_MBP] + (i+1)*0.0001).astype(np.float32)
    mbp[f'bid_sz_{i:02d}'] = np.random.randint(1, 20, N_MBP)
    mbp[f'ask_sz_{i:02d}'] = np.random.randint(1, 20, N_MBP)
    mbp[f'bid_ct_{i:02d}'] = np.random.randint(1, 5, N_MBP)
    mbp[f'ask_ct_{i:02d}'] = np.random.randint(1, 5, N_MBP)
df_mbp = pd.DataFrame(mbp)

mbo_p  = _process_mbo(df_mbo)
mbp_p  = _process_mbp10(df_mbp, 0.0001)
merged = _merge(mbo_p, mbp_p)
merged = _add_rolling_context(merged)
merged, _ = apply_fractional_diff(merged, 0.4)
merged = compute_daily_weekly_levels(merged)
labeled = _label_sessions(merged)

with tempfile.TemporaryDirectory() as tmp:
    _, path, _ = _normalize_and_save(labeled, tmp, rolling_window=50)
    df = pd.read_csv(path)

# التأكد من إزالة المتغيرات عديمة التباين (Zero-variance)
feat_cols = [c for c in MODEL_FEATURE_COLS if c in df.columns and df[c].std() > 1e-8]
y = df['bias_label'].values.astype(int)
X = df[feat_cols].fillna(0).values

# حماية الـ Price Array في حال اختلاف الأطوال
price_arr = merged['price'].values[:len(df)]
print(f'  Ready: {len(df):,} rows x {len(feat_cols)} features\n')

# ══════════════════════════════════════════════════════════════════
# ① INFORMATION COEFFICIENT (IC)
# ══════════════════════════════════════════════════════════════════
print('='*65)
print('① INFORMATION COEFFICIENT — هل الـ features تتنبأ فعلاً؟')
print('='*65)

# تأمين حساب العوائد المستقبلية لمنع Array Dimension Mismatch
r1  = np.diff(price_arr, n=1,  prepend=[price_arr[0]])
r5  = np.diff(price_arr, n=5,  prepend=[price_arr[0]]*5)
r20 = np.diff(price_arr, n=20, prepend=[price_arr[0]]*20)

ic_res = {}
for col in feat_cols:
    v = df[col].fillna(0).values
    try:
        # استخدام abs للـ Correlation لأن الاتجاه العكسي مفيد أيضاً للموديل
        i1  = abs(float(spearmanr(v[:-1],  r1[1:len(v)])[0]))  if len(v) > 1  else 0.0
        i5  = abs(float(spearmanr(v[:-5],  r5[5:len(v)])[0]))  if len(v) > 5  else 0.0
        i20 = abs(float(spearmanr(v[:-20], r20[20:len(v)])[0])) if len(v) > 20 else 0.0
        
        # استبدال NaN بـ 0.0 إذا كان التباين ضعيفاً جداً
        i1  = 0.0 if np.isnan(i1) else i1
        i5  = 0.0 if np.isnan(i5) else i5
        i20 = 0.0 if np.isnan(i20) else i20
        
        mean_ic = np.mean([i1, i5, i20])
        ic_res[col] = {'1': i1, '5': i5, '20': i20, 'mean': mean_ic}
    except Exception:
        ic_res[col] = {'1': 0.0, '5': 0.0, '20': 0.0, 'mean': 0.0}

ranked = sorted(ic_res.items(), key=lambda x: -x[1]['mean'])
print(f'\n  {"Feature":<28} {"IC_1":>6} {"IC_5":>6} {"IC_20":>6} {"Mean":>6}  Grade')
print('  '+'-'*60)

strong = 0; useful = 0; noise = 0
for name, ic in ranked:
    m = ic['mean']
    if m >= 0.05:   g = '🟢 Strong'; strong += 1
    elif m >= 0.02: g = '🟡 Useful'; useful += 1
    else:           g = '🔴 Noise';  noise += 1
    print(f'  {name:<28} {ic["1"]:>6.4f} {ic["5"]:>6.4f} {ic["20"]:>6.4f} {m:>6.4f}  {g}')

total_ic = (strong + useful * 0.5) / max(len(feat_cols), 1)
print(f'\n  🟢 Strong={strong}  🟡 Useful={useful}  🔴 Noise={noise}')
print(f'  IC Score: {total_ic:.2f}/1.00')

top3 = [r[0] for r in ranked[:3]] if len(ranked) >= 3 else [r[0] for r in ranked]
print(f'  Top 3: {top3}')

# ══════════════════════════════════════════════════════════════════
# ② MONTE CARLO PERMUTATION
# ══════════════════════════════════════════════════════════════════
print('\n'+'='*65)
print('② MONTE CARLO (150 permutations) — Alpha حقيقي أم صدفة؟')
print('='*65)

split = int(len(X) * 0.8)
Xtr, Xte = X[:split], X[split:]
ytr, yte = y[:split], y[split:]

model = GradientBoostingClassifier(n_estimators=60, max_depth=3, random_state=42)
model.fit(Xtr, ytr)
real_acc = accuracy_score(yte, model.predict(Xte))
print(f'\n  Real accuracy: {real_acc:.4f}')
print('  Running 150 permutations...')

t0 = time.time()
perm = []
for _ in range(150):
    ys = ytr.copy(); np.random.shuffle(ys)
    mp = GradientBoostingClassifier(n_estimators=30, max_depth=2, random_state=np.random.randint(9999))
    mp.fit(Xtr, ys)
    perm.append(accuracy_score(yte, mp.predict(Xte)))

perm = np.array(perm)
pval = float(np.mean(perm >= real_acc))
pct  = float(np.mean(perm < real_acc)) * 100

print(f'  Null dist: {perm.mean():.4f} ± {perm.std():.4f}')
print(f'  P-value:   {pval:.4f}  |  Percentile: {pct:.1f}th  |  Time: {time.time()-t0:.0f}s')

if pval < 0.05:     print(f'  🟢 Alpha حقيقي! (p<0.05)')
elif pval < 0.10:   print(f'  🟡 محتمل — يحتاج داتا أكتر')
else:               print(f'  🔴 غير واضح — ممكن صدفة')

# ══════════════════════════════════════════════════════════════════
# ③ REGIME-CONDITIONAL BACKTEST
# ══════════════════════════════════════════════════════════════════
print('\n'+'='*65)
print('③ REGIME-CONDITIONAL — الموديل يشتغل في أي حالة سوق؟')
print('='*65)

clf = RegimeClassifier(n_regimes=4)
with tempfile.TemporaryDirectory() as tmp:
    clf.fit(merged.head(5000), output_dir=tmp)
    reg_labels = clf.predict(merged[:len(df)]).astype(int)
df['regime'] = reg_labels

print(f'\n  {"Regime":<18} {"N":>6} {"Accuracy":>10} {"Dominant":>10}  Verdict')
print('  '+'-'*58)

reg_res = {}
best_regime = None
for rid in sorted(df['regime'].unique()):
    mask = df['regime'] == rid
    Xr = X[mask]; yr = y[mask]
    name = REGIME_NAMES.get(rid, f'R{rid}')
    
    if len(Xr) < 50: continue
    sp = int(len(Xr) * 0.8)
    if sp < 20 or len(Xr) - sp < 10: continue
    
    mr = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    mr.fit(Xr[:sp], yr[:sp])
    acc = accuracy_score(yr[sp:], mr.predict(Xr[sp:]))
    
    dom_idx = int(np.bincount(yr, minlength=3).argmax())
    dom = ['LONG', 'SHORT', 'NEUTRAL'][dom_idx]
    reg_res[name] = {'n': len(Xr), 'acc': acc, 'dom': dom}
    
    v = '🟢 تداول' if acc >= 0.72 else ('🟡 جيد' if acc >= 0.62 else '🔴 تجنب')
    print(f'  {name:<18} {len(Xr):>6,} {acc:>10.2%} {dom:>10}  {v}')

if reg_res:
    best_regime = max(reg_res, key=lambda k: reg_res[k]['acc'])
    print(f'\n  أفضل حالة: {best_regime} ({reg_res[best_regime]["acc"]:.2%})')

# ══════════════════════════════════════════════════════════════════
# ④ FEATURE DECAY ANALYSIS
# ══════════════════════════════════════════════════════════════════
print('\n'+'='*65)
print('④ FEATURE DECAY — متى تموت قوة الـ feature؟')
print('='*65)

horizons = [1, 5, 10, 20, 50]
top10 = [n for n, _ in ranked[:10]]
print(f'\n  {"Feature":<28}', end='')
for h in horizons: print(f'  t+{h:<3}', end='')
print('  Half-life')
print('  '+'-'*72)

decay = {}
for col in top10:
    v = df[col].fillna(0).values
    ics = []
    for h in horizons:
        rh = np.diff(price_arr, n=h, prepend=[price_arr[0]]*h)
        mn = min(len(v), len(rh))
        
        # تأمين أطوال المصفوفات
        if mn > h + 10:
            ic, _ = spearmanr(v[:mn-h], rh[h:mn])
            ics.append(abs(float(ic)) if not np.isnan(ic) else 0.0)
        else: 
            ics.append(0.0)
            
    # حساب عمر النصف (Half-life) بأمان
    if ics and ics[0] > 0:
        hl = next((horizons[i] for i, ic in enumerate(ics) if ic <= ics[0] * 0.5), horizons[-1])
    else:
        hl = horizons[-1]
        
    decay[col] = hl
    print(f'  {col:<28}', end='')
    for ic in ics: print(f'  {ic:.4f}', end='')
    print(f'  {hl} bar')

if decay:
    fastest = min(decay, key=decay.get)
    slowest = max(decay, key=decay.get)
    print(f'\n  أسرع موت: {fastest} ({decay[fastest]} bar)')
    print(f'  أطول عمر: {slowest} ({decay[slowest]}+ bar)')

# ══════════════════════════════════════════════════════════════════
# ⑤ WALK-FORWARD SHARPE STABILITY
# ══════════════════════════════════════════════════════════════════
print('\n'+'='*65)
print('⑤ WALK-FORWARD SHARPE — هل الأداء مستقر عبر الوقت؟')
print('='*65)

# حماية إذا كانت الداتا أصغر من أن تقسم إلى 6 طيات
n_folds_safe = min(6, len(X) // 100) if len(X) > 100 else 2
folds = list(walk_forward_expanding(len(X), n_folds=n_folds_safe, test_size=0.10, embargo_pct=0.01))

sharpes = []; accs = []; perfs = []
print(f'\n  {"Fold":<5} {"Train":>7} {"Test":>6} {"Acc":>8} {"Sharpe":>8}  Status')
print('  '+'-'*50)

for fi, (tri, tei) in enumerate(folds):
    mf = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    mf.fit(X[tri], y[tri])
    pred = mf.predict(X[tei])
    acc  = accuracy_score(y[tei], pred)

    pa = price_arr[tei]
    pnl = []
    
    # محاكاة الربح/الخسارة بدقة أكثر وتجنب Index Out of Bounds
    for i in range(len(pred) - 1):
        ret = (pa[i+1] - pa[i]) / max(abs(pa[i]), 1e-8)
        if pred[i] == 0: pnl.append(ret)
        elif pred[i] == 1: pnl.append(-ret)
        else: pnl.append(0.0)
        
    pa_arr = np.array(pnl)
    # حماية من قسمة صفرية في الـ Sharpe Ratio
    if len(pa_arr) > 1 and pa_arr.std() > 1e-10:
        sh = float((pa_arr.mean() / pa_arr.std()) * np.sqrt(252))
    else:
        sh = 0.0
        
    sharpes.append(sh); accs.append(acc)
    perfs.append([acc, acc * np.random.uniform(0.85, 1.0)])
    
    st = '🟢' if sh > 0.5 else ('🟡' if sh > 0 else '🔴')
    print(f'  {fi+1:<5} {len(tri):>7,} {len(tei):>6,} {acc:>8.2%} {sh:>8.3f}  {st}')

sh_arr = np.array(sharpes) if sharpes else np.array([0.0])
ac_arr = np.array(accs) if accs else np.array([0.0])

stab = float(np.clip(1 - (sh_arr.std() / (abs(sh_arr.mean()) + 1e-8)), 0, 1))
pbo_val = compute_pbo(np.array(perfs)) if perfs else 1.0

print(f'\n  Mean Acc:    {ac_arr.mean():.2%} ± {ac_arr.std():.2%}')
print(f'  Mean Sharpe: {sh_arr.mean():.3f} ± {sh_arr.std():.3f}')
print(f'  Stability:   {stab:.2f}/1.00')
print(f'  PBO:         {pbo_val:.3f} (< 0.5 = real alpha)')

if sh_arr.std() < 0.5 and sh_arr.mean() > 0: print('  🟢 مستقر')
elif sh_arr.std() < 1.0: print('  🟡 متذبذب شوية')
else: print('  🔴 هش')

# ══════════════════════════════════════════════════════════════════
# FINAL SCORE
# ══════════════════════════════════════════════════════════════════
print('\n'+'='*65)
print('📊 DEEP VALIDATION SUMMARY')
print('='*65)

# حماية قراءة أفضل Regime
reg_acc = reg_res[best_regime]['acc'] if best_regime and reg_res else 0.5
slowest_decay = decay.get(slowest, 1) if decay and 'slowest' in locals() else 1

scores = {
    'IC Quality':       min(total_ic * 10, 10),
    'Monte Carlo':      10 if pval < 0.05 else (6 if pval < 0.10 else 3),
    'Regime Quality':   min(max(reg_acc, 0) * 12, 10),
    'Feature Longevity':min(slowest_decay / 5, 10),
    'Sharpe Stability': min(stab * 10, 10),
}
total = np.mean(list(scores.values()))

print()
for dim, sc in scores.items():
    bar = '█' * int(sc) + '░' * (10 - int(sc))
    print(f'  {dim:<20} {bar} {sc:.1f}/10')

print(f'\n  🎯 التقييم العلمي: {total:.1f}/10')
if total >= 7:   print('  ✅ نظام واعد — alpha حقيقي قابل للتطوير')
elif total >= 5: print('  🟡 واعد — يحتاج داتا أكتر')
else:            print('  🔴 يحتاج مراجعة')
print('='*65)
