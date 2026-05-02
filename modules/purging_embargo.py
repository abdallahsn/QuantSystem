import numpy as np
import pandas as pd
from itertools import combinations
from typing import Iterator, List, Tuple
from scipy.stats import spearmanr, rankdata

# ══════════════════════════════════════════════════════════════════
# 1. Purging & Embargo
# ══════════════════════════════════════════════════════════════════

def purge_overlapping(train_idx: np.ndarray, test_idx: np.ndarray, t1: pd.Series, t0: pd.Series) -> np.ndarray:
    if len(test_idx) == 0 or len(train_idx) == 0:
        return train_idx

    t0_test_min = t0.iloc[test_idx].min()
    # الفلترة الصحيحة: أي صف في الـ Train يمتد وقته إلى ما بعد بداية الـ Test يجب حذفه
    overlap_mask = t1.iloc[train_idx] >= t0_test_min
    purged = train_idx[~overlap_mask.values]

    return purged

def embargo_observations(train_idx: np.ndarray, test_idx: np.ndarray, embargo_pct: float = 0.01) -> np.ndarray:
    if len(test_idx) == 0 or len(train_idx) == 0:
        return train_idx

    # In expanding splits train rows usually lie entirely before the test block, in
    # which case a post-test embargo is structurally inapplicable and should stay a
    # transparent no-op rather than pretending to purge anything.
    if np.max(train_idx) <= np.max(test_idx):
        return train_idx

    test_end = test_idx.max()
    n = int(max(np.max(train_idx), np.max(test_idx)) + 1)
    embargo_n = max(1, int(n * embargo_pct))

    embargo_start = test_end + 1
    embargo_end = embargo_start + embargo_n

    # استخدام numpy السريع بدل isin
    keep = train_idx[(train_idx < embargo_start) | (train_idx >= embargo_end)]
    return keep

# ══════════════════════════════════════════════════════════════════
# 2. Walk-Forward Expanding Window
# ══════════════════════════════════════════════════════════════════

def walk_forward_expanding(
    n_samples: int,
    n_folds: int = 6,
    test_size: float = 0.1,
    embargo_pct: float = 0.01,
    t0: pd.Series = None,
    t1: pd.Series = None,
    min_train_pct: float = 0.20,
    min_train_rows: int = 200,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    test_n = max(1, int(n_samples * test_size))
    min_train = max(int(n_samples * min_train_pct), int(min_train_rows))
    min_train = min(min_train, max(test_n + 50, n_samples - test_n))
    tail_n = max(0, n_samples - min_train)
    effective_folds = max(int(n_folds), int(np.ceil(tail_n / max(test_n, 1))))
    if tail_n <= 0:
        return

    tail_idx = np.arange(min_train, n_samples, dtype=np.int64)
    effective_folds = min(max(effective_folds, 1), len(tail_idx))
    test_blocks = [block for block in np.array_split(tail_idx, effective_folds) if len(block) > 0]

    for test_idx in test_blocks:
        test_idx = np.asarray(test_idx, dtype=np.int64)
        test_start = int(test_idx[0])
        all_before = np.arange(0, test_start, dtype=np.int64)

        if t0 is not None and t1 is not None:
            train_idx = purge_overlapping(all_before, test_idx, t1, t0)
        else:
            train_idx = all_before

        train_idx = embargo_observations(train_idx, test_idx, embargo_pct)

        if len(train_idx) < 50 or len(test_idx) < 10: continue

        yield train_idx, test_idx

# ══════════════════════════════════════════════════════════════════
# 3. Combinatorial Purged CV (CPCV)
# ══════════════════════════════════════════════════════════════════

def combinatorial_purged_cv(n_samples: int, n_groups: int = 6, n_test_groups: int = 2, embargo_pct: float = 0.01,
                            t0: pd.Series = None, t1: pd.Series = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    group_size = n_samples // n_groups
    groups = [np.arange(g * group_size, (g + 1) * group_size if g < n_groups - 1 else n_samples) for g in range(n_groups)]

    splits = []
    for test_groups in combinations(range(n_groups), n_test_groups):
        test_idx = np.concatenate([groups[g] for g in test_groups])
        # تصحيح الخطأ: جميع المجموعات التي ليست في Test هي Train
        train_idx_raw = np.concatenate([groups[g] for g in range(n_groups) if g not in test_groups])

        if t0 is not None and t1 is not None:
            train_idx = purge_overlapping(train_idx_raw, test_idx, t1, t0)
        else:
            train_idx = train_idx_raw

        # في הـ CPCV، الـ Embargo يجب أن يطبق بعد كل مقطع Test (لأن الـ Test قد يكون في المنتصف)
        # لتسهيل الأمر هنا نفترض الـ Embargo بعد أقصى نقطة في الـ Test (تحسين هندسي)
        train_idx = embargo_observations(train_idx, test_idx, embargo_pct)

        if len(train_idx) < 30 or len(test_idx) < 5: continue
        splits.append((train_idx, test_idx))

    return splits

# ══════════════════════════════════════════════════════════════════
# 4. Probability of Backtest Overfitting (PBO)
# ══════════════════════════════════════════════════════════════════

def compute_pbo(performances: np.ndarray) -> float:
    if len(performances) < 3: return 0.5

    train_scores = performances[:, 0]
    test_scores  = performances[:, 1]

    # تصحيح معادلة الـ PBO لتتوافق مع Bailey et al. (2015)
    # W_c = Logit(Rank_test / (N + 1))
    rank_test = rankdata(test_scores)
    n = len(rank_test)
    
    # حساب العلاقة بين أداء الـ Train والـ Test
    # لو الاستراتيجية اللي جابت أعلى Score في الـ Train جابت سكور قليل في الـ Test = Overfit
    best_train_idx = np.argmax(train_scores)
    
    # الـ W هو أداء أفضل نموذج Train في بيئة الـ Test مقارنة بباقي النماذج
    omega = rank_test[best_train_idx] / (n + 1)
    
    if omega <= 0 or omega >= 1:
        w_c = 0 
    else:
        w_c = np.log(omega / (1 - omega))

    # بما إننا نحسب لنموذج واحد هنا (وليس مصفوفة)، يمكننا محاكاة الـ Distribution
    # لكن الصيغة المبسطة: إذا كان W_c سالباً (يعني أداء أقل من المتوسط)، فالنموذج Overfitted
    pbo_estimate = 1.0 if w_c < 0 else 0.0

    return round(pbo_estimate, 4)

# ══════════════════════════════════════════════════════════════════
# 5. Feature Selection (Spearman & mRMR Vectorized)
# ══════════════════════════════════════════════════════════════════

def spearman_redundancy_filter(df_features: pd.DataFrame, target: pd.Series = None, threshold: float = 0.85, protected: set = None) -> List[str]:
    cols = df_features.columns.tolist()
    if len(cols) <= 1: return cols
    protected = protected or set()

    # 1. حساب مصفوفة الارتباط مرة واحدة (Vectorized) للسرعة
    # استخدام Pandas corr للتعامل مع الـ NaN بأمان
    corr_matrix = df_features.corr(method='spearman').abs()
    
    # 2. حساب الارتباط مع الهدف
    if target is not None:
        target_corr = df_features.apply(lambda x: abs(x.corr(target, method='spearman')))
    else:
        target_corr = pd.Series(1.0, index=cols)

    # 3. ترتيب הـ Features (الأقوى أولاً، والمحمية دائماً في القمة)
    sorted_cols = sorted(cols, key=lambda c: (c in protected, target_corr.get(c, 0)), reverse=True)

    selected = []
    rejected = set()

    for col in sorted_cols:
        if col in rejected and col not in protected: continue
        selected.append(col)
        
        # استبعاد الـ Features المرتبطة بشدة (باستثناء المحمية)
        highly_correlated = corr_matrix.index[corr_matrix[col] > threshold].tolist()
        for other in highly_correlated:
            if other != col and other not in protected:
                rejected.add(other)

    return selected

def mrmr_selection(df_features: pd.DataFrame, target: pd.Series, n_features: int = 20, protected: set = None) -> List[str]:
    protected = protected or set()
    cols = df_features.columns.tolist()
    n_sel = min(n_features, len(cols))

    # Vectorized Correlation Matrix
    corr_matrix = df_features.corr(method='spearman').abs()
    relevance = df_features.apply(lambda x: abs(x.corr(target, method='spearman'))).fillna(0).to_dict()

    selected = [p for p in cols if p in protected][:n_sel]
    remaining = set(cols) - set(selected)

    if not remaining: return selected

    # إذا لم يكن هناك محميات، اختر الأقوى أولاً
    if not selected:
        first = max(remaining, key=lambda c: relevance[c])
        selected.append(first)
        remaining.discard(first)

    # Greedy Selection (Vectorized Approximation)
    while len(selected) < n_sel and remaining:
        scores = {}
        for f in remaining:
            rel = relevance[f]
            # متوسط الارتباط بين الـ Feature الحالية وكل الـ Features المختارة
            red = corr_matrix.loc[f, selected].mean() if selected else 0.0
            scores[f] = rel - red

        best = max(scores, key=scores.get)
        selected.append(best)
        remaining.discard(best)

    print(f"  ✅ mRMR: {len(cols)} → {len(selected)} features")
    return selected
