"""
walk_forward.py — Walk-Forward Validation
═══════════════════════════════════════════════════════════════════════
يتحقق إن الموديل مش overfit بيدرّب على الماضي ويختبر على المستقبل.

الطريقة:
  نقسم الداتا لـ N فترات زمنية متتالية
  كل فترة: يتدرب على اللي قبله — يُختبر على نفسه

مثال (5 folds):
  Fold 1: train [0-20%]  → test [20-40%]
  Fold 2: train [0-40%]  → test [40-60%]
  Fold 3: train [0-60%]  → test [60-80%]
  Fold 4: train [0-80%]  → test [80-100%]

النتيجة: bias_accuracy حقيقية بدون data leakage
═══════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import os, datetime


class WalkForwardValidator:
    """
    Walk-Forward Validation للـ Expansion Bias model.
    """

    def __init__(self, n_splits: int = 4, min_train_pct: float = 0.3):
        self.n_splits       = n_splits
        self.min_train_pct  = min_train_pct

    def split(self, X: np.ndarray) -> list:
        """
        يرجع قائمة من (train_idx, test_idx) زمنية.
        """
        n      = len(X)
        splits = []
        step   = int(n / (self.n_splits + 1))

        for i in range(1, self.n_splits + 1):
            train_end = step * i
            test_end  = min(step * (i + 1), n)
            if train_end < int(n * self.min_train_pct):
                continue
            train_idx = np.arange(0, train_end)
            test_idx  = np.arange(train_end, test_end)
            if len(test_idx) > 0:
                splits.append((train_idx, test_idx))

        return splits

    def validate(self, X: np.ndarray, y_bias: np.ndarray,
                 y_setup: np.ndarray, y_conf: np.ndarray,
                 brain_class, brain_kwargs: dict,
                 epochs: int = 50, batch_size: int = 32,
                 output_dir: str = 'outputs') -> dict:
        """
        يشغّل Walk-Forward كامل ويرجع النتائج.

        Returns:
            {
              'fold_results':   list of per-fold metrics,
              'mean_accuracy':  float,
              'std_accuracy':   float,
              'is_robust':      bool  (mean > 0.60 و std < 0.10)
            }
        """
        os.makedirs(output_dir, exist_ok=True)
        splits = self.split(X)

        if not splits:
            return {'fold_results': [], 'mean_accuracy': 0.0,
                    'std_accuracy': 0.0, 'is_robust': False}

        fold_results = []
        print(f"\n{'='*60}")
        print(f"🔄 Walk-Forward Validation — {len(splits)} folds")
        print(f"{'='*60}")

        for fold_idx, (tr_idx, te_idx) in enumerate(splits, 1):
            print(f"\n  📅 Fold {fold_idx}/{len(splits)}")
            print(f"     Train: {len(tr_idx):,} | Test: {len(te_idx):,}")

            X_tr, X_te     = X[tr_idx],      X[te_idx]
            yb_tr, yb_te   = y_bias[tr_idx], y_bias[te_idx]
            ys_tr, ys_te   = y_setup[tr_idx],y_setup[te_idx]
            yc_tr, yc_te   = y_conf[tr_idx], y_conf[te_idx]

            # بنبني موديل جديد لكل fold
            fold_path = os.path.join(output_dir, f'fold_{fold_idx}.keras')
            kw        = {**brain_kwargs, 'brain_filename': fold_path}

            try:
                brain = brain_class(**kw)
                brain.fit(X_tr, yb_tr, ys_tr, yc_tr,
                          epochs=epochs, batch_size=batch_size,
                          output_dir=output_dir)

                # تقييم على الـ test
                preds    = brain.model.predict(X_te, verbose=0)
                bias_p   = np.argmax(preds['bias_out'], axis=1)
                accuracy = float(np.mean(bias_p == yb_te))

                # توزيع الـ classes
                unique, counts = np.unique(bias_p, return_counts=True)
                dist = {int(u): int(c) for u, c in zip(unique, counts)}

                fold_result = {
                    'fold':       fold_idx,
                    'train_size': len(tr_idx),
                    'test_size':  len(te_idx),
                    'accuracy':   round(accuracy, 4),
                    'pred_dist':  dist,
                }
                fold_results.append(fold_result)

                grade = '✅' if accuracy >= 0.60 else '⚠️' if accuracy >= 0.50 else '❌'
                print(f"     {grade} Bias Accuracy: {accuracy:.2%}")

                # حذف ملف الـ fold عشان توفير مساحة
                if os.path.exists(fold_path):
                    os.remove(fold_path)

            except Exception as e:
                print(f"     ❌ Fold {fold_idx} فشل: {e}")
                continue

        if not fold_results:
            return {'fold_results': [], 'mean_accuracy': 0.0,
                    'std_accuracy': 0.0, 'is_robust': False}

        accs         = [r['accuracy'] for r in fold_results]
        mean_acc     = float(np.mean(accs))
        std_acc      = float(np.std(accs))
        is_robust    = mean_acc >= 0.60 and std_acc < 0.10

        print(f"\n{'='*60}")
        print(f"📊 Walk-Forward Results:")
        print(f"   Mean Accuracy : {mean_acc:.2%}")
        print(f"   Std Accuracy  : {std_acc:.2%}")
        print(f"   {'✅ موثوق — جاهز للتداول' if is_robust else '⚠️ مش موثوق — يحتاج داتا أكتر أو تحسين'}")
        print(f"{'='*60}")

        # حفظ تقرير
        self._save_report(fold_results, mean_acc, std_acc, is_robust, output_dir)

        return {
            'fold_results':  fold_results,
            'mean_accuracy': mean_acc,
            'std_accuracy':  std_acc,
            'is_robust':     is_robust,
        }

    def _save_report(self, fold_results, mean_acc, std_acc,
                     is_robust, output_dir):
        path = os.path.join(output_dir, 'walk_forward_report.txt')
        lines = []
        def L(x=''): lines.append(str(x))

        L(); L('='*60)
        L('🔄 Walk-Forward Validation Report')
        L(f'🕐 {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        L('='*60)
        L()
        for r in fold_results:
            grade = '✅' if r['accuracy'] >= 0.60 else '⚠️' if r['accuracy'] >= 0.50 else '❌'
            L(f"  Fold {r['fold']}: {grade} {r['accuracy']:.2%}  "
              f"(train={r['train_size']:,} | test={r['test_size']:,})")
        L()
        L(f"  Mean : {mean_acc:.2%}")
        L(f"  Std  : {std_acc:.2%}")
        L(f"  {'✅ موثوق' if is_robust else '❌ غير موثوق'}")
        L()
        L('معايير القبول:')
        L('  Mean Accuracy > 60% ✅')
        L('  Std Accuracy  < 10% ✅ (ثبات عبر الفترات)')

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f'  📄 Walk-Forward Report: {path}')
