import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import classification_report, confusion_matrix
from collections import deque

try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    # EarlyStopping و ModelCheckpoint تُحمَّل داخل block التدريب (داخل fit())
    TF_AVAILABLE = True

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        print(f"  ⚡ TensorFlow GPU: {len(gpus)} device(s) detected")
    else:
        print("  ℹ️  TensorFlow: CPU mode")

except ImportError:
    TF_AVAILABLE = False
    print("  ⚠️  TensorFlow غير مثبّت — pip install tensorflow")

BIAS_LABELS  = {0: 'LONG', 1: 'SHORT', 2: 'NEUTRAL'}
SETUP_LABELS = {0: 'Absorption', 1: 'Spoofing', 2: 'OBI', 3: 'Mixed'}

if TF_AVAILABLE:
    class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, warmup_steps=1000):
            super().__init__()
            self.d_model      = tf.cast(d_model, tf.float32)
            self.warmup_steps = warmup_steps

        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            arg1 = tf.math.rsqrt(step + 1e-8)
            arg2 = step * (self.warmup_steps ** -1.5)
            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

        def get_config(self):
            return {'d_model': int(self.d_model.numpy()),
                    'warmup_steps': self.warmup_steps}
else:
    class WarmupCosineDecay:
        def __init__(self, *a, **kw): pass
        def get_config(self): return {}

def make_causal_mask(seq_len):
    return tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

class TransformerQuantitativeBrain_V3:

    def __init__(self, sequence_length=50, num_features=12,
                 brain_filename='outputs/TransformerBrain_V3.keras',
                 d_model=64, num_heads=4, num_layers=3, ff_dim=256,
                 dropout=0.2, confidence_threshold=0.65,
                 embeddings_dim: int = 8): # إضافة بُعد الـ Embeddings

        self.seq_len              = sequence_length
        self.base_num_features    = num_features
        self.embeddings_dim       = embeddings_dim
        # إجمالي عدد الميزات التي ستدخل للنموذج (الميزات الأصلية + الأرقام السحرية)
        self.num_features         = num_features + embeddings_dim 
        
        self.brain_file           = brain_filename
        self.d_model              = d_model
        self.num_heads            = num_heads
        self.num_layers           = num_layers
        self.ff_dim               = ff_dim
        self.dropout_rate         = dropout
        self.confidence_threshold = confidence_threshold
        self.attention_model      = None
        self.model                = None

        if not TF_AVAILABLE:
            return

        if os.path.exists(self.brain_file):
            try:
                self.model = tf.keras.models.load_model(self.brain_file, compile=False)
                self._recompile()
                print(f"[BRAIN] 🧠 تحميل موديل V3: {self.brain_file}")
            except Exception as e:
                print(f"[BRAIN] ⚠️ موديل مكسور ({e}) — بنبني جديد")
                os.remove(self.brain_file)
                self.model = self._build_model()
        else:
            print("[BRAIN] 🧠 بناء Transformer V3 — Expansion Bias...")
            self.model = self._build_model()

    def _recompile(self):
        lr = WarmupCosineDecay(d_model=self.d_model, warmup_steps=500)
        self.model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4),
            loss={
                'bias_out':       'sparse_categorical_crossentropy',
                'setup_out':      'sparse_categorical_crossentropy',
                'confidence_out': 'binary_crossentropy',
            },
            loss_weights={'bias_out': 1.0, 'setup_out': 0.5, 'confidence_out': 0.3},
            metrics={
                'bias_out':  ['accuracy'],
                'setup_out': ['accuracy'],
            }
        )

    def _encoder_block(self, inputs, layer_idx):
        causal   = make_causal_mask(self.seq_len)
        x        = layers.LayerNormalization(epsilon=1e-6)(inputs)
        attn_out, attn_scores = layers.MultiHeadAttention(
            key_dim=self.d_model // self.num_heads,
            num_heads=self.num_heads,
            dropout=self.dropout_rate
        )(x, x, attention_mask=causal, return_attention_scores=True)
        attn_out = layers.Dropout(self.dropout_rate)(attn_out)
        res1 = inputs + attn_out

        x = layers.LayerNormalization(epsilon=1e-6)(res1)
        x = layers.Dense(self.ff_dim, activation='gelu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(inputs.shape[-1])(x)
        x = layers.Dropout(self.dropout_rate)(x)

        survival = 1.0 - (layer_idx / max(self.num_layers, 1)) * 0.2
        return res1 + x * survival, attn_scores

    def _build_model(self):
        inp = layers.Input(shape=(self.seq_len, self.num_features), name='market_seq')

        x = layers.Dense(self.d_model, name='proj')(inp)

        positions = tf.cast(tf.range(self.seq_len), tf.float32)
        dims      = tf.cast(tf.range(self.d_model), tf.float32)
        angles    = positions[:, tf.newaxis] / tf.pow(10000.0, (2*(dims//2)) / self.d_model)
        sin_enc   = tf.sin(angles[:, 0::2])
        cos_enc   = tf.cos(angles[:, 1::2])
        pos_enc   = tf.concat([sin_enc, cos_enc], axis=-1)[:, :self.d_model]
        x         = x + pos_enc[tf.newaxis, :, :]

        last_attn = None
        for i in range(self.num_layers):
            x, last_attn = self._encoder_block(x, layer_idx=i)

        last_tok   = x[:, -1, :]
        global_avg = layers.GlobalAveragePooling1D()(x)
        pooled     = layers.Concatenate()([last_tok, global_avg])

        shared = layers.Dense(128, activation='gelu', name='shared')(pooled)
        shared = layers.Dropout(0.3)(shared)

        b = layers.Dense(64, activation='gelu')(shared)
        b = layers.Dropout(0.2)(b)
        bias_out = layers.Dense(3, activation='softmax', name='bias_out')(b)

        s = layers.Dense(64, activation='gelu')(shared)
        s = layers.Dropout(0.2)(s)
        setup_out = layers.Dense(4, activation='softmax', name='setup_out')(s)

        c = layers.Dense(32, activation='gelu')(shared)
        confidence_out = layers.Dense(1, activation='sigmoid', name='confidence_out')(c)

        main_model = Model(
            inp,
            {'bias_out': bias_out, 'setup_out': setup_out, 'confidence_out': confidence_out},
            name='QuantBrain_V3'
        )

        self.attention_model = Model(
            inp,
            {'bias_out': bias_out, 'attention': last_attn},
            name='AttentionModel_V3'
        )
        
        # الاعتماد على _recompile لتجنب التكرار
        self.model = main_model
        self._recompile()
        self.model.summary()
        return self.model

    def _merge_features(self, X_train: np.ndarray, embeddings: np.ndarray = None) -> np.ndarray:
        """
        تقوم بدمج التتابعات الزمنية (Sequences) مع الـ Embeddings
        تُكرر الـ Embeddings لكل خطوة زمنية في التتابع.
        """
        if embeddings is None:
            # استخدام أصفار إذا لم تتوفر Embeddings
            embeddings = np.zeros((X_train.shape[0], self.embeddings_dim))
            print("  ⚠️ لم يتم تمرير Embeddings لـ Transformer، تم استخدام قيم صفرية مؤقتاً.")
            
        if X_train.shape[0] != embeddings.shape[0]:
             raise ValueError(f"عدم تطابق الأبعاد: X={X_train.shape[0]}, emb={embeddings.shape[0]}")

        # توسيع Embeddings لتطابق بُعد التتابع: (n_samples, seq_len, embeddings_dim)
        emb_expanded = np.repeat(embeddings[:, np.newaxis, :], self.seq_len, axis=1)
        
        # دمج الاثنين معاً
        X_combined = np.concatenate((X_train, emb_expanded), axis=2)
        return X_combined

    def fit(self, X_train, y_bias, y_setup, y_conf, embeddings=None,
            epochs=150, batch_size=32, output_dir='outputs',
            class_weight=None):

        # ✅ FIX: guard عند غياب TF
        if not TF_AVAILABLE:
            print("  ❌ TensorFlow غير مثبَّت — pip install tensorflow")
            return type('H', (), {'history': {}})()

        os.makedirs(output_dir, exist_ok=True)
        brain_path = os.path.join(output_dir, 'TransformerBrain_V3.keras')

        # دمج المدخلات
        X_combined = self._merge_features(X_train, embeddings)

        split     = int(len(X_combined) * 0.8)
        X_tr      = X_combined[:split]
        X_val     = X_combined[split:]
        yb_tr, yb_val = y_bias[:split],  y_bias[split:]
        ys_tr, ys_val = y_setup[:split], y_setup[split:]
        yc_tr, yc_val = y_conf[:split],  y_conf[split:]

        total = len(yb_tr)
        counts = np.bincount(yb_tr, minlength=3)
        weights_map = {k: total / (3 * max(c, 1)) for k, c in enumerate(counts)}
        sample_w = np.array([weights_map[y] for y in yb_tr], dtype=np.float32)
        print(f"  Sample weights: LONG={weights_map[0]:.2f} SHORT={weights_map[1]:.2f} NEUTRAL={weights_map[2]:.2f}")

        # ✅ FIX: import داخل fit() لضمان وجود TF
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
        callbacks = [
            EarlyStopping(monitor='val_bias_out_accuracy', patience=20,
                          mode='max', restore_best_weights=True, verbose=1),
            ModelCheckpoint(brain_path, save_best_only=True,
                            monitor='val_bias_out_accuracy', mode='max', verbose=1),
        ]

        history = self.model.fit(
            X_tr,
            {'bias_out': yb_tr, 'setup_out': ys_tr, 'confidence_out': yc_tr},
            epochs=epochs, batch_size=batch_size,
            # تحديد وزن العينات للـ bias فقط لمنع التسريب للـ outputs الأخرى
            sample_weight={'bias_out': sample_w},
            validation_data=(
                X_val,
                {'bias_out': yb_val, 'setup_out': ys_val, 'confidence_out': yc_val}
            ),
            callbacks=callbacks,
            verbose=1
        )

        self._plot_training_curves(history, output_dir)
        self.generate_full_report(X_combined, y_bias, y_setup, output_dir=output_dir)
        return history

    def predict_session(self, sequence, embeddings=None, n_samples=30):
        """
        تم تسريع الـ MC Dropout عن طريق הـ Vectorization بدل الـ For-Loop
        """
        # تجهيز البيانات للتنبؤ بصف واحد
        seq_2d = sequence[np.newaxis, :, :] 
        if embeddings is not None:
            emb_2d = embeddings[np.newaxis, :]
            inp_combined = self._merge_features(seq_2d, emb_2d)
        else:
             inp_combined = self._merge_features(seq_2d, None)
             
        inp = np.reshape(inp_combined, (1, self.seq_len, self.num_features))
        
        # استنساخ المدخلات لعمل Forward Pass واحد كبير (أسرع 20 مرة)
        inp_tiled = tf.tile(inp, [n_samples, 1, 1])
        
        out = self.model(inp_tiled, training=True) 
        
        bias_probs  = out['bias_out'].numpy()
        setup_probs = out['setup_out'].numpy()
        confs       = out['confidence_out'].numpy().flatten()

        bias_mean  = np.mean(bias_probs,  axis=0)
        setup_mean = np.mean(setup_probs, axis=0)
        conf_mean  = float(np.mean(confs))
        uncertainty= float(np.std(confs))

        bias_idx  = int(np.argmax(bias_mean))
        setup_idx = int(np.argmax(setup_mean))

        return {
            'bias':        BIAS_LABELS[bias_idx],
            'bias_idx':    bias_idx,
            'bias_probs':  bias_mean.tolist(),
            'setup':       SETUP_LABELS[setup_idx],
            'setup_idx':   setup_idx,
            'setup_probs': setup_mean.tolist(),
            'confidence':  round(conf_mean,  4),
            'uncertainty': round(uncertainty, 4),
        }

    def should_trade(self, sequence, embeddings=None):
        result = self.predict_session(sequence, embeddings)
        tradeable = (
            result['confidence']  >= self.confidence_threshold and
            result['uncertainty'] <  0.15 and
            result['bias']        != 'NEUTRAL'
        )
        return tradeable, result

    def feature_attribution(self, sequence, feature_names, embeddings=None, steps=50):
        """
        تم تصحيح الخلل الرياضي: حساب التدرجات بناءً على الفئة المتوقعة وليس مصفوفة الـ Softmax
        """
        # دمج الميزات أولاً لكي تكون متوافقة مع المدخلات
        seq_2d = sequence[np.newaxis, :, :]
        if embeddings is not None:
             emb_2d = embeddings[np.newaxis, :]
             inp_combined = self._merge_features(seq_2d, emb_2d)
        else:
             inp_combined = self._merge_features(seq_2d, None)
             
        # توليد أسماء للميزات المدمجة (في حال تم استدعاؤها للطباعة)
        combined_feature_names = feature_names + [f'emb_{i}' for i in range(self.embeddings_dim)]

        seq_t    = tf.constant(np.reshape(inp_combined, (1, self.seq_len, self.num_features)), dtype=tf.float32)
        baseline = tf.zeros_like(seq_t)
        total_g  = tf.zeros_like(seq_t)
        
        # إيجاد الفئة الأقوى أولاً لتوجيه الـ Gradients إليها
        initial_pred = self.model(seq_t, training=False)
        target_class = tf.argmax(initial_pred['bias_out'][0])

        for step in range(steps):
            alpha  = step / steps
            interp = baseline + alpha * (seq_t - baseline)
            with tf.GradientTape() as tape:
                tape.watch(interp)
                out = self.model(interp, training=False)
                # استهداف احتمالية الفئة المتوقعة فقط
                target_prob = out['bias_out'][0, target_class] 
                
            grads = tape.gradient(target_prob, interp)
            total_g += grads
            
        ig         = (total_g / steps) * (seq_t - baseline)
        importance = tf.reduce_mean(tf.abs(ig[0]), axis=0).numpy()
        sorted_idx = np.argsort(importance)[::-1]
        
        print("\n🔬 Feature Attribution (Bias Head):")
        for i, idx in enumerate(sorted_idx[:6]):
            bar = "█" * int(importance[idx] * 20)
            # تجنب أخطاء Index out of range لو كانت feature_names ناقصة
            name_str = combined_feature_names[idx] if idx < len(combined_feature_names) else f"Feat_{idx}"
            print(f"  {i+1}. {name_str:<28} {bar} ({importance[idx]:.4f})")
        return importance, combined_feature_names

    def generate_full_report(self, X_data, y_bias, y_setup, output_dir='outputs'):
        # X_data هنا هو الـ X_combined الجاهز
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, 'train_results.txt')
        lines = []
        def L(x=''): print(x); lines.append(x)

        L(); L("═"*70)
        L("🧠 [تقرير العقل V3] — Expansion Bias Model")
        L("═"*70)

        split   = int(len(X_data) * 0.8)
        X_val   = X_data[split:]
        yb_val  = y_bias[split:]
        ys_val  = y_setup[split:]

        if len(X_val) > 0:
            preds    = self.model.predict(X_val, verbose=0)
            bias_p   = np.argmax(preds['bias_out'], axis=1)
            setup_p  = np.argmax(preds['setup_out'], axis=1)

            L("\n📊 Bias Accuracy:")
            L(classification_report(yb_val, bias_p, target_names=['LONG','SHORT','NEUTRAL'], zero_division=0))
            
            L("\n📊 Setup Accuracy:")
            try:
                unique_setup = np.unique(np.concatenate([ys_val, setup_p]))
                setup_names  = ['Absorption','Spoofing','OBI','Mixed']
                used_names   = [setup_names[i] for i in unique_setup if i < len(setup_names)]
                L(classification_report(ys_val, setup_p, labels=unique_setup, target_names=used_names, zero_division=0))
            except Exception as e:
                L(f"Setup report skipped: {e}")

        L("═"*70)
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"\n📄 تقرير التدريب: {results_file}")
        self._plot_full_dashboard(X_data, y_bias, output_dir)

    def _plot_training_curves(self, history, output_dir):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor='#0d1117')
        dark = '#161b22'
        metrics = [
            ('bias_out_accuracy', 'val_bias_out_accuracy', 'Bias Accuracy'),
            ('setup_out_accuracy', 'val_setup_out_accuracy', 'Setup Accuracy'),
            ('loss', 'val_loss', 'Total Loss'),
        ]
        colors = ['#00ff88', '#4488ff', '#ff9944']

        for ax, (tr_key, val_key, title), color in zip(axes, metrics, colors):
            if tr_key in history.history:
                ax.plot(history.history[tr_key], color=color, lw=2, label='Train')
            if val_key in history.history:
                ax.plot(history.history[val_key], color=color, lw=2, linestyle='--', label='Val')
            ax.set_title(title, color='#f0c040')
            ax.set_facecolor(dark)
            ax.tick_params(colors='#aaa')
            ax.legend(facecolor=dark, labelcolor='white')
            for sp in ax.spines.values(): sp.set_edgecolor('#333')

        plt.suptitle('🧠 Training Curves V3', color='#f0c040', fontsize=14)
        fig.patch.set_facecolor('#0d1117')
        path = os.path.join(output_dir, 'training_curves.png')
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
        plt.close()
        print(f"✅ منحنيات التدريب: {path}")

    def _plot_full_dashboard(self, X_data, y_bias, output_dir):
        # التأكد من وجود بيانات كافية للرسم
        limit = min(500, len(X_data))
        if limit == 0: return
        
        preds    = self.model.predict(X_data[:limit], verbose=0)
        bias_p   = np.argmax(preds['bias_out'],  axis=1)
        setup_p  = np.argmax(preds['setup_out'], axis=1)
        confs    = preds['confidence_out'].flatten()

        fig = plt.figure(figsize=(20, 12), facecolor='#0d1117')
        gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)
        dark= '#161b22'

        ax1 = fig.add_subplot(gs[0, 0])
        counts = [np.sum(bias_p==i) for i in range(3)]
        ax1.bar(list(BIAS_LABELS.values()), counts, color=['#00ff88','#ff4466','#aaaaaa'], alpha=0.8)
        ax1.set_title('Bias Distribution', color='#f0c040'); ax1.set_facecolor(dark); ax1.tick_params(colors='#aaa')

        ax2 = fig.add_subplot(gs[0, 1])
        counts2 = [np.sum(setup_p==i) for i in range(4)]
        ax2.bar(list(SETUP_LABELS.values()), counts2, color=['#4488ff','#ff9944','#aa44ff','#44ffff'], alpha=0.8)
        ax2.set_title('Setup Distribution', color='#f0c040'); ax2.set_facecolor(dark); ax2.tick_params(colors='#aaa')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=20)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(confs, bins=30, color='#4488ff', alpha=0.8)
        ax3.axvline(self.confidence_threshold, color='yellow', linestyle='--', lw=2)
        ax3.set_title('Confidence Distribution', color='#f0c040'); ax3.set_facecolor(dark); ax3.tick_params(colors='#aaa')

        ax4 = fig.add_subplot(gs[1, 0])
        if len(np.unique(y_bias[:limit])) > 1:
            cm = confusion_matrix(y_bias[:limit], bias_p, labels=[0,1,2])
            ax4.imshow(cm, cmap='Blues')
            ax4.set_xticks([0,1,2]); ax4.set_yticks([0,1,2])
            ax4.set_xticklabels(list(BIAS_LABELS.values()), color='#aaa')
            ax4.set_yticklabels(list(BIAS_LABELS.values()), color='#aaa')
            for i in range(3):
                for j in range(3):
                    ax4.text(j, i, str(cm[i,j]), ha='center', va='center', color='white')
        ax4.set_title('Confusion Matrix — Bias', color='#f0c040'); ax4.set_facecolor(dark)

        ax5 = fig.add_subplot(gs[1, 1])
        lim_scatter = min(200, limit)
        colors5 = ['#00ff88' if b==0 else '#ff4466' if b==1 else '#aaaaaa' for b in bias_p]
        ax5.scatter(range(lim_scatter), confs[:lim_scatter], c=colors5[:lim_scatter], alpha=0.6, s=20)
        ax5.axhline(self.confidence_threshold, color='yellow', linestyle='--', lw=1)
        ax5.set_title('Confidence per Prediction', color='#f0c040'); ax5.set_facecolor(dark); ax5.tick_params(colors='#aaa')

        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off'); ax6.set_facecolor(dark)
        example_bias  = BIAS_LABELS[int(bias_p[0])]  if len(bias_p)  > 0 else 'N/A'
        example_setup = SETUP_LABELS[int(setup_p[0])] if len(setup_p) > 0 else 'N/A'
        example_conf  = f"{confs[0]:.2%}" if len(confs) > 0 else 'N/A'
        ax6.text(0.5, 0.7, "مثال على المخرج:", ha='center', color='#f0c040', fontsize=12, transform=ax6.transAxes)
        ax6.text(0.5, 0.55, f"Bias: {example_bias}", ha='center', color='#00ff88', fontsize=14, transform=ax6.transAxes)
        ax6.text(0.5, 0.40, f"Setup: {example_setup}", ha='center', color='#4488ff', fontsize=12, transform=ax6.transAxes)
        ax6.text(0.5, 0.25, f"Confidence: {example_conf}", ha='center', color='white', fontsize=12, transform=ax6.transAxes)

        for ax in [ax1,ax2,ax3,ax4,ax5]:
            for sp in ax.spines.values(): sp.set_edgecolor('#333')

        plt.suptitle('🧠 QuantBrain V3 — Expansion Bias Dashboard', color='#f0c040', fontsize=16)
        fig.patch.set_facecolor('#0d1117')
        path = os.path.join(output_dir, 'QuantBrain_V3_Dashboard.png')
        plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='#0d1117')
        plt.close()
        print(f"✅ لوحة التحكم: {path}")
