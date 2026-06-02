# QuantSystem V19

QuantSystem V19 هو نظام بحثي/إنتاجي لتجارب التداول الكمي بالذكاء الاصطناعي. الهدف الأساسي ليس إطلاق وعود ربح، بل بناء pipeline قابل للقياس والتحقق يركز على:

- منع تسرب البيانات المستقبلية إلى الميزات أو التدريب.
- بناء labels قابلة للمراجعة زمنياً.
- تدريب نماذج أكثر تعميماً عبر split زمني وwalk-forward.
- تشغيل backtest واقعي قدر الإمكان مع تكلفة، spread، slippage، latency، وقيود تنفيذ.
- تجهيز طبقات shadow/paper/live diagnostics قبل أي broker integration حقيقي.

> ملاحظة مهمة: أي نتيجة تدريب أو backtest داخل هذا المشروع تعتبر نتيجة بحثية حتى تثبت عبر out-of-sample وwalk-forward وقياس حساسية التكاليف. لا تعتمد على accuracy وحدها.

## الصورة العامة

المشروع يحتوي حالياً على مسارين رئيسيين:

1. **Event/Tick Refinery Path**
   - يستخدم `stage1_refinery.py` أو `prepare_training_data.py`.
   - يعالج بيانات MBO/MBP الخام إلى features وlabels وLOB tensors.
   - مناسب لتجارب microstructure/event-level الأصلية.

2. **Day-Trading Hybrid Path**
   - يستخدم `prepare_day_trading.py`.
   - يحول التكات وMBP إلى شموع `5min/15min/30min` مع ميزات intrabar وLOB tensor لكل شمعة.
   - يحافظ قدر الإمكان على نفس واجهة `train_v19.py` حتى يمكن تدريب CatBoost/DeepLOB/MetaLearner على artifact الشموع.

بعد تجهيز البيانات، المسار التدريبي المشترك هو:

```text
Raw MBO/MBP
  -> refinery / day-trading artifact
  -> train_v19.py
  -> predict_v19.py / backtest_v19.py
  -> walkforward_v19.py / shadow_v19.py / paper_v19.py
```

## بنية المشروع

```text
QS_FINAL/
├── README.md
├── requirements.txt
├── configs/
│   └── v19/
├── stage1_refinery.py
├── prepare_training_data.py
├── prepare_day_trading.py
├── train_v19.py
├── stage2_catboost.py
├── stage3_train.py
├── predict_v19.py
├── backtest_v19.py
├── raw_backtest_v19.py
├── walkforward_v19.py
├── shadow_v19.py
├── paper_v19.py
├── live_predictor.py
├── online_learning.py
├── readiness_v19.py
├── tools/
│   └── diagnostics/
│       ├── verify_day_trading_dataset.py
│       ├── find_training_artifacts.py
│       ├── plot_v19_power_dashboard.py
│       └── plot_best_soft_label_lob_heatmap.py
└── modules/
    ├── feature_artifact_v19.py
    ├── feature_factory_v19.py
    ├── labels_v19.py
    ├── soft_label_engine.py
    ├── mc_label_weights.py
    ├── structural_context_labels_v19.py
    ├── tick_intrabar_slices.py
    ├── intrabar_microstructure.py
    ├── intrabar_mbp_microstructure.py
    ├── catboost_brain.py
    ├── deeplob_cnn.py
    ├── meta_learner.py
    ├── decision_policy_v19.py
    ├── slippage_model.py
    ├── failsafe_v19.py
    ├── raw_replay_v19.py
    └── ...
```

## البيئة والمتطلبات

يفضل استخدام Python 3.10 أو 3.11. يمكن تشغيل أجزاء كثيرة على CPU، أما DeepLOB/MetaLearner فيحتاج TensorFlow، وGPU اختياري حسب البيئة.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

المتطلبات الأساسية موجودة في `requirements.txt` وتشمل:

- `numpy`, `pandas`, `scikit-learn`, `scipy`
- `pyarrow`, `pyyaml`, `tqdm`
- `catboost`, `xgboost`
- `tensorflow`
- `matplotlib`, `plotly`, `openpyxl`

لـ Jupyter على سيرفر:

```bash
pip install jupyterlab ipykernel
python -m ipykernel install --user --name quantsystem-v19 --display-name "Python (QuantSystem V19)"
```

## المسار الأول: Event/Tick Refinery

هذا هو المسار الأصلي لبناء artifact من MBO/MBP مع shards وmanifest وLOB tensors.

```bash
python stage1_refinery.py \
  --mbo /path/to/mbo.csv \
  --mbp /path/to/mbp.csv \
  --output outputs_v19 \
  --label_mode v19 \
  --chunk_rows 2000000 \
  --mbo_workers 8 \
  --mbp_workers 8
```

النواتج المهمة:

- `outputs_v19/artifact_manifest.json`
- `outputs_v19/checkpoints/*.json`
- `outputs_v19/normalized/mbo/*.parquet`
- `outputs_v19/normalized/mbp/*.parquet`
- `outputs_v19/features/merged/*.parquet`
- `outputs_v19/final/features_*.parquet`
- `outputs_v19/scaler_params.json`
- `outputs_v19/lob_tensors.npy`
- `outputs_v19/lob_tensor_timestamps.npy`
- `outputs_v19/refinery_report.txt`

الافتراضات الحالية في Stage 1:

- `regime_mode=rules` هو الافتراضي الإنتاجي الأسرع.
- `Wasserstein` يبقى research mode لأنه أبطأ على datasets كبيرة.
- `regime_stride` يقلل تكلفة بناء regime surface ثم يوسعها على كامل الصفوف.
- soft labels يمكن تشغيلها عبر Monte Carlo عند الحاجة، مع `mc_sample_weight` و`label_stability`.

تشغيل research mode للـ Wasserstein:

```bash
python stage1_refinery.py \
  --mbo /path/to/mbo.csv \
  --mbp /path/to/mbp.csv \
  --output outputs_v19_research \
  --label_mode v19 \
  --regime_mode wasserstein \
  --regime_stride 25 \
  --regime_window 50
```

## المسار الثاني: Day-Trading Hybrid Refinery

`prepare_day_trading.py` هو المسار الجديد لتجميع التكات إلى شموع تداول يومي مع ميزات ميكروية داخل الشمعة. الفكرة ليست فقدان معلومات التكات بالكامل، بل تجميعها داخل كل bar:

- OHLCV.
- CVD وorder-flow imbalance.
- absorption/cancel/spoof intrabar slices.
- MBP intrabar telemetry عند توفر MBP10.
- day/session/context features.
- LOB tensors rolling لكل شمعة.
- labels بنظام Event Gate ثم First Barrier Hit.

مثال تشغيل:

```bash
python prepare_day_trading.py \
  --mbo /path/to/mbo_parquet_or_dir \
  --mbp /path/to/mbp10_file_or_dir \
  --output pipeline_day_trading/features \
  --freq 5min \
  --horizon 6 \
  --tp_mult 1.5 \
  --sl_mult 1.0
```

خيارات مهمة:

- `--freq`: يدعم `5min`, `15min`, `30min`.
- `--horizon`: عدد الشموع المستخدمة في label horizon.
- `--tp_mult` و`--sl_mult`: TP/SL كنسبة من ATR.
- `--no_lob`: تخطي LOB tensors إذا أردت تجربة جدولية سريعة.
- `--strict_train_pool`: يضيق `train_event_flag` إلى جلسات نشطة وATR كاف.

النواتج المتوقعة:

- `day_trading_features.parquet`
- `day_trading_manifest.json`
- `lob_tensors.npy` إذا لم تستخدم `--no_lob`
- `lob_tensor_timestamps.npy` إذا تم بناء LOB

فحص artifact الشموع قبل التدريب:

```bash
python -m tools.diagnostics.verify_day_trading_dataset \
  --data pipeline_day_trading/features/day_trading_features.parquet \
  --lob pipeline_day_trading/features/lob_tensors.npy
```

## التدريب

`train_v19.py` هو مدخل التدريب الرئيسي. يقبل إما artifact كامل، manifest، folder فيه `final/`، أو ملف parquet مثل `day_trading_features.parquet`.

### تدريب كامل

```bash
python train_v19.py \
  --data outputs_v19 \
  --output outputs_v19_train \
  --phase full \
  --catboost_device cpu
```

### تدريب Day-Trading

```bash
python train_v19.py \
  --data pipeline_day_trading/features/day_trading_features.parquet \
  --lob pipeline_day_trading/features/lob_tensors.npy \
  --lob_ts pipeline_day_trading/features/lob_tensor_timestamps.npy \
  --output outputs_day_trading_train \
  --phase full \
  --catboost_device cpu
```

### تدريب CatBoost فقط

```bash
python train_v19.py \
  --data outputs_v19 \
  --output outputs_v19_train \
  --phase catboost \
  --catboost_device cpu
```

### تدريب soft label

```bash
python train_v19.py \
  --data outputs_v19 \
  --output outputs_v19_soft \
  --phase catboost \
  --stage1_target soft_label \
  --sample_weight_mode geometric \
  --catboost_device cpu
```

ملاحظات تدريب مهمة:

- split التدريب زمني وليس random.
- يمكن تحديد holdout صريح عبر `--split_time`.
- يمكن تضييق نافذة التدريب والاختبار عبر `--train_days`, `--backtest_days`, `--window_end`.
- `--seq_len` يغير طول تسلسل MetaLearner؛ في day-trading يمكن ضبطه من config أو CLI.
- عند `stage1_target=soft_label` يتعامل CatBoost/XGBoost كـ regression على `soft_label`، ثم يحول الاحتمال إلى `[p, 1-p]` للـ Meta layer.

النواتج المهمة من التدريب:

- `manifest.json`
- `feature_schema_v19.json`
- `catboost_advisor_v19.cbm`
- `catboost_classes_v19.json`
- `meta_features_oof_v19.npy`
- `visual_embeddings_v19.npy`
- `meta_learner_v19.keras`
- `meta_learner_v19_history.json`
- `stage1_v19_metrics.json`
- `calibration_report.json`

## التنبؤ

تشغيل prediction engine في backtest mode:

```bash
python predict_v19.py \
  --models outputs_v19_train \
  --data outputs_v19 \
  --mode backtest \
  --output outputs_v19_predictions \
  --input_scaled
```

`predict_v19.py` يستخدم artifacts التدريب، preprocessor، policy، CatBoost/XGB، DeepLOB embeddings، وMetaLearner عند توفرها. في backtest mode يتم تعطيل confidence head الخاص بالـ Meta لتسريع replay السببي.

## الباك تست السببي

`backtest_v19.py` ينفذ causal replay على نافذة holdout، ويطبق:

- latency rows.
- spread/slippage/cost.
- single-position mode افتراضياً.
- max daily loss.
- Event Gate وDecision Policy.
- OOS guard لمنع اختبار نفس بيانات التدريب دون تصريح واضح.

تشغيل قياسي:

```bash
python backtest_v19.py \
  --data outputs_v19 \
  --models outputs_v19_train \
  --output outputs_v19_backtest \
  --input_scaled \
  --visual_npy outputs_v19_train/visual_embeddings_v19.npy
```

تشغيل Day-Trading مع horizon بالدقائق:

```bash
python backtest_v19.py \
  --data pipeline_day_trading/features/day_trading_features.parquet \
  --models outputs_day_trading_train \
  --output outputs_day_trading_backtest \
  --input_scaled \
  --fixed_horizon 30 \
  --bar_minutes 5
```

خيارات جديدة مهمة:

- `--fixed_horizon`: يبدل replay horizon بالدقائق، مثلاً `30` دقيقة.
- `--bar_minutes`: مدة الشمعة، وإذا لم تمررها يحاول قراءتها من `day_trading_manifest.json`.
- `--long_only`: لا يفتح صفقات SHORT، لكنه يبقي إشاراتها في اللوج مع `trade_skip_reason=long_only_backtest`.
- `--relax_policy_ev`: مفيد للتشخيص إذا كانت policy تمنع كل الصفقات.
- `--policy_min_edge`: override لحد edge الاحتمالي.
- `--skip_event_gate`: للتشخيص فقط، وليس مسار تقييم نهائي.

النواتج:

- `backtest_v19_results.csv`
- `backtest_v19_trades.csv`
- `backtest_v19_summary.json`

راقب في summary:

- `profit_factor`
- `max_drawdown`
- `win_rate`
- `expectancy`
- `number_of_trades`
- `long_only`
- `horizon_alignment`
- `oos_guard`

## Walk-Forward

`walkforward_v19.py` يعيد بناء folds زمنية من raw MBO/MBP، يدرب ويختبر حسب الزمن، ثم يطبق release gates.

```bash
python walkforward_v19.py \
  --mbo /path/to/mbo.csv \
  --mbp /path/to/mbp.csv \
  --output outputs_v19_walkforward
```

النواتج:

- `fold_XX/`
- `walkforward_summary.json`
- `release_gates_report.json`
- `manifest.json`

لا تعتمد على نموذج قبل أن ينجح في walk-forward أو على الأقل out-of-sample زمني واضح.

## Live / Shadow / Paper Layers

هذه الطبقات ليست broker integration كامل، لكنها تساعد في التحضير التشغيلي.

### Shadow

```bash
python shadow_v19.py \
  --data outputs_v19 \
  --models outputs_v19_train \
  --output outputs_v19_shadow \
  --input_scaled
```

### Paper

```bash
python paper_v19.py \
  --data outputs_v19 \
  --models outputs_v19_train \
  --output outputs_v19_paper \
  --input_scaled
```

### Live predictor module

`live_predictor.py` يوفر pipeline لكل bar:

1. Event Gate.
2. Regime Gate.
3. Feature Select.
4. Predict.
5. Confidence threshold.

الاستخدام البرمجي:

```python
from live_predictor import run_bar_pipeline

signal, confidence, debug = run_bar_pipeline(
    df_bar_row=bar,
    models=regime_models,
    ensemble=online_ensemble,
)
```

### Online learning module

`online_learning.py` يحتوي:

- `DriftDetector` باستخدام Page-Hinkley.
- `SlidingWindowTrainer` لإعادة تدريب نافذة متحركة.
- `RegimeConditionalEnsemble` لكل regime.

هذا ما زال طبقة تشغيلية/بحثية ويحتاج حوكمة قوية قبل الإنتاج الحقيقي.

## أدوات مساعدة

البحث عن artifacts داخل المشروع:

```bash
python -m tools.diagnostics.find_training_artifacts /path/to/QS_FINAL
```

رسم dashboard لقوة الإشارة وLOB:

```bash
python -m tools.diagnostics.plot_v19_power_dashboard \
  --pipeline outputs_v19 \
  --out outputs_v19/power_dashboard.png \
  --freq 5min
```

رسم heatmap حول أفضل soft label:

```bash
python -m tools.diagnostics.plot_best_soft_label_lob_heatmap \
  --pipeline outputs_v19 \
  --out outputs_v19/best_soft_label_lob.png
```

فحص readiness:

```bash
python readiness_v19.py
```

## فحوصات سريعة بعد أي تعديل

فحص syntax:

```bash
python -m py_compile \
  prepare_training_data.py \
  prepare_day_trading.py \
  train_v19.py \
  predict_v19.py \
  backtest_v19.py \
  walkforward_v19.py \
  live_predictor.py \
  online_learning.py \
  tools/diagnostics/verify_day_trading_dataset.py
```

فحص artifact day-trading:

```bash
python -m tools.diagnostics.verify_day_trading_dataset \
  --data pipeline_day_trading/features/day_trading_features.parquet \
  --lob pipeline_day_trading/features/lob_tensors.npy
```

تدريب تشخيصي سريع:

```bash
python train_v19.py \
  --data /path/to/artifact_or_parquet \
  --output /path/to/train_out \
  --phase catboost \
  --catboost_device cpu
```

Backtest تشخيصي:

```bash
python backtest_v19.py \
  --data /path/to/artifact_or_parquet \
  --models /path/to/train_out \
  --output /path/to/backtest_out \
  --input_scaled \
  --relax_policy_ev
```

## قواعد السلامة البحثية

قبل اعتبار أي تجربة ناجحة، تحقق من التالي:

- `ts_event` مرتب زمنياً ولا يحتوي duplicates غير مفسرة.
- features لا تستخدم بيانات مستقبلية.
- labels فقط تستخدم future path بعد وقت القرار.
- scaling يتم fit على train فقط ثم يطبق على validation/test.
- split زمني مع embargo عند وجود horizons متداخلة.
- backtest لا يدخل على أسعار مستحيلة ولا يستخدم future candles/order book states.
- النتائج تعرض trading metrics وليس ML metrics فقط.
- يوجد تقرير حساسية للتكاليف: fees/spread/slippage.
- الاختبار يغطي شهور/عقود مختلفة عند توفر البيانات.

## مشاكل شائعة

### TensorFlow غير مثبت

```bash
pip install tensorflow
```

على Linux مع NVIDIA GPU يمكن تجربة سكربتات CUDA الموجودة:

```bash
bash install_tf_gpu_cu12.sh
```

### CatBoost أو XGBoost غير مثبت

```bash
pip install -r requirements.txt
```

### Parquet لا يقرأ

```bash
pip install pyarrow
```

### backtest لا يفتح صفقات

افحص بالتتابع:

- هل Event Gate يمنع كل الصفوف؟
- هل Decision Policy تتطلب EV موجباً بتكاليف عالية؟
- هل `feature_schema_v19.json` مطابق للأعمدة؟
- هل `price` موجود؟ في day-trading يتم ملؤه من `close` إذا كان ناقصاً.
- هل horizon متوافق مع `day_trading_manifest.json`؟

للتشخيص فقط:

```bash
python backtest_v19.py \
  --data /path/to/day_trading_features.parquet \
  --models /path/to/models \
  --output /path/to/backtest_debug \
  --input_scaled \
  --relax_policy_ev \
  --skip_event_gate
```

## مواصفات سيرفر مقترحة

الحد الأدنى العملي:

- 8 vCPU
- 32 GB RAM
- SSD

للتدريب المريح:

- 16+ vCPU
- 64 GB RAM
- GPU اختياري إذا كان DeepLOB/MetaLearner جزءاً من التجربة.

## ترتيب تشغيل مقترح

للمسار الأصلي:

1. شغل `stage1_refinery.py`.
2. راجع `artifact_manifest.json` و`refinery_report.txt`.
3. شغل `train_v19.py --phase catboost`.
4. إذا النتائج مستقرة، شغل `train_v19.py --phase full`.
5. شغل `backtest_v19.py` على holdout.
6. شغل `walkforward_v19.py`.
7. انتقل إلى shadow/paper فقط بعد نجاح الفحوصات.

لمسار day-trading:

1. شغل `prepare_day_trading.py`.
2. شغل `python -m tools.diagnostics.verify_day_trading_dataset`.
3. درب بـ `train_v19.py` مع `--lob` و`--lob_ts` إذا كانت موجودة.
4. شغل `backtest_v19.py` مع horizon متوافق مع manifest.
5. اختبر `--long_only` أو `--fixed_horizon` فقط كتجارب واضحة وموسومة في النتائج.
6. قارن شهرياً، وليس على فترة واحدة فقط.

## مخرجات يجب حفظها مع كل تجربة

- أمر التشغيل الكامل.
- commit أو نسخة الملفات.
- `manifest.json`.
- `feature_schema_v19.json`.
- `stage1_v19_metrics.json`.
- `calibration_report.json`.
- `backtest_v19_summary.json`.
- `backtest_v19_trades.csv`.
- `walkforward_summary.json` إذا توفر.
- ملاحظات عن fees/slippage/spread والـ horizon.

## تعليمات مراجعة AI/Codex

عند مراجعة هذا المشروع:

1. لا تفترض الربحية.
2. ابحث أولاً عن leakage.
3. تحقق من labels وhorizon وTP/SL.
4. ارفض random split في time-series.
5. اطلب walk-forward أو out-of-sample زمني.
6. قيم trading metrics مع الحساسية للتكاليف.
7. فضل تحسينات صغيرة قابلة للتحقق.
8. اربط كل تغيير بأمر verification واضح.
