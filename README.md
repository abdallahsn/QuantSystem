# QuantSystem V19

QuantSystem V19 هو مشروع **quantitative AI trading research system** مبني حول:

1. `prepare_training_data.py`
   - يحول بيانات السوق الخام `MBO/MBP` إلى dataset جاهز للتدريب
   - يبني features
   - يبني labels
   - يحفظ `LOB tensors`

2. `train_v19.py`
   - يدرب pipeline ثلاثي المراحل بشكل آمن ضد الـ leakage
   - Stage 1: `CatBoost + Regime meta-features`
   - Stage 2: `OOF DeepLOB visual embeddings`
   - Stage 3: `MetaLearner LSTM`

### Recommended Operational Phases

للتشغيل العملي المبسط، يُفضّل تقسيم المشروع إلى 3 مراحل واضحة:

1. `stage1_refinery.py`
   - يبني dataset التدريب من الخام
   - يحفظ artifact جديدًا مبنيًا على `sharded parquet + manifest + checkpoints`
   - يعمل افتراضيًا على streaming/shards بدل full-load
   - يستخدم `rules` كـ default للـ regime metadata مع coarse sampling لتقليل زمن الـ stage1

2. `stage2_catboost.py`
   - يدرب `CatBoost + Regime`
   - يحفظ `meta_features_oof_v19.npy`
   - يحفظ `catboost_advisor_v19.cbm`

3. `stage3_train.py`
   - يدرب `MetaLearner`
   - يعتمد على نواتج المرحلة الثانية
   - يستخدم الـ visual embeddings من الكاش إن وُجدت، وإلا يكمل بأصفار

3. `backtest_v19.py`
   - يشغل causal replay backtest على dataset V19 الجاهز

4. `walkforward_v19.py`
   - ينفذ walk-forward evaluation من raw market data
   - يعيد بناء train/test folds
   - يطبق release gates

5. `shadow_v19.py` و `paper_v19.py`
   - لتشغيل طبقات التشغيل غير الحي:
   - shadow mode
   - paper mode
   - rollout control logic بدون broker integration


## Architecture

### 1. Raw Data Layer
- `MBO`: market-by-order / trades / add / cancel
- `MBP10`: top-10 order book snapshots

### 2. Feature Layer
يتم استخراج:
- microstructure features
- order book features
- context / rolling / session features
- autoencoder embeddings
- `LOB tensors` للـ CNN

### 3. Model Layer
- `CatBoost advisor`
- `Regime classifier`
- `DeepLOB CNN`
- `MetaLearner LSTM`

### Regime Defaults In This Version
- `stage1_refinery.py` لم يعد يستخدم `Wasserstein` كمسار افتراضي على كل الصفوف.
- الافتراضي الآن:
  - `regime_mode=rules`
  - `regime_stride=50`
  - `deterministic_stage1=true`
- هذا يعني أن stage1 يبني `regime metadata` على surface أخف، ثم يوسعها على كامل الصفوف بدل تشغيل مصنف regime ثقيل على كل trade row.
- إذا أردت `Wasserstein`, شغّله يدويًا فقط كـ `research mode` وليس كمسار إنتاج افتراضي.

### 4. Evaluation Layer
- causal backtest
- walk-forward validation
- monitoring + drift
- release gates


## Project Structure

```text
QuantSystem V19/
├── README.md
├── requirements.txt
├── prepare_training_data.py
├── train_v19.py
├── predict_v19.py
├── backtest_v19.py
├── walkforward_v19.py
├── shadow_v19.py
├── paper_v19.py
├── monitor_v19.py
├── configs/
│   └── v19/
│       ├── defaults.yaml
│       └── release_gates.yaml
└── modules/
    ├── labels_v19.py
    ├── feature_factory_v19.py
    ├── preprocessing_v19.py
    ├── oof_stacking.py
    ├── logging_v19.py
    ├── monitoring_v19.py
    ├── failsafe_v19.py
    ├── manifest_v19.py
    └── ...
```


## Recommended Environment

### Python
- يوصى بـ `Python 3.10` أو `Python 3.11`
- `Python 3.12` مدعوم أيضًا إذا كنت ستثبّت TensorFlow الحديث عبر `pip`
- بيئة `venv` كافية ومفضّلة؛ `conda` اختياري وليس مطلوبًا

### Base Dependencies
الموجودة في [requirements.txt](/Users/abdallah/Downloads/QS_FINAL/requirements.txt):
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `matplotlib`
- `plotly`
- `openpyxl`
- `pyyaml`
- `pyarrow`
- `catboost`
- `tensorflow`
- `tqdm`

### Common Optional Dependencies
بعض أجزاء المشروع تستخدم أو تستفيد من:
- `jupyterlab`
- `ipykernel`
- `hmmlearn`

إذا كنت ستعمل على سيرفر خارجي مع Jupyter Notebook فالأفضل تثبيت:

```bash
pip install -r requirements.txt
pip install jupyterlab ipykernel hmmlearn
```


## Quick Start

### 1. تجهيز البيانات

```bash
python stage1_refinery.py --mbo mbo2.csv --mbp mbp2.csv --output outputs_v19 --label_mode v19 --chunk_rows 2000000 --mbo_workers 32 --mbp_workers 32
```

السلوك الافتراضي المهم في النسخة الحالية:
- `stage1` صار pipeline sharded/resumable بدل `CSV` واحد في النهاية
- `MBO` و`MBP` يُعالجان على shards مع warmup boundaries وcheckpoints
- `regime` يعمل افتراضيًا بـ `rules` بدل `Wasserstein`
- `regime` يُحسب على coarse sample ثم يُوسَّع على كامل الصفوف

للتشغيل الكبير يفضّل ترك هذه الافتراضيات كما هي واستخدام `--resume` إذا انقطع التشغيل.

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

ملاحظة:
- `lob_tensors.npy` و`lob_tensor_timestamps.npy` يُبنيان من الـ shards نفسها إذا كان `MBP` موجودًا و`DeepLOB runtime` متاحًا.
- إذا قررت Stage1 تخطي `Step 3e` بسبب الميزانية أو غياب runtime، ستجد السبب في `outputs_v19/lob_build_meta.json`.
- على Windows native، TensorFlow سيعمل غالبًا على `CPU` فقط؛ إذا أردت `GPU` للـ DeepLOB فشغّل التدريب على Linux/WSL2.

### 1b. Regime Research Mode

إذا أردت اختبار `Wasserstein` يدويًا على dataset أصغر أو في تجربة بحثية:

```bash
python3 stage1_refinery.py \
  --mbo /path/to/mbo.csv \
  --mbp /path/to/mbp.csv \
  --output outputs_v19_research \
  --label_mode v19 \
  --regime_mode wasserstein \
  --regime_stride 25 \
  --regime_window 50 \
  --regime_progress_every 25000
```

ملاحظات مهمة:
- `Wasserstein` لم يعد default لأنه أبطأ بكثير على datasets ضخمة.
- `regime_stride` يتحكم بعدد الصفوف المستخدمة لبناء `regime surface` قبل توسيعها على كامل dataset.
- كلما زاد `regime_stride` أصبح stage1 أسرع، لكن surface أدقّتها الزمنية تصبح أخشن.


### 2. CatBoost Stage

```bash
python stage2_catboost.py --data outputs_v19 --output outputs_v19
```

النواتج المهمة:
- `catboost_advisor_v19.cbm`
- `catboost_classes_v19.json`
- `regime_classifier.pkl`
- `meta_features_oof_v19.npy`
- `meta_coverage_v19.npy`


### 3. Final Training Stage

```bash
python3 stage3_train.py \
  --data outputs_v19 \
  --output outputs_v19
```

النواتج المهمة:
- `meta_learner_v19.keras`
- `feature_schema_v19.json`
- `manifest.json`


### 4. Full Pipeline Shortcut

إذا أردت تشغيل كل شيء دفعة واحدة كما في السلوك القديم:

```bash
python3 train_v19.py \
  --data outputs_v19 \
  --output outputs_v19 \
  --phase full
```

`train_v19.py --data outputs_v19` يبحث تلقائيًا عن `lob_tensors.npy` داخل نفس artifact dir، وليس في مجلد الأب.

يمكن أيضًا تشغيل CatBoost فقط أو التدريب فقط من نفس الملف:

```bash
python3 train_v19.py --data outputs_v19 --output outputs_v19 --phase catboost
python3 train_v19.py --data outputs_v19 --output outputs_v19 --phase train
```


### 4b. Diagnostic Verification Commands

هذه الأوامر مفيدة بعد أي تعديل في `train_v19.py` أو `predict_v19.py` أو طبقات
الـ backtest / paper / regime / schema contracts:

```bash
python -m py_compile \
  train_v19.py \
  predict_v19.py \
  backtest_v19.py \
  paper_v19.py \
  modules/decision_policy_v19.py \
  modules/regime_classifier.py \
  modules/slippage_model.py
```

للتحقق التشغيلي النهائي على artifact حقيقي:

```bash
python train_v19.py \
  --data /path/to/artifact_dir \
  --output /path/to/output_dir \
  --phase catboost \
  --catboost_device cpu
```

```bash
python backtest_v19.py \
  --data /path/to/artifact_dir \
  --models /path/to/output_dir \
  --output /path/to/backtest_output \
  --input_scaled \
  --visual_npy /path/to/output_dir/visual_embeddings_v19.npy
```

إذا كان هدفك التحقق من إصلاحات التقرير الأخيرة بالتحديد، راقب هذه الملفات بعد التشغيل:
- `stage1_v19_metrics.json`
- `calibration_report.json`
- `feature_schema_v19.json`
- `meta_learner_v19_history.json`
- `manifest.json`


### 5. Backtest

```bash
python3 backtest_v19.py \
  --data outputs_v19 \
  --models outputs_v19 \
  --output outputs_v19_backtest \
  --input_scaled \
  --visual_npy outputs_v19/visual_embeddings_v19.npy
```

النواتج:
- `backtest_v19_results.csv`
- `backtest_v19_trades.csv`
- `backtest_v19_summary.json`


### 6. Walk-Forward

```bash
python3 walkforward_v19.py \
  --mbo /path/to/mbo.csv \
  --mbp /path/to/mbp.csv \
  --output outputs_v19_walkforward
```

النواتج:
- `fold_XX/`
- `walkforward_summary.json`
- `release_gates_report.json`
- `manifest.json`

## Stage1 Performance Notes

- `stage1` لم يعد ينتظر حتى النهاية ليكتب dataset واحدًا؛ ستظهر shards وcheckpoints أثناء التشغيل.
- أكبر عنق زجاجة تاريخيًا كان `Regime Classification` على كل الصفوف في loop Python. الآن:
  - الإنتاج الافتراضي يستخدم `rules`
  - `Wasserstein` بقي مسارًا بحثيًا فقط
- إذا كنت تتعامل مع عشرات الملايين من الصفوف:
  - ابدأ بـ `chunk_rows=2_000_000`
  - اضبط `mbo_workers` و`mbp_workers` حسب عدد الأنوية الفعلية
  - فعّل `--resume` في السيرفرات الرخيصة أو المعرضة للانقطاع


## Running On Jupyter Notebook On A Remote Server

هذا هو السيناريو المقترح إذا كنت ستشتغل من لابتوبك لكن التشغيل الفعلي على سيرفر خارجي.

### 1. ادخل إلى السيرفر

```bash
ssh user@your-server-ip
```

### 2. انسخ المشروع أو ارفع الملفات

مثلاً:

```bash
git clone <your-repo-url>
cd QS_FINAL
```

أو ارفع المجلد يدوياً ثم:

```bash
cd /path/to/QS_FINAL
```

### 3. أنشئ بيئة افتراضية

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 4. ثبّت المتطلبات

```bash
pip install -r requirements.txt
pip install jupyterlab ipykernel hmmlearn
```

إذا كان السيرفر Linux وفيه NVIDIA GPU وتريد تشغيل `TensorFlow/DeepLOB` على الـ GPU:

```bash
bash install_tf_gpu_cu12.sh
```

### 5. أضف kernel خاص بالمشروع

```bash
python -m ipykernel install --user --name quantsystem-v19 --display-name "Python (QuantSystem V19)"
```

### 6. شغّل Jupyter Lab على السيرفر

```bash
jupyter lab --no-browser --ip=0.0.0.0 --port=8888
```

إذا أردت طريقة أكثر أماناً، استخدم:

```bash
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

### 7. اعمل SSH tunnel من جهازك المحلي

من جهازك المحلي:

```bash
ssh -L 8888:127.0.0.1:8888 user@your-server-ip
```

ثم افتح في المتصفح:

```text
http://127.0.0.1:8888
```

### 8. اختر Kernel الصحيح

داخل Jupyter اختر:

```text
Python (QuantSystem V19)
```


## Recommended Notebook Workflow

داخل Jupyter Notebook، يفضّل تقسيم العمل إلى 4 notebooks:

### 1. `01_prepare_data.ipynb`
يشغل:
- قراءة raw data paths
- `prepare_training_data.py`
- مراجعة التقارير والـ CSV

مثال:

```python
!python3 prepare_training_data.py \
  --mbo /data/mbo.csv \
  --mbp /data/mbp.csv \
  --output outputs_v19 \
  --label_mode v19 \
  --chunk_rows 2000000
```

### 2. `02_train_v19.ipynb`
يشغل:

```python
!python3 train_v19.py \
  --data outputs_v19 \
  --output outputs_v19
```

### 3. `03_backtest_v19.ipynb`
يشغل:

```python
!python3 backtest_v19.py \
  --data outputs_v19 \
  --models outputs_v19 \
  --output outputs_v19_backtest \
  --input_scaled \
  --visual_npy outputs_v19/visual_embeddings_v19.npy
```

ثم:

```python
import json
with open("outputs_v19_backtest/backtest_v19_summary.json") as f:
    summary = json.load(f)
summary
```

### 4. `04_walkforward_v19.ipynb`
يشغل:

```python
!python3 walkforward_v19.py \
  --mbo /data/mbo.csv \
  --mbp /data/mbp.csv \
  --output outputs_v19_walkforward
```


## Direct Python Usage Inside Notebook

إذا كنت لا تريد تشغيل scripts عبر `!python3`، يمكنك استدعاء بعض الدوال مباشرة.

### Training

```python
from train_v19 import run_training_pipeline

summary = run_training_pipeline(
    csv_path="outputs_v19",
    output_dir="outputs_v19",
)
summary
```

### Shadow

```python
from shadow_v19 import run_shadow

summary = run_shadow(
    csv_path="outputs_v19",
    models_dir="outputs_v19",
    output_dir="outputs_v19_shadow",
    input_scaled=True,
)
summary
```

### Paper

```python
from paper_v19 import run_paper

summary = run_paper(
    csv_path="outputs_v19",
    models_dir="outputs_v19",
    output_dir="outputs_v19_paper",
    input_scaled=True,
    run_mode="paper",
)
summary
```


## Outputs You Should Track

### Training
- `feature_schema_v19.json`
- `manifest.json`
- `meta_learner_v19_history.json`
- `stage1_v19_metrics.json`
- `visual_metrics_v19.json`

### Backtest
- `backtest_v19_summary.json`
- `backtest_v19_trades.csv`

### Walk-Forward
- `walkforward_summary.json`
- `release_gates_report.json`

### Monitoring / Shadow / Paper
- `shadow_predictions.jsonl`
- `shadow_outcomes.jsonl`
- `paper_orders.jsonl`
- `paper_fills.jsonl`
- `paper_trades.jsonl`
- `monitoring_summary.json`
- `drift_report.json`
- `alerts.jsonl`


## Recommended Server Specs

الحد الأدنى العملي:
- CPU: 8 vCPU
- RAM: 32 GB
- Disk: SSD

أفضلية للتدريب المريح:
- CPU: 16+ vCPU
- RAM: 64 GB
- GPU: اختياري لكنه مفيد إذا كان `TensorFlow` و`DeepLOB` سيُستخدمان فعلاً


## Common Issues

### 1. TensorFlow غير مثبت
سترى تحذيرات مثل:

```text
TensorFlow غير مثبّت — MetaLearner غير متاح
```

الحل:

```bash
python -m venv .venv
source .venv/bin/activate
bash install_tf_gpu_cu12.sh
```

إذا كنت تريد CPU فقط:

```bash
pip install tensorflow
```

### 2. CatBoost غير مثبت

```bash
pip install -r requirements.txt
```

### 3. مشاكل `OpenMP SHM`
قد تظهر أحياناً على بعض السيرفرات أو الحاويات.

جرّب:

```bash
export OMP_NUM_THREADS=1
export KMP_INIT_AT_FORK=FALSE
```

ثم أعد تشغيل الـ notebook kernel أو الـ shell.

### 4. ملفات Parquet لا تُقرأ

```bash
pip install pyarrow
```


## Current Focus

حالياً المشروع مهيأ بشكل ممتاز لـ:
- offline training
- backtesting
- walk-forward evaluation
- shadow/paper operational layers

وما زلنا **مؤجلين broker/live data integration** لمرحلة لاحقة.


## Suggested First Session On A Remote Server

إذا هذه أول مرة تشغل المشروع على سيرفر خارجي، اتبع هذا الترتيب:

1. فعّل البيئة الافتراضية
2. افتح Jupyter Lab
3. شغّل `01_prepare_data.ipynb`
4. تأكد من وجود `artifact_manifest.json` و`final/features_*.parquet`
5. شغّل `02_train_v19.ipynb`
6. راجع `manifest.json` و`feature_schema_v19.json`
7. شغّل `03_backtest_v19.ipynb`
8. فقط بعد ذلك انتقل إلى `walkforward_v19.py`


## Notes

- المرجع الأحدث والوحيد للمشروع الحالي هو هذا الملف: [README.md](/Users/abdallah/Downloads/QS_FINAL/README.md)
## 🔍 Codex Evaluation Instructions

If you are an AI system reviewing this project (e.g. OpenAI Codex), follow these strict rules:

### 1. Do NOT assume profitability
This is a research system. Any performance must be verified via:
- walk-forward validation
- out-of-sample testing
- realistic backtesting

### 2. Prioritize detecting data leakage
Check:
- feature construction
- label generation
- scaling
- temporal alignment

Any leakage invalidates results.

### 3. Validate labeling logic
Ensure:
- labels use future price path only
- features use only past/current data
- no overlap contamination
- TP / SL / horizon are consistent

### 4. Reject invalid validation
Do NOT accept:
- random train_test_split
- shuffled data
- mixing future with past

Only accept:
- chronological split
- walk-forward
- purged validation

### 5. Evaluate trading metrics, not just ML metrics
Must include:
- profit factor
- max drawdown
- win rate
- expectancy
- number of trades
- stability across time

### 6. Check backtesting realism
Ensure:
- slippage
- fees
- execution constraints
- no impossible fills

### 7. Prefer robustness over performance
A slightly worse but stable model is better than an unstable high-performing one.

### 8. All improvements must be verifiable
Every suggested change must include:
- why
- expected effect
- how to test it
