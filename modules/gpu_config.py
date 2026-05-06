import os
import multiprocessing
import subprocess
import numpy as np

# ── Safe CPU Detection (Container-Aware) ──────────────────────────
# استكشاف الأنوية الحقيقية المتاحة حتى داخل بيئات Docker/Cloud
try:
    N_CPU_CORES = len(os.sched_getaffinity(0))
except AttributeError:
    N_CPU_CORES = os.cpu_count() or 4

try:
    _MAX_AUTO_WORKERS = int(os.environ.get('QUANTSYSTEM_MAX_AUTO_WORKERS', '64'))
except ValueError:
    _MAX_AUTO_WORKERS = 64
_MAX_AUTO_WORKERS = max(1, _MAX_AUTO_WORKERS)
N_WORKERS = max(1, min(N_CPU_CORES - 2, _MAX_AUTO_WORKERS))  # حجز نواتين للنظام

# ── Global State ──────────────────────────────────────────────────
GPU_AVAILABLE    = False
GPU_NAME         = 'CPU'
GPU_MEMORY_GB    = 0.0

def _query_nvidia_smi_gpus() -> list[dict]:
    """يرجع قائمة بالكروت من nvidia-smi مع إجمالي الذاكرة لكل كرت."""
    try:
        proc = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=name,memory.total',
                '--format=csv,noheader,nounits',
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if proc.returncode != 0:
            return []

        gpus = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            first = line.split(',', 1)
            name = first[0].strip() if first else 'NVIDIA GPU'
            mem_mb = float(first[1].strip()) if len(first) > 1 else 0.0
            gpus.append({
                'name': name or 'NVIDIA GPU',
                'memory_gb': round(mem_mb / 1024.0, 1),
            })
        return gpus
    except Exception:
        return []

def detect_gpu() -> dict:
    """
    يكتشف الـ GPU المتاح ويرجع معلوماته بأمان تام
    مع دعم NVIDIA RTX PRO 6000 و V100
    """
    global GPU_AVAILABLE, GPU_NAME, GPU_MEMORY_GB

    info = {
        'available':   False,
        'name':        'CPU only',
        'memory_gb':   0.0,
        'memory_total_gb': 0.0,
        'n_gpus':      0,
        'framework':   None,
        'catboost_device': 'CPU',
        'tf_device':   '/CPU:0',
    }

    if os.environ.get('QUANTSYSTEM_SKIP_GPU_DETECT', '').strip() == '1':
        return info

    # 1. TensorFlow GPU Check
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass

            # استخراج تفاصيل الكارت الأول
            try:
                gpu_details = tf.config.experimental.get_device_details(gpus[0])
                name = gpu_details.get('device_name', 'Unknown Nvidia GPU')
            except Exception:
                name = 'Nvidia GPU'

            info['available']       = True
            info['name']            = name
            info['n_gpus']          = len(gpus)
            info['framework']       = 'tensorflow'
            info['catboost_device'] = 'GPU'
            info['tf_device']       = '/GPU:0'

            # TensorFlow لا يعطينا دائمًا إجمالي VRAM بشكل موثوق،
            # لذلك نعتمد على nvidia-smi أولًا لقراءة السعة الحقيقية لكل كرت.
            smi_gpus = _query_nvidia_smi_gpus()
            if smi_gpus:
                info['memory_gb'] = smi_gpus[0]['memory_gb']
                info['memory_total_gb'] = round(sum(g['memory_gb'] for g in smi_gpus), 1)
            else:
                # Fallback Estimate
                if 'A40' in name.upper():
                    info['memory_gb'] = 45.0
                elif 'RTX' in name.upper() or 'PRO' in name.upper() or 'A100' in name.upper():
                    info['memory_gb'] = 80.0
                elif 'V100' in name.upper():
                    info['memory_gb'] = 32.0
                elif 'T4' in name.upper():
                    info['memory_gb'] = 16.0
                else:
                    info['memory_gb'] = 12.0
                info['memory_total_gb'] = round(info['memory_gb'] * max(len(gpus), 1), 1)

            GPU_AVAILABLE  = True
            GPU_NAME       = name
            GPU_MEMORY_GB  = info['memory_gb']
            return info
    except Exception:
        pass

    # 2. NVIDIA SMI Fallback (safer than importing torch on some headless/macOS setups)
    try:
        smi_gpus = _query_nvidia_smi_gpus()
        if smi_gpus:
            first = smi_gpus[0]
            name = first['name']
            mem_gb = first['memory_gb']
            info['available'] = True
            info['name'] = name or 'NVIDIA GPU'
            info['memory_gb'] = mem_gb
            info['memory_total_gb'] = round(sum(g['memory_gb'] for g in smi_gpus), 1)
            info['n_gpus'] = len(smi_gpus)
            info['framework'] = 'nvidia-smi'
            info['catboost_device'] = 'GPU'
            info['tf_device'] = '/GPU:0'

            GPU_AVAILABLE = True
            GPU_NAME = info['name']
            GPU_MEMORY_GB = info['memory_gb']
            return info
    except Exception:
        pass

    # 3. PyTorch Fallback (opt-in because some environments abort on import)
    if os.environ.get('QUANTSYSTEM_ENABLE_TORCH_GPU_DETECT', '').strip() == '1':
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                mem  = torch.cuda.get_device_properties(0).total_memory

                info['available']       = True
                info['name']            = name
                info['memory_gb']       = round(mem / (1024**3), 1)
                info['memory_total_gb'] = round((mem / (1024**3)) * max(torch.cuda.device_count(), 1), 1)
                info['n_gpus']          = torch.cuda.device_count()
                info['framework']       = 'torch'
                info['catboost_device'] = 'GPU'

                GPU_AVAILABLE  = True
                GPU_NAME       = name
                GPU_MEMORY_GB  = info['memory_gb']
        except Exception:
            pass

    return info

def get_optimal_batch_size(base_batch: int = 32, seq_len: int = 50, n_features: int = 37) -> int:
    """
    يحسب أفضل batch_size بناءً على الـ VRAM المتاح.
    يتجنب التضخم المفرط الذي يدمر دقة النموذج (Generalization).
    """
    if not GPU_AVAILABLE or GPU_MEMORY_GB <= 0:
        return base_batch

    # القواعد الآمنة لتدريب النماذج المالية المعقدة (Transformers/LSTMs)
    # لا نعتمد فقط على حجم المدخلات، بل نترك مساحة لـ (Gradients + Optimizer States + Attention Maps)
    if GPU_MEMORY_GB >= 80.0:    # A100 / RTX 6000 Ada
        optimal = 512
    elif GPU_MEMORY_GB >= 24.0:  # RTX 3090 / 4090 / V100
        optimal = 256
    elif GPU_MEMORY_GB >= 15.0:  # T4 / RTX 4080
        optimal = 128
    elif GPU_MEMORY_GB >= 8.0:   # Entry Level GPUs
        optimal = 64
    else:
        optimal = 32

    return optimal

def get_multiprocessing_workers(target_workers: int = None) -> int:
    """يرجع عدد الـ workers المأمون لمنع الـ CPU Thrashing"""
    if target_workers:
        return max(1, min(target_workers, N_CPU_CORES - 1))
    return N_WORKERS

def print_gpu_report():
    info = detect_gpu()
    sep = "-" * 50
    print("\n" + sep)
    print("Hardware & GPU Configuration:")
    if info["available"]:
        print(f'   GPU:       {info["name"]}')
        total_vram = info.get("memory_total_gb", 0.0)
        if total_vram and info.get("n_gpus", 0) > 1:
            print(f'   VRAM:      {info["memory_gb"]} GB per GPU | Total: {total_vram} GB')
        else:
            print(f'   VRAM:      {info["memory_gb"]} GB')
        print(f'   GPUs:      {info["n_gpus"]}')
        print(f'   Framework: {info["framework"]}')
        batch = get_optimal_batch_size()
        print(f"   Optimal Batch Size: {batch}")
    else:
        print("   GPU: not available - running on CPU")

    print(f"   CPU Cores: {N_CPU_CORES} (Available)")
    print(f"   MP Workers:{N_WORKERS}")
    print(sep + "\n")
    return info

# نتجنب استكشاف العتاد أثناء الاستيراد حتى لا يفرض TensorFlow side effects
_GPU_INFO = None
