"""
Shim للتوافق مع الاستيرادات القديمة.

التنفيذ الحقيقي لـ enrich_bars_with_intrabar (تيكات MBO) موجود في modules.tick_intrabar_slices
حتى لا يُستبدل هذا الملف بالخطأ بنسخة intrabar_mbp_microstructure.
"""

from modules.tick_intrabar_slices import enrich_bars_with_intrabar

__all__ = ["enrich_bars_with_intrabar"]
